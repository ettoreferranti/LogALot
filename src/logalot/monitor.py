"""Step A — live rig monitor (web dashboard).

A thin local FastAPI app that polls the CAT client and shows what the rig is
doing: frequency, band, mode, S-meter and PTT. Read-only by design — it never
touches the canonical log; it just proves the rig → UI half of the chain works
and gives us a surface to bolt audio (Step B) and transcript (Step C) onto.

Local-first: no CDN assets. The page is a single self-contained document with a
tiny vanilla-JS poller, so it works with the network cable unplugged.

Behind the ``[ui]`` extra (fastapi/uvicorn); import this module only when those
are installed (the stdlib core and the M0 CLI never import it).
"""
from __future__ import annotations

import asyncio
import json
import queue
from dataclasses import asdict

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse

from .audio import AudioCapture, AudioError, dbfs_to_pct
from .capture import CatError, RigctldClient
from .feed import TranscriptEntry, TranscriptFeed
from .mlx_runtime import DEFAULT_PARSE_MODEL, is_available

# Whisper supports many languages; this is a pragmatic EU-centric shortlist plus
# "auto" (per-segment detection). Operators can still pass any code via the CLI.
_LANGUAGES = ["auto", "en", "de", "fr", "it", "es", "nl", "pt", "ru", "pl", "sv", "cs"]

# Preset mlx-community parse models, small→large. The running default is added in
# the settings payload if it isn't already here.
_PARSE_MODELS = [
    "mlx-community/Qwen2.5-1.5B-Instruct-4bit",
    "mlx-community/Qwen2.5-3B-Instruct-4bit",
    "mlx-community/Qwen2.5-7B-Instruct-4bit",
]


def smeter_label(dbs9: int | None) -> str:
    """dB-relative-to-S9 -> ham S-meter label. S9 == 0 dB, 6 dB per S-unit."""
    if dbs9 is None:
        return "—"
    if dbs9 > 0:
        return f"S9+{dbs9}dB"
    s = max(0, min(9, round(9 + dbs9 / 6)))
    return f"S{s}"


def smeter_pct(dbs9: int | None) -> int:
    """Map the S-meter onto 0–100 for a bar. S0 (~-54 dB) → 0, S9+30 → 100."""
    if dbs9 is None:
        return 0
    lo, hi = -54.0, 30.0
    return int(max(0, min(100, (dbs9 - lo) / (hi - lo) * 100)))


def _audio_block(audio: AudioCapture | None) -> dict:
    """Audio half of the snapshot. Absent/failed capture is reported as off,
    never raised."""
    if audio is None or not audio.is_running():
        return {"audio_on": False, "audio_device": "—"}
    db = audio.level_dbfs()
    return {
        "audio_on": True,
        "audio_device": audio.device_label,
        "level_dbfs": round(db, 1),
        "level_pct": dbfs_to_pct(db),
    }


def _snapshot(client: RigctldClient, audio: AudioCapture | None = None) -> dict:
    """One poll, as a JSON-able dict. CAT failure is reported, never raised —
    the dashboard shows 'rig offline' rather than 500-ing."""
    import datetime as dt

    utc = dt.datetime.now(dt.timezone.utc).strftime("%H:%M:%S")
    audio_block = _audio_block(audio)
    try:
        st = client.monitor()
    except CatError as e:
        return {"ok": False, "error": str(e), "utc": utc, **audio_block}
    return {
        "ok": True,
        "utc": utc,
        "freq_mhz": st.freq_mhz,
        "freq_display": f"{st.freq_mhz:.6f}" if st.freq_mhz is not None else "—",
        "band": st.band or "—",
        "mode": st.mode or "—",
        "raw_mode": st.raw_mode or "—",
        "smeter_db": st.strength_dbs9,
        "smeter_label": smeter_label(st.strength_dbs9),
        "smeter_pct": smeter_pct(st.strength_dbs9),
        "ptt": bool(st.ptt),
        **audio_block,
    }


def create_app(rig_host: str = "127.0.0.1", rig_port: int = 4532,
               audio_device: str | None = None, enable_audio: bool = True,
               enable_asr: bool = True, enable_parse: bool = True,
               vad_threshold: float = -45.0, min_logprob: float = -1.0,
               language: str = "auto", translate: bool = False) -> FastAPI:
    app = FastAPI(title="LogALot rig monitor")
    # One persistent CAT client for the app's lifetime; reconnects internally.
    client = RigctldClient(rig_host, rig_port)

    # Best-effort audio capture: if the device or [capture] extra is missing the
    # dashboard simply shows 'audio off' — it never blocks the rig view.
    audio: AudioCapture | None = None
    if enable_audio:
        audio = AudioCapture(audio_device)
        try:
            audio.start()
            print(f"audio: capturing from {audio.device_label!r} @ {audio.samplerate} Hz")
        except AudioError as e:
            print(f"audio: off ({e})")
            audio = None

    # Transcript feed needs audio + the MLX [asr] extra. Without them the panel
    # shows why; the rest of the dashboard is unaffected.
    feed: TranscriptFeed | None = None
    asr_status = "off"
    if enable_asr and audio is not None and is_available():
        from .asr import WhisperTranscriber

        lang = None if language in (None, "", "auto") else language
        parser = None
        if enable_parse:
            try:
                import mlx_lm  # noqa: F401

                from .parse import MLXParser
                parser = MLXParser()
            except ImportError:
                print("parse: install the [parse] extra for the candidate panel")
        feed = TranscriptFeed(audio, WhisperTranscriber(language=lang, translate=translate),
                              cat=client, parser=parser, vad_threshold_dbfs=vad_threshold,
                              min_logprob=min_logprob)
        feed.start()
        asr_status = "listening"
        print(f"asr model:   {feed.transcriber.model_path}")
        if parser is not None:
            print(f"parse model: {parser.model_path}")
        else:
            print("parse model: off (candidate panel disabled)")
    elif enable_asr and audio is not None:
        asr_status = "ASR unavailable — install the [asr] extra (Apple Silicon)"
        print(f"asr: {asr_status}")

    @app.get("/api/state")
    def state() -> JSONResponse:
        return JSONResponse(_snapshot(client, audio))

    @app.get("/api/transcript")
    async def transcript(request: Request) -> StreamingResponse:
        return StreamingResponse(
            _transcript_events(request, feed, asr_status),
            media_type="text/event-stream",
        )

    @app.get("/api/settings")
    def get_settings() -> JSONResponse:
        if feed is None:
            return JSONResponse({"ok": False})
        models = list(_PARSE_MODELS)
        current = feed.settings().get("parse_model")
        if current and current not in models:
            models.insert(0, current)
        return JSONResponse({
            "ok": True,
            "settings": feed.settings(),
            "languages": _LANGUAGES,
            "parse_models": models,
            "has_parser": feed.parser is not None,
        })

    @app.post("/api/settings")
    async def post_settings(request: Request) -> JSONResponse:
        if feed is None:
            return JSONResponse({"ok": False, "error": "feed not running"}, status_code=409)
        body = await request.json()
        if "vad_threshold" in body:
            feed.set_vad_threshold(float(body["vad_threshold"]))
        if "min_logprob" in body:
            feed.set_min_logprob(float(body["min_logprob"]))
        if "language" in body:
            feed.set_language(body["language"])
        if "translate" in body:
            feed.set_translate(body["translate"])
        if "parse_model" in body:
            feed.set_parse_model(body["parse_model"])
        return JSONResponse({"ok": True, "settings": feed.settings()})

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return _PAGE

    @app.on_event("shutdown")
    def _shutdown() -> None:
        if feed is not None:
            feed.stop()
        if audio is not None:
            audio.stop()

    return app


def _sse(payload: dict) -> str:
    return f"data: {json.dumps(payload)}\n\n"


def _event(msg) -> str:
    """Wrap a feed message as an SSE 'data:' line. Transcript entries get a kind
    tag; candidate dicts already carry one."""
    if isinstance(msg, TranscriptEntry):
        return _sse({"kind": "transcript", **asdict(msg)})
    return _sse(msg)


async def _transcript_events(request: Request, feed: TranscriptFeed | None, asr_status: str):
    """SSE generator: replay recent history (transcript + last candidate), then
    push new messages as they land. Keepalive comments so a disconnect is noticed."""
    if feed is None:
        yield _sse({"info": asr_status})
        return
    q = feed.subscribe()
    loop = asyncio.get_event_loop()
    try:
        for e in list(feed.entries):
            yield _event(e)
        if feed.last_candidate is not None:
            yield _event(feed.last_candidate)
        while True:
            if await request.is_disconnected():
                break
            try:
                msg = await loop.run_in_executor(None, lambda: q.get(timeout=1.0))
            except queue.Empty:
                yield ": keepalive\n\n"
                continue
            yield _event(msg)
    finally:
        feed.unsubscribe(q)


# Single self-contained page: dark panel, big frequency readout, mode/band chips,
# an S-meter bar, a TX/RX lamp, and a UTC clock. Polls /api/state ~1.3x/sec.
_PAGE = """<!doctype html>
<html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>LogALot · rig monitor</title>
<style>
  :root { color-scheme: dark; }
  body { margin:0; font:16px/1.4 system-ui,sans-serif; background:#0d1117; color:#e6edf3;
         display:flex; min-height:100vh; align-items:center; justify-content:center; }
  .panel { width:min(560px,92vw); background:#161b22; border:1px solid #30363d;
           border-radius:14px; padding:28px 32px; box-shadow:0 10px 40px #0008; }
  .top { display:flex; justify-content:space-between; align-items:baseline; }
  h1 { font-size:14px; letter-spacing:.12em; text-transform:uppercase; color:#7d8590; margin:0; font-weight:600; }
  .clock { font-variant-numeric:tabular-nums; color:#7d8590; font-size:14px; }
  .freq { font-size:54px; font-weight:700; font-variant-numeric:tabular-nums; margin:14px 0 2px;
          letter-spacing:.01em; }
  .freq small { font-size:22px; color:#7d8590; font-weight:500; }
  .chips { display:flex; gap:8px; margin:10px 0 22px; flex-wrap:wrap; }
  .chip { background:#21262d; border:1px solid #30363d; border-radius:999px; padding:4px 12px;
          font-size:13px; color:#c9d1d9; }
  .chip b { color:#e6edf3; }
  .meter { margin:18px 0 6px; }
  .meter .lbl { display:flex; justify-content:space-between; font-size:13px; color:#7d8590; margin-bottom:6px; }
  .bar { height:14px; background:#21262d; border-radius:8px; overflow:hidden; border:1px solid #30363d; }
  .fill { height:100%; width:0; background:linear-gradient(90deg,#2ea043,#d29922 70%,#f85149);
          transition:width .25s ease; }
  .afill { background:linear-gradient(90deg,#1f6feb,#2ea043 60%,#d29922); transition:width .08s linear; }
  .ptt { display:flex; align-items:center; gap:10px; margin-top:22px; }
  .lamp { width:14px; height:14px; border-radius:50%; background:#2ea043; box-shadow:0 0 10px #2ea04388; }
  .lamp.tx { background:#f85149; box-shadow:0 0 12px #f85149aa; }
  .ptt .txt { font-weight:600; letter-spacing:.05em; }
  .offline { color:#f85149; }
  .muted { color:#7d8590; }
  .tx { margin-top:24px; border-top:1px solid #30363d; padding-top:16px; }
  .txhead { display:flex; justify-content:space-between; font-size:13px; color:#7d8590; margin-bottom:10px; }
  .txlog { max-height:240px; overflow-y:auto; display:flex; flex-direction:column; gap:8px;
           font:13px/1.45 ui-monospace,SFMono-Regular,Menlo,monospace; }
  .row { display:grid; grid-template-columns:64px 1fr auto; gap:10px; align-items:baseline; }
  .rtime { color:#7d8590; font-variant-numeric:tabular-nums; }
  .rtext { color:#e6edf3; word-break:break-word; }
  .rmeta { color:#6e7681; font-size:11px; white-space:nowrap; }
  .txlog .empty { color:#6e7681; }
  .cand { margin-top:20px; border-top:1px solid #30363d; padding-top:16px; }
  .candbody { display:grid; grid-template-columns:auto 1fr; gap:5px 14px; font-size:14px; align-items:baseline; }
  .ck { color:#7d8590; text-transform:uppercase; font-size:11px; letter-spacing:.06em; }
  .cv { color:#e6edf3; word-break:break-word; }
  .cv.call { font-weight:700; font-size:20px; font-variant-numeric:tabular-nums; }
  .cv.call.ok { color:#2ea043; } .cv.call.review { color:#d29922; }
  .badge { font-size:10px; border:1px solid #30363d; border-radius:999px; padding:1px 7px;
           color:#7d8590; vertical-align:middle; margin-left:8px; }
  .controls { margin-top:24px; border-top:1px solid #30363d; padding-top:16px;
              display:grid; grid-template-columns:auto 1fr auto; gap:12px 14px; align-items:center; font-size:13px; }
  .controls > label { color:#7d8590; }
  .controls .val { color:#e6edf3; font-variant-numeric:tabular-nums; text-align:right; min-width:62px; }
  .controls input[type=range] { width:100%; accent-color:#1f6feb; }
  .controls select { width:100%; background:#0d1117; color:#e6edf3; border:1px solid #30363d;
                     border-radius:6px; padding:5px 8px; font:inherit; }
</style></head>
<body>
  <div class="panel">
    <div class="top"><h1>LogALot · rig monitor</h1><span class="clock" id="clock">—</span></div>
    <div class="freq" id="freq">—<small> MHz</small></div>
    <div class="chips">
      <span class="chip">mode <b id="mode">—</b></span>
      <span class="chip">band <b id="band">—</b></span>
      <span class="chip">cat <b id="raw">—</b></span>
    </div>
    <div class="meter">
      <div class="lbl"><span>S-meter</span><span id="smeter">—</span></div>
      <div class="bar"><div class="fill" id="fill"></div></div>
    </div>
    <div class="meter">
      <div class="lbl"><span>Audio RX <span class="muted" id="adev"></span></span><span id="alevel">—</span></div>
      <div class="bar"><div class="fill afill" id="afill"></div></div>
    </div>
    <div class="ptt"><span class="lamp" id="lamp"></span><span class="txt" id="pttxt">—</span>
      <span class="muted" id="status" style="margin-left:auto"></span></div>
    <div class="tx">
      <div class="txhead"><span>Transcript · remote operator (RX)</span><span id="asr">—</span></div>
      <div class="txlog" id="txlog"><span class="empty">waiting for speech…</span></div>
    </div>
    <div class="cand" id="cand" hidden>
      <div class="txhead"><span>Candidate QSO<span class="badge">advisory · not logged</span></span></div>
      <div class="candbody" id="candbody"></div>
    </div>
    <div class="controls" id="controls" hidden>
      <label for="c_lang">Language</label>
      <select id="c_lang"></select><span class="val" id="c_lang_v"></span>
      <label for="c_xlate">Translate→EN</label>
      <input type="checkbox" id="c_xlate" style="justify-self:start; width:18px; height:18px; accent-color:#1f6feb;">
      <span class="val" id="c_xlate_v"></span>
      <label for="c_vad">VAD gate</label>
      <input type="range" id="c_vad" min="-60" max="-10" step="1"><span class="val" id="c_vad_v"></span>
      <label for="c_lp">Min logprob</label>
      <input type="range" id="c_lp" min="-3" max="0" step="0.1"><span class="val" id="c_lp_v"></span>
      <label for="c_model" id="c_model_l">LLM model</label>
      <select id="c_model"></select><span class="val"></span>
      <label>ASR model</label>
      <span id="c_asr" class="muted" style="grid-column:2 / -1;"></span>
    </div>
  </div>
<script>
async function tick() {
  const $ = id => document.getElementById(id);
  try {
    const r = await fetch('/api/state', {cache:'no-store'});
    const d = await r.json();
    $('clock').textContent = d.utc + ' UTC';
    // Audio is independent of CAT — render it whether or not the rig answers.
    if (d.audio_on) {
      $('adev').textContent = '· ' + d.audio_device;
      $('alevel').textContent = d.level_dbfs + ' dBFS';
      $('afill').style.width = d.level_pct + '%';
    } else {
      $('adev').textContent = '· off'; $('alevel').textContent = '—';
      $('afill').style.width = '0%';
    }
    if (!d.ok) {
      $('status').innerHTML = '<span class="offline">rig offline</span>';
      $('freq').innerHTML = '—<small> MHz</small>';
      $('lamp').className = 'lamp'; $('pttxt').textContent = '—';
      return;
    }
    $('status').textContent = 'connected';
    $('freq').innerHTML = d.freq_display + '<small> MHz</small>';
    $('mode').textContent = d.mode; $('band').textContent = d.band; $('raw').textContent = d.raw_mode;
    $('smeter').textContent = d.smeter_label + (d.smeter_db!=null ? ' ('+d.smeter_db+' dB)' : '');
    $('fill').style.width = d.smeter_pct + '%';
    $('lamp').className = 'lamp' + (d.ptt ? ' tx' : '');
    $('pttxt').textContent = d.ptt ? 'TRANSMITTING' : 'RECEIVING';
  } catch (e) {
    $('status').innerHTML = '<span class="offline">no server</span>';
  }
}
tick(); setInterval(tick, 750);

// Transcript via Server-Sent Events (push, not poll).
function esc(s){ const d=document.createElement('div'); d.textContent=s; return d.innerHTML; }
const es = new EventSource('/api/transcript');
es.onmessage = (ev) => {
  const d = JSON.parse(ev.data);
  if (d.info !== undefined) { document.getElementById('asr').textContent = d.info; return; }
  if (d.kind === 'candidate') { renderCandidate(d); return; }
  // transcript line
  const asr = document.getElementById('asr'), log = document.getElementById('txlog');
  asr.textContent = 'listening';
  const empty = log.querySelector('.empty'); if (empty) empty.remove();
  const conf = (d.avg_logprob != null) ? ' · lp ' + d.avg_logprob.toFixed(2) : '';
  const row = document.createElement('div'); row.className = 'row';
  row.innerHTML = '<span class="rtime">'+d.utc+'</span><span class="rtext">'+esc(d.text)+
                  '</span><span class="rmeta">'+d.dur_s+'s'+conf+'</span>';
  log.appendChild(row);
  log.scrollTop = log.scrollHeight;
};
es.onerror = () => { document.getElementById('asr').textContent = 'reconnecting…'; };

// Live controls — apply without restarting (settings are read per segment).
function postSettings(body){
  fetch('/api/settings', {method:'POST', headers:{'Content-Type':'application/json'},
                          body:JSON.stringify(body)});
}
async function loadControls(){
  let d; try { d = await (await fetch('/api/settings')).json(); } catch(e){ return; }
  if (!d.ok) return;                       // no feed (audio/ASR off)
  const s = d.settings, $ = id => document.getElementById(id);
  $('controls').hidden = false;

  const lang = $('c_lang');
  lang.innerHTML = d.languages.map(l => '<option'+(l===s.language?' selected':'')+'>'+l+'</option>').join('');
  $('c_lang_v').textContent = s.language;
  lang.onchange = () => { postSettings({language: lang.value}); $('c_lang_v').textContent = lang.value; };

  const xl = $('c_xlate'); xl.checked = !!s.translate;
  $('c_xlate_v').textContent = s.translate ? 'on' : 'off';
  xl.onchange = () => { postSettings({translate: xl.checked}); $('c_xlate_v').textContent = xl.checked ? 'on' : 'off'; };

  const vad = $('c_vad'); vad.value = s.vad_threshold;
  $('c_vad_v').textContent = s.vad_threshold + ' dBFS';
  vad.oninput  = () => $('c_vad_v').textContent = vad.value + ' dBFS';
  vad.onchange = () => postSettings({vad_threshold: parseFloat(vad.value)});

  const lp = $('c_lp'); lp.value = s.min_logprob;
  $('c_lp_v').textContent = (+s.min_logprob).toFixed(1);
  lp.oninput  = () => $('c_lp_v').textContent = (+lp.value).toFixed(1);
  lp.onchange = () => postSettings({min_logprob: parseFloat(lp.value)});

  $('c_asr').textContent = (s.asr_model || '').replace('mlx-community/', '');

  const ml = $('c_model');
  if (d.has_parser){
    ml.innerHTML = d.parse_models.map(m =>
      '<option value="'+m+'"'+(m===s.parse_model?' selected':'')+'>'+esc(m.replace('mlx-community/',''))+'</option>').join('');
    ml.onchange = () => postSettings({parse_model: ml.value});
  } else {
    ml.disabled = true; $('c_model_l').textContent = 'LLM model (off)';
  }
}
loadControls();

const CAND_FIELDS = {name:'name', qth:'qth', rst_sent:'rst s', rst_rcvd:'rst r',
                     gridsquare:'grid', comment:'cmt'};
function renderCandidate(d){
  const panel = document.getElementById('cand'), body = document.getElementById('candbody');
  const rows = [];
  if (d.call){
    const aff = d.affixes ? ' <span class="muted">/'+esc(d.affixes.join('/'))+'</span>' : '';
    const ctry = d.call_country ? ' <span class="muted">· '+esc(d.call_country)+'</span>' : '';
    rows.push('<div class="ck">call</div><div class="cv call '+(d.call_confidence||'')+'">'+
              esc(d.call)+aff+ctry+'</div>');
  } else if (d.call_tentative){
    rows.push('<div class="ck">call?</div><div class="cv muted">'+esc(d.call_tentative)+
              ' <span class="badge">unconfirmed</span></div>');
  }
  for (const k in CAND_FIELDS) if (d[k])
    rows.push('<div class="ck">'+CAND_FIELDS[k]+'</div><div class="cv">'+esc(d[k])+'</div>');
  const rig = [];
  if (d.freq_mhz != null) rig.push(d.freq_mhz.toFixed(3)+' MHz');
  if (d.band) rig.push(d.band);
  if (d.mode) rig.push(d.mode);
  if (rig.length) rows.push('<div class="ck">rig</div><div class="cv muted">'+esc(rig.join(' · '))+
                            ' · '+d.qso_date+' '+d.time_on+' UTC</div>');
  if (d.source_text) rows.push('<div class="ck">heard</div><div class="cv muted">'+esc(d.source_text)+'</div>');
  body.innerHTML = rows.join('');
  panel.hidden = false;
}
</script>
</body></html>
"""
