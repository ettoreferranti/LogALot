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

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse

from .capture import CatError, RigctldClient


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


def _snapshot(client: RigctldClient) -> dict:
    """One poll, as a JSON-able dict. CAT failure is reported, never raised —
    the dashboard shows 'rig offline' rather than 500-ing."""
    import datetime as dt

    utc = dt.datetime.now(dt.timezone.utc).strftime("%H:%M:%S")
    try:
        st = client.monitor()
    except CatError as e:
        return {"ok": False, "error": str(e), "utc": utc}
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
    }


def create_app(rig_host: str = "127.0.0.1", rig_port: int = 4532) -> FastAPI:
    app = FastAPI(title="LogALot rig monitor")
    # One persistent CAT client for the app's lifetime; reconnects internally.
    client = RigctldClient(rig_host, rig_port)

    @app.get("/api/state")
    def state() -> JSONResponse:
        return JSONResponse(_snapshot(client))

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return _PAGE

    return app


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
  .ptt { display:flex; align-items:center; gap:10px; margin-top:22px; }
  .lamp { width:14px; height:14px; border-radius:50%; background:#2ea043; box-shadow:0 0 10px #2ea04388; }
  .lamp.tx { background:#f85149; box-shadow:0 0 12px #f85149aa; }
  .ptt .txt { font-weight:600; letter-spacing:.05em; }
  .offline { color:#f85149; }
  .muted { color:#7d8590; }
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
    <div class="ptt"><span class="lamp" id="lamp"></span><span class="txt" id="pttxt">—</span>
      <span class="muted" id="status" style="margin-left:auto"></span></div>
  </div>
<script>
async function tick() {
  const $ = id => document.getElementById(id);
  try {
    const r = await fetch('/api/state', {cache:'no-store'});
    const d = await r.json();
    $('clock').textContent = d.utc + ' UTC';
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
</script>
</body></html>
"""
