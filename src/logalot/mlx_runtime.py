"""In-process MLX runtime — no LM Studio, no Ollama, no server on the side.

Everything runs inside this one Python process: models are pulled from the
HuggingFace ``mlx-community`` hub on first use and cached on disk by the HF
libraries, then loaded into memory once and kept for the process lifetime.
Mirrors the pattern proven in code-rally's ``mlx_runtime``.

**The one rule:** MLX uses a process-wide default Metal command stream, so two
inferences on different OS threads collide on the same command-buffer state
(silent corruption / crashes). Every MLX call in LogALot — Whisper transcription
*and* the parse LLM — must therefore funnel through the single shared worker
thread here. They run strictly serially. At human logging cadence that is fine;
a contest pileup is the scenario that would stress it (CLAUDE.md open Q3).

Optional dependency: this module imports cleanly without ``mlx`` installed. The
``mlx``/``mlx_lm``/``mlx_whisper`` imports happen lazily, only when something
actually loads or runs a model, and raise a clear error if the ``[asr]``/
``[parse]`` extras are missing. Apple Silicon only.
"""
from __future__ import annotations

import asyncio
import concurrent.futures
import os
import queue
import threading
from typing import Any, Callable, TypeVar

T = TypeVar("T")


class _MetalWorker:
    """Single shared worker thread for ALL MLX/Metal work in the process — do not
    parallelise it; that serialisation is load-bearing (see module docstring).

    It is a **daemon** thread on purpose: a ``ThreadPoolExecutor`` uses non-daemon
    workers and joins them at interpreter exit, so a Ctrl+C while a model is
    downloading or an inference is running would hang the process forever. A
    daemon thread lets the process exit immediately (an in-flight HF download just
    resumes next launch).
    """

    def __init__(self) -> None:
        self._q: queue.Queue = queue.Queue()
        self._thread = threading.Thread(target=self._run, name="mlx", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while True:
            fn, fut = self._q.get()
            if not fut.set_running_or_notify_cancel():
                continue
            try:
                fut.set_result(fn())
            except BaseException as exc:  # propagate to the caller's future
                fut.set_exception(exc)

    def submit(self, fn: Callable[[], T]) -> "concurrent.futures.Future[T]":
        fut: concurrent.futures.Future = concurrent.futures.Future()
        self._q.put((fn, fut))
        return fut


_MLX_WORKER = _MetalWorker()

# Default models, overridable by env so we never hard-code a single choice.
# Parse: Qwen2.5 instruct follows JSON instructions well (code-rally precedent).
# Apertus (ETH/EPFL, Swiss) is the natural HB9 alternative once an MLX quant is
# handy — set LOGALOT_PARSE_MODEL to swap. ASR: large-v3-turbo trades a little
# accuracy for the throughput we need near real time on weak SSB.
DEFAULT_PARSE_MODEL = os.environ.get(
    "LOGALOT_PARSE_MODEL", "mlx-community/Qwen2.5-7B-Instruct-4bit"
)
DEFAULT_ASR_MODEL = os.environ.get(
    "LOGALOT_ASR_MODEL", "mlx-community/whisper-large-v3-turbo"
)


def is_available() -> bool:
    """True if MLX can be imported (does not load any model)."""
    try:
        import mlx.core  # noqa: F401
        return True
    except ImportError:
        return False


def run_blocking(fn: Callable[..., T], *args: Any, **kwargs: Any) -> T:
    """Run ``fn`` on the shared MLX thread and block for the result. Use from
    sync/threaded callers (the capture daemon's worker threads)."""
    return _MLX_WORKER.submit(lambda: fn(*args, **kwargs)).result()


async def run_async(fn: Callable[..., T], *args: Any, **kwargs: Any) -> T:
    """Awaitable version for the async web layer — keeps heavy MLX work off the
    event loop while still serialising through the one Metal thread."""
    return await asyncio.wrap_future(_MLX_WORKER.submit(lambda: fn(*args, **kwargs)))


class LLMRuntime:
    """Cached mlx-lm model+tokenizer for one model path.

    Loaded once per unique path and kept for the process lifetime; repeat
    callers on the same model share it and skip the multi-GB reload. Generation
    runs on the shared Metal thread via :func:`run_blocking`.
    """

    _cache: dict[str, "LLMRuntime"] = {}

    def __init__(self, model_path: str, max_tokens: int = 256) -> None:
        try:
            from mlx_lm import generate, load
        except ImportError as e:
            raise RuntimeError(
                "parse needs the [parse] extra (Apple Silicon): "
                "pip install -e '.[parse]'"
            ) from e
        self.model_path = model_path
        self.max_tokens = max_tokens
        # Mistral-Small tokenizers ship a buggy regex in transformers; the
        # upstream-recommended fix_mistral_regex flag corrects tokenization (else
        # numbers/callsigns can be split wrong). Only pass it to Mistral.
        tok_cfg = {"fix_mistral_regex": True} if "mistral" in model_path.lower() else {}
        # Visible confirmation of which model is actually in memory (a dropdown
        # swap loads here on the next parse; first use of an uncached model
        # downloads, otherwise it's read from the HF cache).
        print(f"parse: loading {model_path} …", flush=True)
        self._model, self._tok = load(model_path, tokenizer_config=tok_cfg)
        print(f"parse: loaded {model_path}", flush=True)
        self._generate = generate

    def _gen_sync(self, messages: list[dict], prefix: str = "") -> str:
        prompt = self._tok.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
        if prefix:
            # "Put words in the model's mouth": the assistant turn starts with
            # this text, so the continuation must extend it (e.g. JSON priming).
            prompt = prompt + prefix
        out = self._generate(
            self._model, self._tok, prompt=prompt,
            max_tokens=self.max_tokens, verbose=False,
        )
        return prefix + out

    def generate(self, messages: list[dict], prefix: str = "") -> str:
        """Chat-complete ``messages`` (system/user dicts) to text, serialised on
        the shared Metal thread. ``prefix`` primes the assistant's reply."""
        return run_blocking(self._gen_sync, messages, prefix)


def get_llm(model_path: str | None = None, max_tokens: int = 256) -> LLMRuntime:
    """Cached :class:`LLMRuntime` for ``model_path`` (defaults to
    :data:`DEFAULT_PARSE_MODEL`). First call for a path loads the model."""
    path = model_path or DEFAULT_PARSE_MODEL
    rt = LLMRuntime._cache.get(path)
    if rt is None:
        rt = LLMRuntime(path, max_tokens=max_tokens)
        LLMRuntime._cache[path] = rt
    return rt
