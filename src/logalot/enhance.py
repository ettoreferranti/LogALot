"""Speech-enhancement front-end — pull voice out of the static before Whisper.

A learned denoiser (Demucs "DNS" model from facebookresearch/denoiser) loaded via
``torch.hub`` — so it needs only ``torch`` (the ``[enhance]`` extra), no extra
package and none of that project's old dependencies. It runs on CPU at ~0.03×
real time, so it never contends with the MLX Metal work.

The enhancer sits between the 16 kHz resample and Whisper: audio in, cleaner
audio out, same length/rate. It is optional and live-toggleable; the dashboard's
per-line ``avg_logprob`` is the A/B yardstick (toggle it and compare on the same
signal). Lazy: importing this module never requires torch; the model loads (and
its code/checkpoint download once) on first use.
"""
from __future__ import annotations


def torch_available() -> bool:
    try:
        import torch  # noqa: F401
        return True
    except ImportError:
        return False


class SpeechEnhancer:
    """Wraps a Demucs DNS denoiser. ``enhance(audio)`` takes/returns a mono
    float32 ndarray at 16 kHz."""

    def __init__(self, model: str = "dns64") -> None:
        self.model_name = model
        self._model = None
        self._torch = None

    def _ensure(self):
        if self._model is None:
            try:
                import torch
            except ImportError as e:
                raise RuntimeError(
                    "enhance needs the [enhance] extra (torch): pip install -e '.[enhance]'"
                ) from e
            print(f"enhance: loading denoiser {self.model_name} …", flush=True)
            self._torch = torch
            self._model = torch.hub.load(
                "facebookresearch/denoiser", self.model_name,
                verbose=False, trust_repo=True,
            ).eval()
            print("enhance: denoiser loaded", flush=True)
        return self._model

    def enhance(self, audio):
        """Denoise a 16 kHz mono float32 ndarray. Output matches the input length
        so downstream timing is unchanged."""
        import numpy as np

        model = self._ensure()
        torch = self._torch
        x = np.ascontiguousarray(audio, dtype=np.float32)
        with torch.no_grad():
            y = model(torch.from_numpy(x)[None, None])[0, 0].cpu().numpy()
        n = len(audio)
        if len(y) >= n:
            return y[:n]
        return np.pad(y, (0, n - len(y)))
