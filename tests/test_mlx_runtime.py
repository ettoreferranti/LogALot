"""Tests for the shared MLX runtime plumbing — the parts that work without MLX
installed: optional-import safety, the single-thread executor, and async routing.
The actual model load/generate needs Apple-Silicon hardware and is manual."""
import threading

import pytest

from logalot import mlx_runtime


def test_imports_without_mlx():
    # The module must import cleanly and report availability as a bool either way.
    assert isinstance(mlx_runtime.is_available(), bool)


def test_default_models_are_mlx_community():
    assert "mlx-community/" in mlx_runtime.DEFAULT_PARSE_MODEL
    assert "mlx-community/" in mlx_runtime.DEFAULT_ASR_MODEL


def test_run_blocking_returns_value():
    assert mlx_runtime.run_blocking(lambda x: x * 2, 21) == 42


def test_run_blocking_uses_single_named_mlx_thread():
    # Every MLX call must land on the one shared worker thread (Metal serialisation).
    seen = set()
    for _ in range(5):
        seen.add(mlx_runtime.run_blocking(lambda: threading.current_thread().name))
    assert len(seen) == 1
    assert next(iter(seen)).startswith("mlx")


def test_run_async_routes_to_mlx_thread():
    import asyncio

    name = asyncio.run(mlx_runtime.run_async(lambda: threading.current_thread().name))
    assert name.startswith("mlx")


def test_get_llm_without_mlx_raises_clear_error():
    if mlx_runtime.is_available():
        pytest.skip("MLX is installed; the missing-dependency path can't be tested")
    with pytest.raises(RuntimeError, match=r"\[parse\] extra"):
        mlx_runtime.get_llm("mlx-community/does-not-matter")
