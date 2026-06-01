"""Tests for the audio module's hardware-independent parts: level conversion,
device-name matching, and the ring buffer. Live capture needs the FTDX10 USB
CODEC and is exercised manually."""
import math

import pytest

from logalot.audio import (
    DBFS_FLOOR,
    _norm,
    dbfs_to_pct,
    rms_to_dbfs,
)

np = pytest.importorskip("numpy")  # ring buffer is numpy-backed ([capture] extra)

from logalot.audio import RingBuffer  # noqa: E402


def test_rms_to_dbfs_floor_on_silence():
    assert rms_to_dbfs(0.0) == DBFS_FLOOR
    assert rms_to_dbfs(1e-12) == DBFS_FLOOR


def test_rms_to_dbfs_full_scale():
    # RMS of 1.0 == 0 dBFS; half amplitude ~ -6 dBFS.
    assert rms_to_dbfs(1.0) == pytest.approx(0.0, abs=1e-6)
    assert rms_to_dbfs(0.5) == pytest.approx(-6.02, abs=0.05)


def test_dbfs_to_pct_clamped_and_monotonic():
    assert dbfs_to_pct(DBFS_FLOOR) == 0
    assert dbfs_to_pct(0.0) == 100
    assert dbfs_to_pct(-200.0) == 0
    assert dbfs_to_pct(-20.0) > dbfs_to_pct(-40.0)


def test_norm_collapses_whitespace_and_case():
    # The FTDX10 enumerates as "USB AUDIO  CODEC" (double space).
    assert _norm("USB AUDIO  CODEC") == _norm("USB Audio CODEC")


def test_ring_buffer_basic_read_last():
    rb = RingBuffer(10)
    rb.write(np.arange(4, dtype=np.float32))      # [0 1 2 3]
    out = rb.read_last(3)
    assert list(out) == [1.0, 2.0, 3.0]


def test_ring_buffer_wraps():
    rb = RingBuffer(5)
    rb.write(np.arange(4, dtype=np.float32))      # fill [0 1 2 3]
    rb.write(np.arange(4, 8, dtype=np.float32))   # wrap; last 5 should be 3..7
    out = rb.read_last(5)
    assert list(out) == [3.0, 4.0, 5.0, 6.0, 7.0]


def test_ring_buffer_read_more_than_filled():
    rb = RingBuffer(100)
    rb.write(np.arange(3, dtype=np.float32))
    out = rb.read_last(50)                          # only 3 available
    assert list(out) == [0.0, 1.0, 2.0]


def test_ring_buffer_oversized_write_keeps_tail():
    rb = RingBuffer(4)
    rb.write(np.arange(10, dtype=np.float32))       # bigger than capacity
    out = rb.read_last(4)
    assert list(out) == [6.0, 7.0, 8.0, 9.0]
