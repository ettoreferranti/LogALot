"""Tests for the cat-probe helpers that don't need a rig or the rigctld binary."""
from logalot import cat


def test_rigctld_command_default_port_omits_t():
    cmd = cat.rigctld_command("/dev/cu.SLAB_USBtoUART5", "1042", 38400, 4532)
    assert cmd == "rigctld -m 1042 -r /dev/cu.SLAB_USBtoUART5 -s 38400"
    assert " -t " not in cmd          # default TCP port is implicit


def test_rigctld_command_nondefault_port_includes_t():
    cmd = cat.rigctld_command("/dev/cu.X", "1042", 9600, 5000)
    assert cmd.endswith("-s 9600 -t 5000")


def test_candidate_devices_returns_sorted_unique_list():
    devs = cat.candidate_devices()
    assert isinstance(devs, list)
    assert devs == sorted(set(devs))   # de-duplicated, stable order
