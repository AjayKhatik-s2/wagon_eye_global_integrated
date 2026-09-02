"""Processing order follows ARRIVAL; the global result does not.

Sequential models a real pipeline: the camera whose video arrives first is
processed first. A fixed RIGHT_UP-first order made the first camera an artifact
of configuration rather than of when its clip landed.

The two properties are separate and both must hold:

    execution order      = arrival order
    final global result  = independent of arrival order

The second is what makes the first safe.
"""

from __future__ import annotations

import os

import pytest

from core import constants as C
from sequential import evidence as ev
from sequential import global_assembly
from sequential.runner import arrival_from_files, camera_order

ALL = list(C.ALL_CAMERAS)          # RIGHT_UP, LEFT_UP, RIGHT_UP_TOP, LEFT_UP_TOP
VP = {c: "/videos/%s.mp4" % c for c in ALL}


# -----------------------------------------------------------------------------
# Arrival order drives execution
# -----------------------------------------------------------------------------

def test_the_specs_first_example():
    """LEFT_UP, RIGHT_UP_TOP, RIGHT_UP, LEFT_UP_TOP arrive in that order."""
    arrival = {"LEFT_UP": 10, "RIGHT_UP_TOP": 20,
               "RIGHT_UP": 30, "LEFT_UP_TOP": 40}
    assert camera_order(VP, arrival=arrival) == [
        "LEFT_UP", "RIGHT_UP_TOP", "RIGHT_UP", "LEFT_UP_TOP"]


def test_the_specs_second_example():
    arrival = {"RIGHT_UP_TOP": 1, "LEFT_UP": 2,
               "LEFT_UP_TOP": 3, "RIGHT_UP": 4}
    assert camera_order(VP, arrival=arrival) == [
        "RIGHT_UP_TOP", "LEFT_UP", "LEFT_UP_TOP", "RIGHT_UP"]


def test_right_up_is_not_privileged():
    """RIGHT_UP goes last when it arrives last -- it is not the default first."""
    arrival = {"LEFT_UP_TOP": 1, "LEFT_UP": 2, "RIGHT_UP_TOP": 3, "RIGHT_UP": 4}
    assert camera_order(VP, arrival=arrival)[-1] == C.CAMERA_RIGHT_UP
    assert camera_order(VP, arrival=arrival)[0] == C.CAMERA_LEFT_UP_TOP


@pytest.mark.parametrize("perm", [
    ["RIGHT_UP", "LEFT_UP", "RIGHT_UP_TOP", "LEFT_UP_TOP"],
    ["LEFT_UP_TOP", "RIGHT_UP", "LEFT_UP", "RIGHT_UP_TOP"],
    ["RIGHT_UP_TOP", "LEFT_UP_TOP", "LEFT_UP", "RIGHT_UP"],
    ["LEFT_UP", "RIGHT_UP_TOP", "LEFT_UP_TOP", "RIGHT_UP"],
])
def test_every_permutation_is_honoured(perm):
    arrival = {cam: i for i, cam in enumerate(perm)}
    assert camera_order(VP, arrival=arrival) == perm


# -----------------------------------------------------------------------------
# Determinism
# -----------------------------------------------------------------------------

def test_no_signal_falls_back_to_configuration():
    """A directory has no arrival order; the result must still be total."""
    assert camera_order(VP) == ALL
    assert camera_order(VP, arrival={}) == ALL


def test_ties_break_by_configuration_not_by_chance():
    """Four clips written in the same second must still order reproducibly."""
    arrival = {c: 5 for c in ALL}
    assert camera_order(VP, arrival=arrival) == ALL


def test_an_unknown_signal_sorts_last_not_first():
    """None must not read as 'earliest'."""
    arrival = {"LEFT_UP": 100, "RIGHT_UP": None}
    order = camera_order({"LEFT_UP": "a", "RIGHT_UP": "b"}, arrival=arrival)
    assert order == ["LEFT_UP", "RIGHT_UP"]


def test_repeated_calls_agree():
    arrival = {"LEFT_UP": 2, "RIGHT_UP": 1, "RIGHT_UP_TOP": 4,
               "LEFT_UP_TOP": 3}
    assert camera_order(VP, arrival=arrival) == camera_order(VP,
                                                             arrival=arrival)


def test_absent_cameras_are_excluded():
    partial = {"LEFT_UP": "a", "RIGHT_UP": "b"}
    order = camera_order(partial, arrival={"LEFT_UP": 1, "RIGHT_UP": 2})
    assert order == ["LEFT_UP", "RIGHT_UP"]


# -----------------------------------------------------------------------------
# The local signal
# -----------------------------------------------------------------------------

def test_arrival_from_files_uses_mtime(tmp_path):
    paths = {}
    for i, cam in enumerate(ALL):
        p = tmp_path / ("%s.mp4" % cam)
        p.write_bytes(b"x")
        os.utime(p, (1_000_000 + i * 10, 1_000_000 + i * 10))
        paths[cam] = str(p)
    arrival = arrival_from_files(paths)
    assert camera_order(paths, arrival=arrival) == ALL

    # Reverse the mtimes -> reverse the processing order.
    for i, cam in enumerate(reversed(ALL)):
        os.utime(paths[cam], (2_000_000 + i * 10, 2_000_000 + i * 10))
    assert camera_order(paths, arrival=arrival_from_files(paths)) == \
        list(reversed(ALL))


def test_a_missing_file_yields_no_signal_rather_than_a_fake_one(tmp_path):
    real = tmp_path / "a.mp4"
    real.write_bytes(b"x")
    arrival = arrival_from_files({"LEFT_UP": str(real),
                                  "RIGHT_UP": str(tmp_path / "gone.mp4")})
    assert arrival["LEFT_UP"] is not None
    assert arrival["RIGHT_UP"] is None


# -----------------------------------------------------------------------------
# The global result must NOT depend on arrival order
# -----------------------------------------------------------------------------

def test_phase2_restores_evidence_in_configuration_order(tmp_path, monkeypatch):
    """`sealed_cameras` iterates C.ALL_CAMERAS, whatever order Phase 1 ran.

    This is what makes arrival-order execution safe: the master camera, the
    canonical roster and every global value are read back in a fixed order, so
    permuting arrival cannot permute the global result.
    """
    seen = {}

    def _fake_load_seal(workspace, camera_id):
        seen.setdefault("order", []).append(camera_id)
        return {"status": ev.STATUS_SEALED}

    monkeypatch.setattr(ev, "load_seal", _fake_load_seal)
    out = ev.sealed_cameras(str(tmp_path))
    assert out == ALL
    assert seen["order"] == ALL


def test_the_required_set_is_order_independent():
    """All four are required regardless of which arrived first."""
    required = list(global_assembly.required_cameras())
    assert sorted(required) == sorted(ALL)


def test_camera_order_does_not_touch_the_required_set():
    """Reordering execution must not change what the gate demands."""
    before = list(global_assembly.required_cameras())
    camera_order(VP, arrival={"LEFT_UP_TOP": 1, "RIGHT_UP": 2,
                              "LEFT_UP": 3, "RIGHT_UP_TOP": 4})
    assert list(global_assembly.required_cameras()) == before
