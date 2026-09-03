"""Regression: the Phase-1 summary log crashed a camera that had SUCCEEDED.

OBSERVED ON EC2
LEFT_UP completed its whole Phase-1 pipeline -- engine 2582 frames, 56 unique
gaps, cache 55 local wagons / 2481 frames, Load 55, Damage 55, Door 55/55 in
212.6 s, fusion done, local wagon states written -- and then died in
sequential/camera_features.py at the verbose summary with:

    TypeError: %d format: a real number is required, not dict

The camera was marked FAILED, never reached seal/deliver/release, and the next
camera logged "REPROCESS: no seal found".

ROOT CAUSE
`CacheBuildResult.frames_written` is `Dict[str, Dict[str, int]]` --
`{gw_id -> {camera_id -> n_frames}}` -- and was passed to `%d`.

WHY THE EXISTING TESTS DID NOT CATCH IT
Every stub declared that field as a plain integer. The stubs were wrong, so the
suite exercised a type the real dataclass never produces. A stub that does not
match the real type is worse than no stub: it reports confidence it has not
earned. These tests therefore use a REAL CacheBuildResult, and the stubs
elsewhere were corrected to the real shape.
"""

from __future__ import annotations

import pytest

from core import constants as C
from materializer.wagon_cache_builder import CacheBuildResult
from sequential import camera_features as F
from sequential import evidence as ev
from sequential import local_state_adapter as A

CAM = C.CAMERA_LEFT_UP


def _real_cache_result():
    """A genuine CacheBuildResult shaped the way the real builder returns one."""
    return CacheBuildResult(
        cache_root="/ws/camera_local/LEFT_UP/wagon_cache",
        # The nested map -- the shape that broke %d.
        frames_written={"LEFT_UP_W1": {CAM: 45},
                        "LEFT_UP_W2": {CAM: 47},
                        "LEFT_UP_W3": {CAM: 44}},
        per_camera_total={CAM: 136},
        missing_cameras=[],
        elapsed_seconds=12.5,
    )


def _local_state():
    gaps = [ev.GapObservation(local_gap_id="%s_G%d" % (CAM, i),
                              confirmation_frame=f, first_frame=f - 2,
                              last_frame=f + 2, normalized_position=i * 100.0,
                              max_confidence=0.9)
            for i, f in enumerate((100, 200, 300, 400), start=1)]
    segments = [{"segment_id": ev.SEGMENT_ID_FORMAT % (CAM, i),
                 "segment_index": i, "start_frame": a.confirmation_frame,
                 "end_frame": b.confirmation_frame,
                 "opening_gap": a.local_gap_id, "closing_gap": b.local_gap_id,
                 "canonical": False}
                for i, (a, b) in enumerate(zip(gaps, gaps[1:]), start=1)]
    return A.build_local_state(ev.CameraEvidence(
        camera_id=CAM, status=ev.STATUS_SEALED,
        timing=ev.CameraTiming(fps=15.0, total_frames=1000),
        gaps=gaps, segments=segments))


@pytest.fixture
def wired(monkeypatch):
    """Real CacheBuildResult; models stubbed. The bug was after fusion."""
    from fusion import wagon_state_builder
    from materializer import wagon_cache_builder
    from orchestrator import master_runner

    monkeypatch.setattr(wagon_cache_builder, "build",
                        lambda **kw: _real_cache_result())
    monkeypatch.setattr(master_runner, "load_feature_runner",
                        lambda name: (lambda **kw: {"ok": name}))
    monkeypatch.setattr(wagon_state_builder, "build",
                        lambda **kw: {"LEFT_UP_W1": object()})


# -----------------------------------------------------------------------------
# The exact failure
# -----------------------------------------------------------------------------

def test_the_real_nested_map_is_what_broke_percent_d():
    """Reproduce the TypeError directly, so the cause is documented in code."""
    result = _real_cache_result()
    assert isinstance(result.frames_written, dict)
    assert isinstance(next(iter(result.frames_written.values())), dict)
    with pytest.raises(TypeError, match="a real number is required, not dict"):
        _ = "%d frames" % result.frames_written


def test_verbose_summary_does_not_raise_on_a_real_cache_result(tmp_path, wired,
                                                               capsys):
    """The whole point: verbose=True must survive a genuine result."""
    out = F.run_camera_local(state=_local_state(), camera_id=CAM,
                             video_path="/v.mp4", workspace=str(tmp_path),
                             feat_models_dir="/m",
                             features=["door", "load", "damage"], fps=15.0,
                             verbose=True)
    printed = capsys.readouterr().out
    assert "[SEQ/P1/LEFT_UP]" in printed
    assert "136 frames" in printed, printed
    assert out["cache"]["total_frames"] == 136


def test_the_scalar_comes_from_the_dataclasss_own_accessor():
    """`total_frames()` is CacheBuildResult's method -- not a re-derivation."""
    result = _real_cache_result()
    assert result.total_frames() == 136
    assert result.total_frames() == sum(result.per_camera_total.values())


# -----------------------------------------------------------------------------
# The information is preserved, not discarded to make the log print
# -----------------------------------------------------------------------------

def test_the_per_wagon_map_is_kept_alongside_the_scalar(tmp_path, wired):
    """Requirement: preserve the underlying statistics, do not flatten them."""
    out = F.run_camera_local(state=_local_state(), camera_id=CAM,
                             video_path="/v.mp4", workspace=str(tmp_path),
                             feat_models_dir="/m", features=["door"],
                             fps=15.0, verbose=False)
    cache = out["cache"]
    assert cache["frames_written"] == {"LEFT_UP_W1": {CAM: 45},
                                       "LEFT_UP_W2": {CAM: 47},
                                       "LEFT_UP_W3": {CAM: 44}}
    assert cache["per_camera_total"] == {CAM: 136}
    assert cache["elapsed_seconds"] == 12.5
    assert cache["missing_cameras"] == []


def test_total_frames_is_an_int_and_frames_written_is_not(tmp_path, wired):
    out = F.run_camera_local(state=_local_state(), camera_id=CAM,
                             video_path="/v.mp4", workspace=str(tmp_path),
                             feat_models_dir="/m", features=["door"],
                             fps=15.0, verbose=False)
    assert isinstance(out["cache"]["total_frames"], int)
    assert isinstance(out["cache"]["frames_written"], dict)


# -----------------------------------------------------------------------------
# The camera must reach seal/deliver/release
# -----------------------------------------------------------------------------

def test_a_succeeding_camera_is_not_reported_as_failed(tmp_path, wired):
    """The bug's real cost: a completed camera was marked FAILED and reprocessed.

    `run_camera_local` returning normally is what lets the caller go on to
    write_seal -> deliver_camera -> _release.
    """
    out = F.run_camera_local(state=_local_state(), camera_id=CAM,
                             video_path="/v.mp4", workspace=str(tmp_path),
                             feat_models_dir="/m",
                             features=["door", "load", "damage"], fps=15.0,
                             verbose=True)
    assert out["feature_summary"]["load"] == {"ok": "load"}
    assert out["feature_summary"]["door"] == {"ok": "door"}
    assert out["feature_summary"]["damage"] == {"ok": "damage"}
    assert out["unified"]
    assert "fusion" in out["timings"]


def test_an_empty_cache_result_still_formats(tmp_path, monkeypatch):
    """A camera that cached nothing must log 0, not crash or omit the line."""
    from fusion import wagon_state_builder
    from materializer import wagon_cache_builder
    from orchestrator import master_runner
    monkeypatch.setattr(wagon_cache_builder, "build",
                        lambda **kw: CacheBuildResult(cache_root="/x"))
    monkeypatch.setattr(master_runner, "load_feature_runner",
                        lambda name: (lambda **kw: {}))
    monkeypatch.setattr(wagon_state_builder, "build", lambda **kw: {})
    out = F.run_camera_local(state=_local_state(), camera_id=CAM,
                             video_path="/v.mp4", workspace=str(tmp_path),
                             feat_models_dir="/m", features=["door"],
                             fps=15.0, verbose=True)
    assert out["cache"]["total_frames"] == 0


# -----------------------------------------------------------------------------
# Guard the guard
# -----------------------------------------------------------------------------

def test_no_stub_in_the_suite_declares_frames_written_as_a_scalar():
    """The stubs are what hid this. They must match the real dataclass.

    A stub whose type differs from the real one reports confidence it has not
    earned -- this failure reached EC2 with a green suite for exactly that
    reason.
    """
    import glob
    import re
    offenders = []
    for path in glob.glob("tests/test_*.py"):
        with open(path, encoding="utf-8") as handle:
            for lineno, line in enumerate(handle, start=1):
                # Code only: a docstring discussing the bug is not a stub.
                stripped = line.strip()
                if stripped.startswith("#") or stripped.startswith(("\"", "'")):
                    continue
                if re.search(r"frames_written\s*=\s*\d", line):
                    offenders.append("%s:%d %s" % (path, lineno, line.strip()))
    assert not offenders, (
        "frames_written is Dict[gw_id -> Dict[camera_id -> int]]; these stubs "
        "declare a scalar:\n  " + "\n  ".join(offenders))
