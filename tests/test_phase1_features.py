"""Phase 1 runs Batch's Stage 2/3/4 on one camera's own wagon windows.

The point of these tests is that no feature logic is reimplemented: the cache
builder, the three processors and the fusion builder are Batch's, reached
through the same `load_feature_runner` dispatch Batch uses. What differs is the
state they receive.

Two behaviours are copied from Batch on purpose and pinned here, because getting
either wrong would produce a camera-local verdict Batch would not:

  * LOAD runs to completion BEFORE damage -- damage reads the sibling load JSON
    to drop floor_damage on LOADED wagons.
  * a disabled feature gets a DISABLED_BY_USER sentinel, not silence.
"""

from __future__ import annotations

import json
import os

import pytest

from core import constants as C
from sequential import camera_features as F
from sequential import evidence as ev
from sequential import local_state_adapter as A

CAM = C.CAMERA_LEFT_UP


def _evidence(frames=(100, 200, 300)):
    gaps = [ev.GapObservation(local_gap_id="%s_G%d" % (CAM, i),
                              confirmation_frame=f, first_frame=f - 2,
                              last_frame=f + 2, normalized_position=i * 100.0,
                              max_confidence=0.9)
            for i, f in enumerate(frames, start=1)]
    segments = [{"segment_id": ev.SEGMENT_ID_FORMAT % (CAM, i),
                 "segment_index": i, "start_frame": a.confirmation_frame,
                 "end_frame": b.confirmation_frame,
                 "opening_gap": a.local_gap_id, "closing_gap": b.local_gap_id,
                 "canonical": False}
                for i, (a, b) in enumerate(zip(gaps, gaps[1:]), start=1)]
    return ev.CameraEvidence(
        camera_id=CAM, status=ev.STATUS_SEALED,
        timing=ev.CameraTiming(fps=15.0, total_frames=1000),
        gaps=gaps, segments=segments,
        engine_result={"video_info": {"width": 960, "height": 540}})


@pytest.fixture
def local_state():
    return A.build_local_state(_evidence())


# -----------------------------------------------------------------------------
# Isolation
# -----------------------------------------------------------------------------

def test_phase1_writes_only_under_camera_local(tmp_path):
    paths = F.paths_for(str(tmp_path), CAM)
    for key in ("cache_root", "states_root", "evidence_root", "tracking_path"):
        assert "%s/%s" % (F.CAMERA_LOCAL_DIRNAME, CAM) in paths[key]


def test_phase1_does_not_touch_the_canonical_trees(tmp_path, local_state,
                                                   monkeypatch):
    """Local ids in canonical trees would reach S3 and break artifact parity."""
    _stub_everything(monkeypatch)
    F.run_camera_local(state=local_state, camera_id=CAM,
                       video_path="/v.mp4", workspace=str(tmp_path),
                       feat_models_dir="/m", features=["door"], fps=15.0,
                       verbose=False)
    for canonical in ("wagon_cache", "wagon_states", "evidence",
                      "global_state", "combined", "reports"):
        assert not (tmp_path / canonical).exists(), (
            "Phase 1 created the canonical directory %r" % canonical)
    assert (tmp_path / F.CAMERA_LOCAL_DIRNAME / CAM).exists()


# -----------------------------------------------------------------------------
# Batch's order and Batch's dispatch
# -----------------------------------------------------------------------------

def _stub_everything(monkeypatch, order_sink=None):
    """Replace the models, keep the orchestration under test."""
    from materializer import wagon_cache_builder

    # Shaped like the REAL CacheBuildResult: frames_written is
    # {gw_id -> {camera_id -> n}}, never a scalar. A stub with the wrong type
    # is how a %d-vs-dict crash reached EC2 with a green suite.
    from materializer.wagon_cache_builder import CacheBuildResult
    _Res = lambda: CacheBuildResult(
        cache_root="/x",
        frames_written={"LEFT_UP_W1": {CAM: 21}, "LEFT_UP_W2": {CAM: 21}},
        per_camera_total={CAM: 42}, missing_cameras=[], elapsed_seconds=0.1)

    monkeypatch.setattr(wagon_cache_builder, "build", lambda **kw: _Res())

    from orchestrator import master_runner

    def _fake_loader(name):
        def _run(**kw):
            if order_sink is not None:
                order_sink.append(name)
            return {"w": name}
        return _run

    monkeypatch.setattr(master_runner, "load_feature_runner", _fake_loader)

    from fusion import wagon_state_builder
    monkeypatch.setattr(wagon_state_builder, "build",
                        lambda **kw: {"LEFT_UP_W1": object()})


def test_load_completes_before_damage(tmp_path, local_state, monkeypatch):
    """Damage reads the load JSON; Batch orders it this way for that reason."""
    order = []
    _stub_everything(monkeypatch, order_sink=order)
    F.run_camera_local(state=local_state, camera_id=CAM, video_path="/v.mp4",
                       workspace=str(tmp_path), feat_models_dir="/m",
                       features=["door", "load", "damage"], fps=15.0,
                       verbose=False)
    assert order[0] == "load", "load must run first, got %s" % order
    assert "damage" in order[1:]


def test_features_are_reached_through_batchs_dispatch():
    """No second registry -- the same `load_feature_runner` Batch uses."""
    import inspect
    src = inspect.getsource(F.run_camera_local)
    assert "load_feature_runner" in src
    for banned in ("import features.door", "from features.door",
                   "DoorTracker(", "def _door", "def _damage", "def _load"):
        assert banned not in src, "Phase 1 reimplements %r" % banned


def test_the_same_kwargs_batch_passes(tmp_path, local_state, monkeypatch):
    seen = {}
    from materializer import wagon_cache_builder

    from materializer.wagon_cache_builder import CacheBuildResult
    monkeypatch.setattr(wagon_cache_builder, "build",
                        lambda **kw: CacheBuildResult(
                            cache_root="/x",
                            frames_written={"LEFT_UP_W1": {CAM: 1}},
                            per_camera_total={CAM: 1}))
    from orchestrator import master_runner
    monkeypatch.setattr(master_runner, "load_feature_runner",
                        lambda name: (lambda **kw: seen.update({name: kw})))
    from fusion import wagon_state_builder
    monkeypatch.setattr(wagon_state_builder, "build", lambda **kw: {})

    F.run_camera_local(state=local_state, camera_id=CAM, video_path="/v.mp4",
                       workspace=str(tmp_path), feat_models_dir="/models",
                       features=["door", "load", "damage"], fps=15.0,
                       door_stride=7, damage_stride=9, load_stride=11,
                       verbose=False)
    for name in ("door", "load", "damage"):
        kw = seen[name]
        assert set(("state", "cache_root", "feature_models_dir", "output_dir",
                    "evidence_root", "verbose")) <= set(kw)
    assert seen["door"]["sample_stride"] == 7
    assert seen["damage"]["sample_stride"] == 9
    assert seen["load"]["sample_stride"] == 11


# -----------------------------------------------------------------------------
# Disabled features
# -----------------------------------------------------------------------------

def test_a_disabled_feature_gets_batchs_sentinel(tmp_path, local_state,
                                                 monkeypatch):
    """"DISABLED BY USER" is visible in the report; silence is not the same."""
    _stub_everything(monkeypatch)
    out = F.run_camera_local(state=local_state, camera_id=CAM,
                             video_path="/v.mp4", workspace=str(tmp_path),
                             feat_models_dir="/m", features=["door"],
                             fps=15.0, verbose=False)
    states = out["paths"]["states_root"]
    for disabled in ("load", "damage", "ocr"):
        p = os.path.join(states, disabled, "LEFT_UP_W1.json")
        assert os.path.isfile(p), "%s got no sentinel" % disabled
        doc = json.load(open(p))
        assert doc.get("status") == C.STATUS_DISABLED
        assert doc.get("disabled_by_user") is True


def test_an_enabled_feature_gets_no_sentinel(tmp_path, local_state,
                                             monkeypatch):
    _stub_everything(monkeypatch)
    out = F.run_camera_local(state=local_state, camera_id=CAM,
                             video_path="/v.mp4", workspace=str(tmp_path),
                             feat_models_dir="/m", features=["door"],
                             fps=15.0, verbose=False)
    assert not os.path.isfile(os.path.join(out["paths"]["states_root"],
                                           "door", "LEFT_UP_W1.json"))


# -----------------------------------------------------------------------------
# Failure isolation and measurement
# -----------------------------------------------------------------------------

def test_a_crashed_feature_costs_the_feature_not_the_camera(tmp_path,
                                                            local_state,
                                                            monkeypatch):
    _stub_everything(monkeypatch)
    from orchestrator import master_runner

    def _loader(name):
        if name == "door":
            def _boom(**kw):
                raise RuntimeError("model exploded")
            return _boom
        return lambda **kw: {}

    monkeypatch.setattr(master_runner, "load_feature_runner", _loader)
    out = F.run_camera_local(state=local_state, camera_id=CAM,
                             video_path="/v.mp4", workspace=str(tmp_path),
                             feat_models_dir="/m",
                             features=["door", "load"], fps=15.0,
                             verbose=False)
    assert out["feature_summary"]["door"] == {}
    assert "feature_door" in out["timings"]


def test_per_stage_timings_are_recorded(tmp_path, local_state, monkeypatch):
    """An aggregate hides which model is expensive; commit 5 needs the split."""
    _stub_everything(monkeypatch)
    out = F.run_camera_local(state=local_state, camera_id=CAM,
                             video_path="/v.mp4", workspace=str(tmp_path),
                             feat_models_dir="/m",
                             features=["door", "load", "damage"], fps=15.0,
                             verbose=False)
    assert "cache" in out["timings"]
    assert "fusion" in out["timings"]
    for name in ("door", "load", "damage"):
        assert "feature_%s" % name in out["timings"]


def test_cache_stats_come_from_the_real_result_fields(tmp_path, local_state,
                                                      monkeypatch):
    """`CacheBuildResult` has no `.counts`; a getattr default hid an empty cache."""
    _stub_everything(monkeypatch)
    out = F.run_camera_local(state=local_state, camera_id=CAM,
                             video_path="/v.mp4", workspace=str(tmp_path),
                             feat_models_dir="/m", features=["door"],
                             fps=15.0, verbose=False)
    assert out["cache"]["total_frames"] == 42          # the SCALAR
    assert out["cache"]["frames_written"] == {          # the per-wagon map
        "LEFT_UP_W1": {CAM: 21}, "LEFT_UP_W2": {CAM: 21}}
    assert out["cache"]["per_camera_total"] == {CAM: 42}
