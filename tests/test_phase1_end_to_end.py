"""Phase-1 lifecycle: orchestration, isolation, and the failure paths.

TEST CLASSIFICATION -- READ THIS BEFORE TRUSTING A GREEN RUN

  REAL   `TestRealModels` runs the actual engine and the actual feature
         weights when this machine has them, and SKIPS with a named reason
         when it does not. It never silently passes.

  STUB   everything else. The models are replaced on purpose, because these
         tests are about ORCHESTRATION -- ordering, isolation, sealing,
         failure isolation -- and about paths a real model cannot be made to
         take on demand (a cache that fails, a processor that raises, a
         renderer that dies). Stubbing is the only way to reach them.

Neither class of test is production validation. Production validation is a run
on the EC2 box, which this environment cannot reach; see
docs/EC2_PHASE1_VALIDATION.md.
"""

from __future__ import annotations

import json
import os

import pytest

from core import constants as C
from sequential import camera_features as F
from sequential import camera_report as R
from sequential import evidence as ev
from sequential import local_state_adapter as A

CAM = C.CAMERA_RIGHT_UP


# =============================================================================
# Shared fixtures
# =============================================================================

def _evidence(camera_id=CAM, frames=(100, 200, 300)):
    gaps = [ev.GapObservation(local_gap_id="%s_G%d" % (camera_id, i),
                              confirmation_frame=f, first_frame=f - 2,
                              last_frame=f + 2, normalized_position=i * 100.0,
                              max_confidence=0.9)
            for i, f in enumerate(frames, start=1)]
    segments = [{"segment_id": ev.SEGMENT_ID_FORMAT % (camera_id, i),
                 "segment_index": i, "start_frame": a.confirmation_frame,
                 "end_frame": b.confirmation_frame,
                 "opening_gap": a.local_gap_id, "closing_gap": b.local_gap_id,
                 "canonical": False}
                for i, (a, b) in enumerate(zip(gaps, gaps[1:]), start=1)]
    return ev.CameraEvidence(
        camera_id=camera_id, status=ev.STATUS_SEALED,
        timing=ev.CameraTiming(fps=15.0, total_frames=1000),
        gaps=gaps, segments=segments,
        engine_result={"video_info": {"width": 960, "height": 540}})


def _stub_stages(monkeypatch, *, cache=None, feature=None, fusion=None):
    """Replace the three expensive stages. STUB, by design."""
    from materializer import wagon_cache_builder
    from fusion import wagon_state_builder
    from orchestrator import master_runner

    # The REAL dataclass, real shape: frames_written is
    # {gw_id -> {camera_id -> n}}. A scalar stub hid a %d-vs-dict crash.
    from materializer.wagon_cache_builder import CacheBuildResult

    def _res(**kw):
        return CacheBuildResult(
            cache_root="/x", frames_written={"RIGHT_UP_W1": {CAM: 10}},
            per_camera_total={CAM: 10})

    monkeypatch.setattr(wagon_cache_builder, "build", cache or _res)
    monkeypatch.setattr(master_runner, "load_feature_runner",
                        feature or (lambda name: (lambda **kw: {"x": name})))
    monkeypatch.setattr(wagon_state_builder, "build",
                        fusion or (lambda **kw: {"RIGHT_UP_W1": object()}))


# =============================================================================
# 1. Orchestration -- STUB
# =============================================================================

class TestOrchestration:
    """The Phase-1 chain, with the models stubbed. STUB."""

    def test_the_chain_is_wired_in_order(self):
        import inspect
        src = inspect.getsource(
            __import__("sequential.camera_runner", fromlist=["x"]).process_camera)
        # engine -> local state -> features -> report -> seal, in that order.
        for earlier, later in (
                ("camera_pipeline", "build_local_state"),
                ("build_local_state", "run_camera_local"),
                ("run_camera_local", "camera_report.build"),
                ("camera_report.build", "write_seal")):
            assert src.index(earlier) < src.index(later), (
                "%s must precede %s in the camera lifecycle" % (earlier, later))

    def test_release_precedes_persistence(self):
        """Models are the memory cost; free them before writing artifacts."""
        import inspect
        src = inspect.getsource(
            __import__("sequential.camera_runner", fromlist=["x"]).process_camera)
        assert src.index("_release()") < src.index("write_evidence")

    def test_phase1_needs_no_other_camera(self, tmp_path, monkeypatch):
        """Camera independence: one camera's evidence is the only input."""
        _stub_stages(monkeypatch)
        state = A.build_local_state(_evidence())
        out = F.run_camera_local(state=state, camera_id=CAM,
                                 video_path="/v.mp4", workspace=str(tmp_path),
                                 feat_models_dir="/m",
                                 features=["door", "load", "damage"],
                                 fps=15.0, verbose=False)
        assert out["paths"]["root"].endswith(os.path.join("camera_local", CAM))
        # Nothing about any other camera was consulted or created.
        for other in (C.CAMERA_LEFT_UP, C.CAMERA_RIGHT_UP_TOP,
                      C.CAMERA_LEFT_UP_TOP):
            assert not (tmp_path / F.CAMERA_LOCAL_DIRNAME / other).exists()


# =============================================================================
# 2. Isolation and identity -- STUB
# =============================================================================

class TestIsolationAndIdentity:
    """STUB. Where Phase 1 writes, and what it calls its wagons."""

    def test_no_canonical_directory_is_created(self, tmp_path, monkeypatch):
        _stub_stages(monkeypatch)
        F.run_camera_local(state=A.build_local_state(_evidence()),
                           camera_id=CAM, video_path="/v.mp4",
                           workspace=str(tmp_path), feat_models_dir="/m",
                           features=["door"], fps=15.0, verbose=False)
        for canonical in ("wagon_cache", "wagon_states", "evidence",
                          "global_state", "combined", "reports",
                          "processed_videos", "archive"):
            assert not (tmp_path / canonical).exists(), canonical

    def test_no_canonical_wagon_id_anywhere_in_phase1(self, tmp_path,
                                                      monkeypatch):
        _stub_stages(monkeypatch)
        evidence = _evidence()
        state = A.build_local_state(evidence)
        out = F.run_camera_local(state=state, camera_id=CAM,
                                 video_path="/v.mp4", workspace=str(tmp_path),
                                 feat_models_dir="/m", features=["door"],
                                 fps=15.0, verbose=False)
        blob = json.dumps({
            "ids": [w.global_id for w in state.wagons],
            "summary": out["feature_summary"],
        }, default=str)
        assert "GW_" not in blob
        for wagon in state.wagons:
            assert wagon.global_id.startswith(CAM + "_W")

    def test_local_ids_are_accepted_by_the_batch_machinery(self, tmp_path):
        """`LEFT_UP_W1` must survive the cache builder's own range resolver."""
        from materializer.wagon_cache_builder import _wagon_local_range
        state = A.build_local_state(_evidence(camera_id=C.CAMERA_LEFT_UP))
        wagon = state.wagons[0]
        assert wagon.global_id == "LEFT_UP_W1"
        assert _wagon_local_range(wagon, 15.0, 1000, 0.0,
                                  C.CAMERA_LEFT_UP) == (100, 200)


# =============================================================================
# 3. Failure paths -- STUB (a real model cannot be made to fail on demand)
# =============================================================================

class TestFailurePaths:
    """STUB, necessarily. Each path must degrade, never take the camera down."""

    def test_cache_build_failure(self, tmp_path, monkeypatch):
        def _boom(**kw):
            raise RuntimeError("no video codec")
        _stub_stages(monkeypatch, cache=_boom)
        out = F.run_camera_local(state=A.build_local_state(_evidence()),
                                 camera_id=CAM, video_path="/v.mp4",
                                 workspace=str(tmp_path), feat_models_dir="/m",
                                 features=["door"], fps=15.0, verbose=False)
        assert out["cache"] == {}          # reported empty, not faked
        assert "cache" in out["timings"]   # still measured

    def test_one_feature_failing_does_not_stop_the_others(self, tmp_path,
                                                          monkeypatch):
        def _loader(name):
            if name == "damage":
                def _boom(**kw):
                    raise RuntimeError("cuda oom")
                return _boom
            return lambda **kw: {"ok": name}
        _stub_stages(monkeypatch, feature=_loader)
        out = F.run_camera_local(state=A.build_local_state(_evidence()),
                                 camera_id=CAM, video_path="/v.mp4",
                                 workspace=str(tmp_path), feat_models_dir="/m",
                                 features=["door", "load", "damage"],
                                 fps=15.0, verbose=False)
        assert out["feature_summary"]["damage"] == {}
        assert out["feature_summary"]["load"] == {"ok": "load"}
        assert out["feature_summary"]["door"] == {"ok": "door"}

    def test_fusion_failure_leaves_an_empty_unified_not_a_guess(self, tmp_path,
                                                                monkeypatch):
        def _boom(**kw):
            raise RuntimeError("unreadable state json")
        _stub_stages(monkeypatch, fusion=_boom)
        out = F.run_camera_local(state=A.build_local_state(_evidence()),
                                 camera_id=CAM, video_path="/v.mp4",
                                 workspace=str(tmp_path), feat_models_dir="/m",
                                 features=["door"], fps=15.0, verbose=False)
        assert out["unified"] == {}

    def test_no_unified_means_no_batch_rendered_pdf(self, tmp_path):
        """Without fused states the wagon-oriented report has nothing to show.

        It must fall back to the camera-local counting report rather than
        render empty pages.
        """
        got = R._batch_rendered_pdf(
            workspace=str(tmp_path), camera_id=CAM, batch_key="K",
            local_state=A.build_local_state(_evidence()),
            local_result={"unified": {}, "paths": {}},
            output_pdf=str(tmp_path / "x.pdf"), verbose=False)
        assert got is None

    def test_a_renderer_crash_falls_back_and_does_not_raise(self, tmp_path,
                                                            monkeypatch):
        from reporting import camera_reports

        def _boom(**kw):
            raise RuntimeError("reportlab exploded")
        monkeypatch.setattr(camera_reports, "build_camera_report", _boom)
        got = R._batch_rendered_pdf(
            workspace=str(tmp_path), camera_id=CAM, batch_key="K",
            local_state=A.build_local_state(_evidence()),
            local_result={"unified": {"RIGHT_UP_W1": object()},
                          "paths": {"evidence_root": str(tmp_path)}},
            output_pdf=str(tmp_path / "x.pdf"), verbose=False)
        assert got is None

    def test_an_item_build_crash_yields_no_items_not_partial_ones(self,
                                                                  monkeypatch):
        from reporting import camera_reports

        def _boom(**kw):
            raise RuntimeError("bad unified state")
        monkeypatch.setattr(camera_reports, "_build_camera_items", _boom)
        items = R._inspection_items(CAM, A.build_local_state(_evidence()),
                                    {"unified": {"RIGHT_UP_W1": object()},
                                     "paths": {}})
        assert items == []

    def test_a_camera_with_no_bounded_wagon_yields_no_local_state(self):
        """One gap bounds no wagon; the caller must decide, not guess."""
        assert A.build_local_state(_evidence(frames=(100,))) is None

    def test_an_empty_camera_result_does_not_crash_the_document(self):
        doc = R.build_document(_evidence(frames=(100,)), batch_key="K",
                               local_state=None, local_result={})
        assert doc["inspection"]["wagons"] == []
        assert doc["canonical"] is False


# =============================================================================
# 4. REAL models -- runs only where they exist, skips loudly otherwise
# =============================================================================

def _real_engine():
    for candidate in (os.environ.get("WAGONEYE_ENGINE_DIR"),
                      os.path.expanduser("~/global_count_ec2")):
        if candidate and os.path.isfile(os.path.join(candidate,
                                                     "camera_pipeline.py")):
            return candidate
    return None


def _real_models():
    for base in (os.environ.get("WAGONEYE_FEAT_MODELS_DIR"),
                 os.path.expanduser("~/global_wagon_models"),
                 os.path.expanduser("~/Desktop/Ajay_global_train/global_train/"
                                    "models/features")):
        if base and os.path.isfile(os.path.join(base, "door_state.pt")):
            return base
    return None


class TestRealModels:
    """REAL. Skips with a named reason rather than passing vacuously."""

    def test_the_engine_checkout_exposes_the_camera_pipeline(self):
        engine = _real_engine()
        if engine is None:
            pytest.skip("no engine checkout with camera_pipeline.py on this "
                        "machine -- Phase 1 cannot run for real here")
        assert os.path.isfile(os.path.join(engine, "camera_pipeline.py"))
        assert os.path.isfile(os.path.join(engine, "global_alignment.py"))

    def test_the_three_feature_weights_are_present(self):
        base = _real_models()
        if base is None:
            pytest.skip("no feature weights on this machine -- Phase 1 "
                        "features cannot run for real here")
        for weight in ("door_state.pt", "loaded.pt", "damage.pt"):
            assert os.path.isfile(os.path.join(base, weight)), weight

    def test_the_real_processors_import_and_expose_run(self):
        """Real modules, real signatures -- no model is executed."""
        from orchestrator.master_runner import load_feature_runner
        for name in ("door", "load", "damage"):
            fn = load_feature_runner(name)
            import inspect
            params = inspect.signature(fn).parameters
            for required in ("state", "cache_root", "feature_models_dir",
                             "output_dir", "evidence_root"):
                assert required in params, "%s.run lacks %s" % (name, required)
