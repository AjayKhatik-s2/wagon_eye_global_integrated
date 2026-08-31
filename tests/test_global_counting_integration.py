"""Integration contract: NEW global counting engine -> OLD f3d2d81 pipeline.

Requirements A-L of the integration brief, each as an explicit test.

Nothing here needs model weights, videos or the engine itself: the engine's
harvested output is synthesized, and everything downstream of it is the REAL
production code -- the real adapter, the real `core.global_state_loader`
parser and roster verifier, the real materializer frame-window resolver, the
real fusion builder and the real reporting layer.

No wagon count is hard-coded as a magic number: the expectations are the
relationships the integration must satisfy (count == gaps - 1, id contiguity,
reversal ordering, provenance preservation), so they cannot be satisfied by
fabricating a value.

    python -m pytest tests/test_global_counting_integration.py -q
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

_TEST_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_TEST_DIR)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from core import constants as C
from core.global_state_loader import (
    parse_global_train_state, roster_fingerprint, verify_roster_integrity,
)
from global_counting import adapter
from global_counting.runner import CameraHarvest, GlobalCountingResult, WagonHarvest


# -----------------------------------------------------------------------------
# A synthetic engine harvest, shaped like a real train
# -----------------------------------------------------------------------------

MASTER = C.CAMERA_RIGHT_UP_TOP          # dynamic master, NOT the old fixed one
REVERSED_CAMERA = C.CAMERA_LEFT_UP_TOP  # a camera whose timeline runs backwards
PARTIAL_CAMERA = C.CAMERA_LEFT_UP       # a camera that loses the last wagon

# Deliberately IRREGULAR spacing: evenly spaced gaps are genuinely
# un-orientable, so a reversal test on them would prove nothing.
GAP_POSITIONS = [0.0, 90.0, 210.0, 360.0, 545.0, 780.0, 1000.0]
GAP_COUNT = len(GAP_POSITIONS)
EXPECTED_WAGONS = GAP_COUNT - 1

CROP_START = {
    C.CAMERA_RIGHT_UP:     40,
    C.CAMERA_LEFT_UP:      75,
    C.CAMERA_RIGHT_UP_TOP: 120,     # master: non-zero, so the shift is tested
    C.CAMERA_LEFT_UP_TOP:  60,
}
FPS = {
    C.CAMERA_RIGHT_UP:     15.0,
    C.CAMERA_LEFT_UP:      15.0,
    C.CAMERA_RIGHT_UP_TOP: 15.0,
    C.CAMERA_LEFT_UP_TOP:  12.5,    # a genuinely different clock rate
}
TRIMMED = 1500


def _trimmed_frame(position: float, reverse: bool) -> int:
    """Normalized 0-1000 position -> frame inside a camera's trimmed clip."""
    fraction = position / 1000.0
    if reverse:
        fraction = 1.0 - fraction
    return int(round(fraction * (TRIMMED - 1)))


def build_harvest() -> GlobalCountingResult:
    cameras = {}
    for camera_id in C.ALL_CAMERAS:
        cameras[camera_id] = CameraHarvest(
            camera_id=camera_id,
            video_path="/videos/%s.mp4" % camera_id,
            fps=FPS[camera_id],
            total_frames=CROP_START[camera_id] + TRIMMED + 200,
            crop_start_frame=CROP_START[camera_id],
            crop_end_frame=CROP_START[camera_id] + TRIMMED - 1,
            trimmed_total_frames=TRIMMED,
            unique_gap_count=GAP_COUNT if camera_id != PARTIAL_CAMERA
            else GAP_COUNT - 1,
            trim_status="VALID",
            alignment_status="MASTER" if camera_id == MASTER else "RESOLVED",
            is_reversed=(camera_id == REVERSED_CAMERA),
            scale=-1.0 if camera_id == REVERSED_CAMERA else 1.0,
            offset=1000.0 if camera_id == REVERSED_CAMERA else 0.0,
            matched_gaps=GAP_COUNT if camera_id != PARTIAL_CAMERA
            else GAP_COUNT - 1,
        )

    wagons = []
    for index in range(EXPECTED_WAGONS):
        start_position = GAP_POSITIONS[index]
        end_position = GAP_POSITIONS[index + 1]
        wagon = WagonHarvest(
            wagon_number=index + 1,
            global_start_position=start_position,
            global_end_position=end_position,
        )
        for camera_id in C.ALL_CAMERAS:
            reverse = camera_id == REVERSED_CAMERA
            crop = CROP_START[camera_id]

            # The partially-covered camera cannot resolve the LAST wagon.
            if camera_id == PARTIAL_CAMERA and index == EXPECTED_WAGONS - 1:
                wagon.cameras[camera_id] = {
                    "start_frame": None, "end_frame": None,
                    "status": adapter.STATUS_UNMATCHED, "reversed": False,
                    "start_position": None, "end_position": None,
                }
                continue

            low = _trimmed_frame(start_position, reverse)
            high = _trimmed_frame(end_position, reverse)
            low, high = min(low, high), max(low, high)
            # One interval per camera is RECOVERED rather than DETECTED, so the
            # audit trail has something real to preserve.
            status = (adapter.STATUS_RECOVERED if index == 1
                      else adapter.STATUS_DETECTED)
            wagon.cameras[camera_id] = {
                "start_frame": low + crop,
                "end_frame": high + crop,
                "status": status,
                "reversed": reverse,
                "start_position": min(start_position, end_position),
                "end_position": max(start_position, end_position),
            }
        wagons.append(wagon)

    return GlobalCountingResult(
        master_camera=MASTER,
        global_gap_count=GAP_COUNT,
        global_wagon_count=EXPECTED_WAGONS,
        wagons=wagons,
        cameras=cameras,
        engine_dir="/opt/global_wagon_app",
        engine_output_dir="/tmp/engine_out",
        leading_non_wagon={C.CLASS_ENGINE: 1},
        trailing_non_wagon={C.CLASS_BRAKE_VAN: 1},
        csv_paths={"global_wagon_timeline": "/tmp/engine_out/gwt.csv"},
    )


@pytest.fixture(scope="module")
def document():
    return adapter.build_global_train_state_document(build_harvest())


@pytest.fixture(scope="module")
def state(document):
    # Round-tripped through JSON on purpose: Stage 1 writes a file and
    # downstream reads it, so the test must exercise the same path.
    return parse_global_train_state(json.loads(json.dumps(document)))


# -----------------------------------------------------------------------------
# A. global wagon count == global gap count - 1
# -----------------------------------------------------------------------------

def test_a_wagon_count_is_gap_count_minus_one(document, state):
    assert document["global_gap_count"] == GAP_COUNT
    assert document["total_wagons"] == GAP_COUNT - 1
    assert len(document["wagons"]) == GAP_COUNT - 1
    assert state.total_wagons == state.global_gap_count - 1
    invariants = document["invariant_checks"]
    assert invariants["invariant_holds"] is True
    assert invariants["violations"] == []
    assert invariants["rule"] == "global_wagon_count == global_gap_count - 1"


def test_a_invariant_actually_detects_a_violation():
    """The invariant must be a real check, not decoration."""
    harvest = build_harvest()
    harvest.global_wagon_count += 1          # lie about the count
    broken = adapter.build_global_train_state_document(harvest)
    assert broken["invariant_checks"]["invariant_holds"] is False
    assert broken["invariant_checks"]["violations"]


# -----------------------------------------------------------------------------
# B. wagon ids: GW_1..GW_n, unique and sequential
# -----------------------------------------------------------------------------

def test_b_wagon_ids_are_contiguous_and_unique(state):
    ids = [w.global_id for w in state.wagons]
    assert ids == ["GW_%d" % n for n in range(1, EXPECTED_WAGONS + 1)]
    assert len(set(ids)) == len(ids)
    assert [w.wagon_index for w in state.wagons] == list(
        range(1, EXPECTED_WAGONS + 1))
    # The old pipeline's own verifier is the authority here.
    assert verify_roster_integrity(state) == []


def test_b_id_format_matches_the_old_pipeline(document):
    assert adapter.GLOBAL_ID_FORMAT % 7 == "GW_7"
    for wagon in document["wagons"]:
        assert wagon["global_id"].startswith("GW_")


# -----------------------------------------------------------------------------
# C. four-camera mapping
# -----------------------------------------------------------------------------

def test_c_every_wagon_maps_to_all_four_cameras(state):
    assert state.uses_camera_frame_ranges
    for wagon in state.wagons:
        assert set(wagon.camera_frame_ranges) == set(C.ALL_CAMERAS)


def test_c_frames_are_shifted_out_of_trimmed_space(state):
    """Stage 2 opens the ORIGINAL videos, so no index may be trimmed-relative."""
    for wagon in state.wagons:
        for camera_id, entry in wagon.camera_frame_ranges.items():
            if entry["start_frame"] is None:
                continue
            assert entry["start_frame"] >= CROP_START[camera_id]
            assert entry["end_frame"] >= entry["start_frame"]


def test_c_supporting_cameras_excludes_the_unresolved_one(state):
    last = state.wagons[-1]
    assert PARTIAL_CAMERA not in last.supporting_cameras
    assert MASTER in last.supporting_cameras
    assert PARTIAL_CAMERA in state.wagons[0].supporting_cameras


# -----------------------------------------------------------------------------
# D. reversed camera
# -----------------------------------------------------------------------------

def test_d_reversed_camera_intervals_run_backwards(state):
    """As the master advances, a reversed camera's frames must DECREASE."""
    master_starts, reversed_starts = [], []
    for wagon in state.wagons:
        master_starts.append(wagon.camera_frame_ranges[MASTER]["start_frame"])
        reversed_starts.append(
            wagon.camera_frame_ranges[REVERSED_CAMERA]["start_frame"])

    assert master_starts == sorted(master_starts), "master must advance"
    assert reversed_starts == sorted(reversed_starts, reverse=True), \
        "reversed camera must run backwards against the master"


def test_d_reversal_is_flagged_not_just_implied(state):
    for wagon in state.wagons:
        assert wagon.camera_frame_ranges[REVERSED_CAMERA]["timeline_reversed"]
        assert not wagon.camera_frame_ranges[MASTER]["timeline_reversed"]
    assert state.support_alignment_summary[REVERSED_CAMERA]["timeline_reversed"]


def test_d_reversed_camera_declares_no_usable_clock_offset(state):
    """A single additive delta cannot describe a reversed timeline.

    Claiming one would corrupt every consumer of the old offset contract, so
    the adapter must mark it unusable and let the frame ranges carry the truth.
    """
    assert state.camera_offsets[REVERSED_CAMERA]["status"] == \
        adapter.OFFSET_REVERSED
    assert REVERSED_CAMERA not in state.camera_time_offsets()
    assert state.camera_time_offset(REVERSED_CAMERA) == 0.0
    # ...while the master is the reference and forward cameras do resolve.
    assert state.camera_offsets[MASTER]["status"] == adapter.OFFSET_REFERENCE
    assert state.camera_offsets[C.CAMERA_RIGHT_UP]["status"] == \
        adapter.OFFSET_RESOLVED


def test_d_interval_ordering_is_always_low_to_high(state):
    for wagon in state.wagons:
        for entry in wagon.camera_frame_ranges.values():
            if entry["start_frame"] is None:
                continue
            assert entry["start_frame"] <= entry["end_frame"]


# -----------------------------------------------------------------------------
# E. missing-gap recovery stays audited
# -----------------------------------------------------------------------------

def test_e_recovered_intervals_are_preserved_per_camera(state):
    recovered = [w.global_id for w in state.wagons
                 if w.camera_frame_ranges[MASTER]["status"]
                 == adapter.STATUS_RECOVERED]
    assert recovered, "the fixture's RECOVERED interval must survive"
    for camera_id in C.ALL_CAMERAS:
        summary = state.support_alignment_summary[camera_id]
        assert (summary["detected_intervals"]
                + summary["recovered_intervals"]
                + summary["unmatched_intervals"]) == EXPECTED_WAGONS


def test_e_unmatched_is_reported_not_fabricated(state):
    summary = state.support_alignment_summary[PARTIAL_CAMERA]
    assert summary["unmatched_intervals"] == 1
    entry = state.wagons[-1].camera_frame_ranges[PARTIAL_CAMERA]
    assert entry["status"] == adapter.STATUS_UNMATCHED
    assert entry["start_frame"] is None
    # local_range must say "no window", never guess one.
    assert state.wagons[-1].local_range(PARTIAL_CAMERA) is None


def test_e_alignment_provenance_is_kept(state):
    for camera_id in C.ALL_CAMERAS:
        summary = state.support_alignment_summary[camera_id]
        for key in ("alignment_status", "scale", "offset", "matched_gaps",
                    "unique_gaps", "timeline_reversed"):
            assert key in summary


# -----------------------------------------------------------------------------
# F. the OLD GlobalTrainState contract
# -----------------------------------------------------------------------------

def test_f_old_loader_parses_the_adapted_state(state):
    assert state.master_camera == MASTER
    assert state.master_fps == FPS[MASTER]
    assert state.master_total_frames > 0
    assert state.fusion_mode == adapter.FUSION_MODE
    assert state.master_wagon_count == EXPECTED_WAGONS
    assert state.regular_wagon_count >= 0
    assert roster_fingerprint(state)


def test_f_engine_and_brake_van_kpis_are_populated_from_wagon_window(state):
    """Requirement 17: keep the KPIs honest WITHOUT touching Stage-1 counting.

    The engine trims to the wagon region, so the locomotive and brake van get
    no GW id -- exactly as the old master-fixed counter behaved. Their counts
    ride in `wagon_window`, the channel GlobalTrainState already reads.
    """
    assert state.engine_count == 1
    assert state.brake_van_count == 1
    assert state.wagon_window["source"] == adapter.ENGINE_NAME
    # ...and they are NOT in the roster.
    assert all(w.classification != C.CLASS_ENGINE for w in state.wagons)


def test_f_state_declares_its_engine(document):
    assert document["global_counting_engine"] == adapter.ENGINE_NAME
    assert document["schema"] == adapter.STATE_SCHEMA


def test_f_per_camera_tracking_document_carries_fps():
    from core.global_state_loader import load_per_camera_fps

    harvest = build_harvest()
    tracking = adapter.build_per_camera_tracking_document(harvest)
    assert set(tracking) == set(C.ALL_CAMERAS)
    import tempfile
    with tempfile.TemporaryDirectory() as directory:
        path = os.path.join(directory, "per_camera_tracking.json")
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(tracking, handle)
        assert load_per_camera_fps(path) == FPS


def test_f_state_survives_a_file_round_trip(tmp_path):
    harvest = build_harvest()
    state_path, tracking_path = adapter.write_documents(harvest, str(tmp_path))
    assert os.path.isfile(state_path) and os.path.isfile(tracking_path)
    from core.global_state_loader import load_global_train_state
    reloaded = load_global_train_state(state_path)
    assert verify_roster_integrity(reloaded) == []
    assert reloaded.uses_camera_frame_ranges


# -----------------------------------------------------------------------------
# G. the wagon cache uses the ALIGNED ranges
# -----------------------------------------------------------------------------

def test_g_materializer_prefers_the_aligned_range(state):
    """Requirement 8: never fall back to master_time x fps when a real one exists."""
    from materializer.wagon_cache_builder import _wagon_local_range

    for wagon in state.wagons:
        for camera_id in C.ALL_CAMERAS:
            explicit = wagon.local_range(camera_id)
            resolved = _wagon_local_range(
                wagon, FPS[camera_id], 10 ** 7,
                time_offset=999.0,        # a wrong offset, deliberately
                camera_id=camera_id)
            if explicit is None:
                continue
            assert resolved == explicit, (
                "%s/%s used the projection instead of the aligned window"
                % (wagon.global_id, camera_id))


def test_g_reversed_camera_window_is_not_the_master_projection(state):
    from materializer.wagon_cache_builder import _wagon_local_range

    wagon = state.wagons[0]
    aligned = _wagon_local_range(wagon, FPS[REVERSED_CAMERA], 10 ** 7,
                                 camera_id=REVERSED_CAMERA)
    projected = _wagon_local_range(wagon, FPS[REVERSED_CAMERA], 10 ** 7,
                                   camera_id=None)
    assert aligned != projected, (
        "a reversed camera must not resolve to the master-time projection")
    assert aligned == wagon.local_range(REVERSED_CAMERA)


def test_g_legacy_state_still_uses_the_old_projection():
    """A state without aligned windows must behave exactly as before."""
    from core.global_state_loader import GlobalWagon
    from materializer.wagon_cache_builder import _wagon_local_range

    wagon = GlobalWagon(
        global_id="GW_1", wagon_index=1, start_frame_master=0,
        end_frame_master=30, start_time=2.0, end_time=4.0,
        classification=C.CLASS_WAGON)
    assert wagon.local_range(C.CAMERA_RIGHT_UP) is None
    assert _wagon_local_range(wagon, 10.0, 10 ** 7,
                              camera_id=C.CAMERA_RIGHT_UP) == (20, 39)
    # the offset still applies on that path
    assert _wagon_local_range(wagon, 10.0, 10 ** 7, time_offset=1.0,
                              camera_id=C.CAMERA_RIGHT_UP) == (10, 29)


def test_g_window_outside_the_footage_contributes_nothing(state):
    """Clamping would fabricate evidence from a different wagon."""
    from materializer.wagon_cache_builder import _wagon_local_range

    wagon = state.wagons[-1]
    start, end = _wagon_local_range(wagon, FPS[MASTER], 5,
                                    camera_id=MASTER)
    assert end < start, "an out-of-footage window must be empty"


# -----------------------------------------------------------------------------
# H + J. features and fusion run against the new wagon ids
# -----------------------------------------------------------------------------

def _write_feature_json(root, feature, gw_ids, payload):
    directory = os.path.join(root, feature)
    os.makedirs(directory, exist_ok=True)
    for gw_id in gw_ids:
        document = dict(payload)
        document.update({"global_id": gw_id, "feature": feature,
                         "status": C.STATUS_OK})
        with open(os.path.join(directory, "%s.json" % gw_id), "w",
                  encoding="utf-8") as handle:
            json.dump(document, handle)


def _states_without_ocr(root, gw_ids):
    _write_feature_json(root, "door", gw_ids, {
        "left_door": C.DOOR_CLOSED, "left_door_confidence": 0.9,
        "right_door": C.DOOR_OPEN, "right_door_confidence": 0.8})
    _write_feature_json(root, "load", gw_ids, {
        "load_status": C.LOAD_LOADED, "load_confidence": 0.77})
    _write_feature_json(root, "damage", gw_ids, {
        "top_damage": C.DAMAGE_OK, "top_damage_details": []})
    assert not os.path.exists(os.path.join(root, "ocr"))


def test_h_feature_results_key_on_the_new_global_ids(state, tmp_path):
    from features._common import write_per_wagon_json, empty_payload

    root = str(tmp_path / "wagon_states")
    for wagon in state.wagons:
        write_per_wagon_json(
            os.path.join(root, "door"), wagon.global_id,
            empty_payload(wagon.global_id, "door", C.STATUS_OK))
    written = sorted(os.listdir(os.path.join(root, "door")))
    assert written == sorted("%s.json" % w.global_id for w in state.wagons)


def test_j_fusion_builds_one_unified_state_per_new_wagon(state, tmp_path):
    from fusion import wagon_state_builder

    root = str(tmp_path / "wagon_states")
    gw_ids = [w.global_id for w in state.wagons]
    _states_without_ocr(root, gw_ids)

    unified = wagon_state_builder.build(
        state=state, wagon_states_root=root, write_per_wagon_json=False,
        verbose=False)

    assert set(unified) == set(gw_ids)
    for gw_id in gw_ids:
        wagon = unified[gw_id]
        assert wagon.left_door == C.DOOR_CLOSED
        assert wagon.right_door == C.DOOR_OPEN
        assert wagon.load_status == C.LOAD_LOADED
        assert wagon.top_damage == C.DAMAGE_OK
        # OCR absent -> nothing fabricated
        assert wagon.wagon_identifier == C.NO_DATA


def test_j_fusion_does_not_touch_the_roster(state, tmp_path):
    from core.global_state_loader import assert_roster_unchanged
    from fusion import wagon_state_builder

    root = str(tmp_path / "wagon_states")
    _states_without_ocr(root, [w.global_id for w in state.wagons])
    guard = roster_fingerprint(state)
    wagon_state_builder.build(state=state, wagon_states_root=root,
                              write_per_wagon_json=False, verbose=False)
    assert_roster_unchanged(state, guard, stage="test fusion")


# -----------------------------------------------------------------------------
# I. OCR disabled means OCR is never imported
# -----------------------------------------------------------------------------

def test_i_orchestrator_import_pulls_in_no_processor():
    code = (
        "import sys; sys.path.insert(0, %r);"
        "import orchestrator.master_runner;"
        "leaked=[m for m in sys.modules if m.startswith('features.')"
        " and m.endswith('.processor')];"
        "print('LEAKED=' + ','.join(sorted(leaked)))" % _REPO_ROOT
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True,
                         text=True, cwd=_REPO_ROOT)
    assert out.returncode == 0, out.stderr
    assert out.stdout.split("LEAKED=")[1].strip() == "", out.stdout


def test_i_selecting_three_features_never_imports_ocr_or_easyocr():
    """The whole point: no OCR module, no EasyOCR, no OCR weights."""
    code = (
        "import sys; sys.path.insert(0, %r);"
        "from orchestrator import master_runner as mr;"
        "cfg = mr.feature_config_from_selection(mr.parse_features"
        "('door,load,damage'));"
        "runners = {k: mr.load_feature_runner(k) for k in cfg.enabled_keys()};"
        "print('ENABLED=' + ','.join(sorted(runners)));"
        "print('OCR_MODULE=' + str('features.ocr.processor' in sys.modules));"
        "print('EASYOCR=' + str('easyocr' in sys.modules))" % _REPO_ROOT
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True,
                         text=True, cwd=_REPO_ROOT)
    assert out.returncode == 0, out.stderr
    assert "ENABLED=damage,door,load" in out.stdout, out.stdout
    assert "OCR_MODULE=False" in out.stdout, out.stdout
    assert "EASYOCR=False" in out.stdout, out.stdout


def test_i_ocr_remains_available_on_request():
    """--features ocr / all must still work: OCR is optional, not removed."""
    from orchestrator import master_runner as mr

    assert "ocr" in mr.parse_features("all")
    assert mr.parse_features("ocr") == ("ocr",)
    assert mr.feature_config_from_selection(("ocr",)).is_enabled("ocr")
    assert callable(mr.load_feature_runner("ocr"))


def test_i_disabled_feature_uses_the_existing_sentinel_path(state):
    """A skipped feature follows the pipeline's own DISABLED path, not a new one."""
    from orchestrator import master_runner as mr

    config = mr.feature_config_from_selection(("door", "load", "damage"))
    assert config.disabled_keys() == ["ocr"]
    assert config.enabled_keys() == ["door", "load", "damage"]


# -----------------------------------------------------------------------------
# K. reporting
# -----------------------------------------------------------------------------

def test_k_report_json_and_pdf_are_generated(state, tmp_path):
    from fusion import wagon_state_builder
    from reporting import combined_train_report

    states_root = str(tmp_path / "wagon_states")
    _states_without_ocr(states_root, [w.global_id for w in state.wagons])
    unified = wagon_state_builder.build(
        state=state, wagon_states_root=states_root,
        write_per_wagon_json=False, verbose=False)

    result = combined_train_report.build(
        state=state, unified=unified, output_dir=str(tmp_path / "reports"),
        batch_key="integrationtest", verbose=False)

    assert result.get("json_path") and os.path.isfile(result["json_path"])
    with open(result["json_path"], "r", encoding="utf-8") as handle:
        document = json.load(handle)
    assert document
    # PDF needs reportlab; when it is installed it must actually be produced.
    try:
        import reportlab                                    # noqa: F401
    except ImportError:
        pytest.skip("reportlab not installed")
    assert result.get("pdf_path") and os.path.isfile(result["pdf_path"])


def test_k_report_counts_agree_with_the_new_engine(state, tmp_path):
    from core.unified_wagon_state import summarize_wagons
    from fusion import wagon_state_builder

    states_root = str(tmp_path / "wagon_states")
    _states_without_ocr(states_root, [w.global_id for w in state.wagons])
    unified = wagon_state_builder.build(
        state=state, wagon_states_root=states_root,
        write_per_wagon_json=False, verbose=False)

    summary = summarize_wagons(list(unified.values()))
    assert summary["total_wagons"] == EXPECTED_WAGONS
    assert summary["ocr_captured"] == 0          # no OCR, honestly reported


# -----------------------------------------------------------------------------
# L. module isolation
# -----------------------------------------------------------------------------

def test_l_colliding_module_names_are_identified():
    """`reporting` and `models` exist on BOTH sides -- that is the hazard."""
    from global_counting import runner as gc_runner

    ours = {name for name in os.listdir(_REPO_ROOT)
            if os.path.isdir(os.path.join(_REPO_ROOT, name))}
    collisions = ours & set(gc_runner.ENGINE_MODULES)
    assert {"reporting", "models"} <= collisions
    # ...and every colliding name must be covered by the session's stash list.
    for name in collisions:
        assert name in gc_runner.ENGINE_MODULES


def test_l_engine_session_restores_our_modules(tmp_path):
    """Our `reporting` package must be the same object after the session."""
    from global_counting import runner as gc_runner
    import reporting as our_reporting

    fake_engine = tmp_path / "global_wagon_app"
    fake_engine.mkdir()
    # A stand-in engine `reporting.py` that would shadow ours if unprotected.
    (fake_engine / "reporting.py").write_text(
        "MARKER = 'engine'\n", encoding="utf-8")
    (fake_engine / gc_runner.ENGINE_MARKER).write_text("", encoding="utf-8")

    before = sys.modules["reporting"]
    with gc_runner.engine_session(str(fake_engine)):
        import reporting as engine_reporting
        assert getattr(engine_reporting, "MARKER", None) == "engine", \
            "inside the session the ENGINE's module must win"

    import reporting as after_reporting
    assert after_reporting is before is our_reporting
    assert not hasattr(after_reporting, "MARKER")
    assert str(fake_engine) not in sys.path


def test_l_session_leaves_no_engine_module_behind(tmp_path):
    from global_counting import runner as gc_runner

    fake_engine = tmp_path / "global_wagon_app"
    fake_engine.mkdir()
    (fake_engine / "camera_map.py").write_text("CAMERAS = []\n", encoding="utf-8")
    (fake_engine / gc_runner.ENGINE_MARKER).write_text("", encoding="utf-8")

    with gc_runner.engine_session(str(fake_engine)):
        import camera_map                                    # noqa: F401
        assert "camera_map" in sys.modules
    assert "camera_map" not in sys.modules


def test_l_engine_is_not_vendored_into_this_repo():
    """The engine must stay external -- one source of truth for counting."""
    from global_counting import runner as gc_runner

    assert not os.path.exists(
        os.path.join(_REPO_ROOT, gc_runner.ENGINE_MARKER))
    for package in ("core", "features", "fusion", "materializer",
                    "orchestrator", "reporting", "wagon_count"):
        root = os.path.join(_REPO_ROOT, package)
        for dirpath, _dirs, files in os.walk(root):
            if "__pycache__" in dirpath:
                continue
            assert gc_runner.ENGINE_MARKER not in files, (
                "engine source copied into %s" % dirpath)


# -----------------------------------------------------------------------------
# Stage-1 wiring
# -----------------------------------------------------------------------------

def test_stage1_default_engine_is_the_new_one():
    from reconstruction import runner as reconstruction_runner

    assert reconstruction_runner.ENGINE_DEFAULT == \
        reconstruction_runner.ENGINE_GLOBAL_APP
    assert reconstruction_runner.resolve_engine() == \
        reconstruction_runner.ENGINE_GLOBAL_APP


def test_stage1_rollback_engine_is_still_selectable(monkeypatch):
    from reconstruction import runner as reconstruction_runner

    assert reconstruction_runner.resolve_engine("wagon_count") == \
        reconstruction_runner.ENGINE_WAGON_COUNT
    monkeypatch.setenv(reconstruction_runner.ENGINE_ENV_VAR, "wagon_count")
    assert reconstruction_runner.resolve_engine() == \
        reconstruction_runner.ENGINE_WAGON_COUNT
    # the argument still wins over the environment
    assert reconstruction_runner.resolve_engine("global_wagon_app") == \
        reconstruction_runner.ENGINE_GLOBAL_APP


def test_stage1_rejects_an_unknown_engine():
    from reconstruction import runner as reconstruction_runner

    with pytest.raises(reconstruction_runner.ReconstructionError):
        reconstruction_runner.resolve_engine("something_else")


def test_only_one_engine_runs_per_batch(monkeypatch, tmp_path):
    """The architecture must never become new counting + old counting."""
    from reconstruction import runner as reconstruction_runner

    called = []
    monkeypatch.setattr(reconstruction_runner, "_run_global_app",
                        lambda **kw: called.append("new"))
    monkeypatch.setattr(reconstruction_runner, "_run_wagon_count",
                        lambda **kw: called.append("old"))

    reconstruction_runner.run(
        video_paths={c: "x" for c in C.ALL_CAMERAS},
        reconstruction_models_dir=str(tmp_path), output_dir=str(tmp_path),
        repo_root=_REPO_ROOT, verbose=False)
    assert called == ["new"]

    called.clear()
    reconstruction_runner.run(
        video_paths={c: "x" for c in C.ALL_CAMERAS},
        reconstruction_models_dir=str(tmp_path), output_dir=str(tmp_path),
        repo_root=_REPO_ROOT, engine="wagon_count", verbose=False)
    assert called == ["old"]
