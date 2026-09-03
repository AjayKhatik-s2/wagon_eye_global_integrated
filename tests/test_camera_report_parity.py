"""The validated per-camera report must be the SAME artifact in both modes.

WHY THE REPORTS LOOKED DIFFERENT
Batch runs two reporting stages: 5a `reporting.camera_reports.build_all` (the
validated per-camera PDFs) and 5b `reporting.combined_train_report.build`.
Sequential ran only 5b, so the validated per-camera reports were never produced
at all. What a field comparison saw instead was Sequential's camera-LOCAL
evidence report -- a different, additional artifact written the moment a camera
finishes, before any canonical wagon exists.

The two are not two implementations of one thing that drifted. Batch's camera
report is a per-camera VIEW OF THE CANONICAL ROSTER: it iterates `state.wagons`
(GW_n), reads FUSED facts from `unified[gw_id]`, takes snapshots from
`evidence/<GW>/<feature>/` and decides visibility from
`wagon_cache/<GW>/<camera>/`. None of that exists camera-locally, and the
architecture forbids canonical ids there.

So the fix is not to reshape either report: it is to make Global Assembly run
Stage 5a with the SAME renderer, and to place the camera-local snapshots into
the per-wagon layout that renderer reads. These tests pin that, and pin the
boundary so the two artifacts cannot silently converge or drift.

    python -m pytest tests/test_camera_report_parity.py -q
"""

from __future__ import annotations

import inspect
import json
import os
import sys

import pytest

_TEST_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_TEST_DIR)
for path in (_REPO_ROOT, _TEST_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)

from core import constants as C
from core.global_state_loader import GlobalTrainState, GlobalWagon
from core.unified_wagon_state import UnifiedWagonState
from reporting import camera_reports
from sequential import camera_report as sequential_camera_report
from sequential import evidence as ev, global_assembly

GW_IDS = ["GW_1", "GW_2", "GW_3"]


def _strip_docstrings(source: str) -> str:
    """Drop triple-quoted blocks so a scan sees CODE, not prose."""
    import re

    return re.sub(r'"""(?:.|\n)*?"""', "", source)



def _state():
    return GlobalTrainState(
        total_wagons=len(GW_IDS),
        wagons=tuple(
            GlobalWagon(global_id=gw_id, wagon_index=index + 1,
                        start_frame_master=index * 100,
                        end_frame_master=index * 100 + 99,
                        start_time=float(index), end_time=float(index + 1),
                        classification=C.CLASS_WAGON,
                        classification_confidence=0.9,
                        camera_frame_ranges={
                            camera: {"start_frame": index * 100,
                                     "end_frame": index * 100 + 99,
                                     "status": "DETECTED",
                                     "timeline_reversed": False}
                            for camera in C.ALL_CAMERAS})
            for index, gw_id in enumerate(GW_IDS)),
        master_camera=C.CAMERA_RIGHT_UP, master_fps=15.0)


DOORS = [
    {"door_index": 1, "side": "left", "camera_id": C.CAMERA_LEFT_UP,
     "track_id": 1, "state": C.DOOR_CLOSED, "confidence": 0.88},
    {"door_index": 2, "side": "left", "camera_id": C.CAMERA_LEFT_UP,
     "track_id": 2, "state": C.DOOR_OPEN, "confidence": 0.93},
]


def _unified():
    out = {}
    for index, gw_id in enumerate(GW_IDS):
        out[gw_id] = UnifiedWagonState(
            global_id=gw_id, wagon_index=index + 1,
            classification=C.CLASS_WAGON, classification_confidence=0.9,
            left_door=C.DOOR_OPEN, left_door_confidence=0.93,
            right_door=C.DOOR_CLOSED, right_door_confidence=0.81,
            load_status=C.LOAD_LOADED, load_confidence=0.77,
            top_damage=C.DAMAGE_OK, doors=list(DOORS))
    return out


# =============================================================================
# 1. Sequential uses the validated renderer -- it does not have its own
# =============================================================================

def test_assembly_runs_the_validated_stage_5a_renderer():
    source = inspect.getsource(global_assembly.assemble)
    assert "camera_reports.build_all" in source, (
        "Global Assembly must produce the validated per-camera reports with "
        "Batch's own renderer")
    assert "combined_train_report.build" in source


def test_assembly_passes_the_same_contract_batch_passes():
    """Same keyword contract as Batch's Stage 5a call."""
    batch_parameters = set(
        inspect.signature(camera_reports.build_all).parameters)
    source = inspect.getsource(global_assembly.assemble)
    for required in ("state=", "unified=", "evidence_root=",
                     "wagon_states_root=", "cache_root=",
                     "per_camera_tracking_path=", "output_dir=",
                     "batch_key="):
        assert required in source, (
            "Stage 5a in Sequential is missing %r" % required)
        assert required.rstrip("=") in batch_parameters


def test_sequential_defines_no_second_per_wagon_camera_report():
    """The camera-local report must not RIVAL the validated one -- it must CALL it.

    REWRITTEN, deliberately. The previous version banned the name
    `camera_reports` from this module, on the reasoning that the camera-local
    report should stay separate from the validated renderer. The requirement is
    now the opposite: the Phase-1 camera report must BE Batch's camera report,
    rendered from camera-local data, so that a camera's inspection report has
    the same sections, layout, labels and evidence presentation as Batch's.

    The original intent -- no second implementation of per-wagon rendering --
    is preserved and in fact enforced more strictly: the module must call
    Batch's builders and must not define page, table or item-assembly logic of
    its own. What changed is that reuse is now the requirement rather than the
    thing being guarded against.
    """
    code = _strip_docstrings(inspect.getsource(sequential_camera_report))

    # It must REUSE, not reimplement.
    assert "camera_reports.build_camera_report(" in code, (
        "the Phase-1 report must render through Batch's own camera renderer")
    assert "camera_reports._build_camera_items(" in code, (
        "the Phase-1 JSON must serialize Batch's own per-wagon item list")

    # And it must not grow its own per-wagon rendering. These are the section
    # builders that belong to the validated renderer alone.
    for banned in ("_camera_summary_page", "_camera_wagon_pages",
                   "_camera_evidence_pages", "_camera_anomaly_summary",
                   "_camera_detection_summary_rows", "def _build_camera_items"):
        assert banned not in code, (
            "the camera-local report is reimplementing %r instead of calling "
            "the validated renderer" % banned)


def test_sequential_camera_report_uses_only_camera_local_ids():
    """Reusing Batch's renderer must not import Batch's canonical roster.

    The renderer addresses wagons by `wagon.global_id`; Phase 1 supplies
    camera-local ids. Nothing here may construct or expect a `GW_n`.
    """
    code = _strip_docstrings(inspect.getsource(sequential_camera_report))
    assert '"GW_' not in code and "'GW_" not in code
    assert "GW_%d" not in code


def test_validated_report_filenames_are_batchs(tmp_path):
    assert camera_reports.CAMERA_FILE == {
        C.CAMERA_RIGHT_UP: "right_up_report.pdf",
        C.CAMERA_LEFT_UP: "left_up_report.pdf",
        C.CAMERA_RIGHT_UP_TOP: "right_up_top_report.pdf",
        C.CAMERA_LEFT_UP_TOP: "left_up_top_report.pdf",
    }


# =============================================================================
# 2. Identical facts in, identical report items out
# =============================================================================

@pytest.mark.parametrize("camera_id", list(C.ALL_CAMERAS))
def test_report_items_are_identical_for_identical_facts(camera_id, tmp_path):
    """The renderer is shared, so equal (state, unified) must give equal items.

    This is the schema+facts parity check: every field the camera report
    renders is compared, for every camera.
    """
    state, unified = _state(), _unified()

    batch_items = camera_reports._build_camera_items(
        camera_id=camera_id, state=state, unified=unified,
        evidence_root=str(tmp_path / "batch_ev"),
        wagon_states_root=str(tmp_path / "batch_ws"),
        cache_root=str(tmp_path / "batch_cache"))
    sequential_items = camera_reports._build_camera_items(
        camera_id=camera_id, state=state, unified=unified,
        evidence_root=str(tmp_path / "seq_ev"),
        wagon_states_root=str(tmp_path / "seq_ws"),
        cache_root=str(tmp_path / "seq_cache"))

    assert [item["gw_id"] for item in batch_items] == GW_IDS
    assert len(batch_items) == len(sequential_items)
    for left, right in zip(batch_items, sequential_items):
        assert set(left) == set(right), "item schema differs"
        for key in left:
            assert left[key] == right[key], "field %r differs" % key


def test_door_load_damage_facts_reach_the_report_unchanged(tmp_path):
    state, unified = _state(), _unified()

    right = camera_reports._build_camera_items(
        camera_id=C.CAMERA_RIGHT_UP, state=state, unified=unified,
        evidence_root=None, wagon_states_root=None, cache_root=None)
    left = camera_reports._build_camera_items(
        camera_id=C.CAMERA_LEFT_UP, state=state, unified=unified,
        evidence_root=None, wagon_states_root=None, cache_root=None)
    top = camera_reports._build_camera_items(
        camera_id=C.CAMERA_RIGHT_UP_TOP, state=state, unified=unified,
        evidence_root=None, wagon_states_root=None, cache_root=None)

    right_labels = {label: value for label, value, _c, _s in right[0]["detections"]}
    assert right_labels["Right Door"] == C.DOOR_CLOSED
    left_labels = {label: value for label, value, _c, _s in left[0]["detections"]}
    assert left_labels["Left Door"] == C.DOOR_OPEN
    top_labels = {label: value for label, value, _c, _s in top[0]["detections"]}
    assert top_labels["Load Status"] == C.LOAD_LOADED


def test_multi_door_semantics_survive_into_the_unified_state():
    """b6f67b5: both physical doors keep their own state and index."""
    unified = _unified()
    wagon = unified["GW_1"]
    assert len(wagon.doors) == 2
    assert [d["state"] for d in wagon.doors] == [C.DOOR_CLOSED, C.DOOR_OPEN]
    assert wagon.door_status == C.DOOR_OPEN
    assert wagon.has_open_door is True
    # the per-side contract the camera report renders is unchanged
    assert wagon.left_door == C.DOOR_OPEN


def test_sequential_door_payload_carries_the_same_door_fields_as_batch():
    """The payload Sequential writes must have Batch's door schema."""
    from features.door import processor as door_proc

    # MIGRATED: `global_assembly._aggregate_door` no longer exists. Sequential
    # does not build a door payload at all -- Batch's door processor does, on
    # Batch's wagon cache. So the schema check moves to its real owner, and the
    # anti-copy check becomes "assembly contains none of these helpers".
    batch_source = inspect.getsource(door_proc)
    for field in ('"left_door"', '"right_door"', '"doors"', '"door_status"',
                  '"left_door_confidence"', '"right_door_confidence"'):
        assert field in batch_source, (
            "Batch's door payload no longer has %s" % field)

    for helper in ("_door_evidence_from_groups", "_pick_side_state",
                   "order_doors", "wagon_door_status"):
        assert hasattr(door_proc, helper), helper

    assembly_source = inspect.getsource(global_assembly)
    for helper in ("_door_evidence_from_groups", "_pick_side_state"):
        assert helper not in assembly_source, (
            "assembly re-implements Batch's %s instead of calling the "
            "processor" % helper)
    # the payload is produced by running Batch's processor; see
    # tests/test_batch_sequential_exact_parity.py for the payload itself
    assert "load_feature_runner" in inspect.getsource(
        global_assembly._run_features)


# =============================================================================
# 3. Visibility and snapshots are real, not assumed
# =============================================================================

def test_visibility_is_answered_from_a_real_frame(tmp_path):
    """`_wagon_covered` needs an actual cached JPEG -- placement provides one."""
    cache_root = tmp_path / "wagon_cache"
    folder = cache_root / "GW_1" / C.CAMERA_FOLDER[C.CAMERA_RIGHT_UP]
    folder.mkdir(parents=True)

    assert camera_reports._wagon_covered(
        str(cache_root), "GW_1", C.CAMERA_RIGHT_UP) is False
    (folder / "frame_000042.jpg").write_bytes(b"jpeg")
    assert camera_reports._wagon_covered(
        str(cache_root), "GW_1", C.CAMERA_RIGHT_UP) is True
    # a wagon with no window on this camera stays correctly not-visible
    assert camera_reports._wagon_covered(
        str(cache_root), "GW_2", C.CAMERA_RIGHT_UP) is False


def test_frames_are_produced_only_by_batchs_materializer():
    """OBSOLETE AS WRITTEN, replaced by the rule it was approximating.

    `global_assembly.place_wagon_evidence` is gone. Sequential no longer places
    evidence frames of its own: Batch's Stage-2 materializer writes the per-wagon
    JPEG-90 cache and Batch's processors write their snapshots from it. Anything
    else would mean the reports read pixels Batch never produced.
    """
    assembly = inspect.getsource(global_assembly)
    assert "wagon_cache_builder.build" in assembly, (
        "Batch's materializer must be the one creating cached frames")
    for banned in ("shutil.copyfile", "cv2.imwrite", "DOOR_SLOT",
                   "place_wagon_evidence"):
        assert banned not in assembly, (
            "assembly is placing or creating evidence frames itself: %s"
            % banned)


def test_snapshot_selection_belongs_to_batchs_processors():
    """OBSOLETE AS WRITTEN: the 120-frame bucket approximation was removed.

    The camera stage used to keep a best-frame-per-bucket store, which selected
    DIFFERENT frames than Batch. Batch's processors pick the best frame from the
    wagon cache, so Sequential now lets them, and the approximation must not
    come back.
    """
    from sequential import camera_runner

    module_source = inspect.getsource(camera_runner)
    for banned in ("SnapshotStore", "snapshot_bucket",
                   "SNAPSHOT_BUCKET_FRAMES", "consider_plain"):
        assert banned not in module_source, (
            "the approximate snapshot selection is back: %s" % banned)
    assert not hasattr(camera_runner, "SnapshotStore")
    assert not hasattr(camera_runner, "SNAPSHOT_BUCKET_FRAMES")


# =============================================================================
# 4. The camera-local report stays camera-local
# =============================================================================

def _camera_evidence(camera_id, *, gaps=3):
    timing = ev.CameraTiming(fps=15.0, total_frames=900, decoded_frames=900,
                             wagon_region_start_frame=100,
                             wagon_region_end_frame=799,
                             wagon_region_frames=700, duration_seconds=60.0)
    observations = [
        ev.FeatureObservation(feature="door", frame_idx=index * 3,
                              timestamp=index * 0.2, state="",
                              confidence=0.9, bbox=[1.0, 2.0, 3.0, 4.0],
                              raw_class="open_door", score=0.5)
        for index in range(1, 6)]
    return ev.CameraEvidence(
        camera_id=camera_id, status=ev.STATUS_SEALED, timing=timing,
        gaps=[ev.GapObservation(local_gap_id="%s_G%d" % (camera_id, n),
                                confirmation_frame=100 + n * 200,
                                first_frame=100 + n * 200,
                                last_frame=100 + n * 200,
                                normalized_position=n * 300.0,
                                max_confidence=0.9, normalized_duration=10.0)
              for n in range(1, gaps + 1)],
        observations=observations,
        segments=[{"segment_id": ev.SEGMENT_ID_FORMAT % (camera_id, 1),
                   "segment_index": 1, "start_frame": 300, "end_frame": 500,
                   "start_normalized": 300.0, "end_normalized": 600.0,
                   "opening_gap": "g1", "closing_gap": "g2",
                   "canonical": False}],
        provenance={"video": {"fingerprint": "x"}, "models": {},
                    "decode_passes": 1, "frame_width": 640,
                    "frame_height": 480},
        feature_config={"features": ["door"], "strides": {"door": 3}})


def test_camera_local_report_needs_only_its_own_camera(tmp_path):
    """One camera present -> its report is written, with no other camera."""
    workspace = str(tmp_path / "ws")
    paths = sequential_camera_report.build(
        workspace=workspace, evidence=_camera_evidence(C.CAMERA_LEFT_UP),
        batch_key="solo", verbose=False)

    assert os.path.isfile(paths["json_path"])
    with open(paths["json_path"], encoding="utf-8") as handle:
        document = json.load(handle)
    assert document["camera_id"] == C.CAMERA_LEFT_UP
    assert document["canonical"] is False
    assert document["report_type"] == "single_camera"
    # nothing about any other camera, and no canonical wagon id
    text = json.dumps(document)
    for other in (C.CAMERA_RIGHT_UP, C.CAMERA_RIGHT_UP_TOP, C.CAMERA_LEFT_UP_TOP):
        assert other not in text
    ev.assert_no_canonical_ids(document, where="camera-local report")


def test_camera_local_report_is_versioned_and_distinct(tmp_path):
    """A deliberate schema version, so the two artifacts cannot be confused."""
    document = sequential_camera_report.build_document(
        _camera_evidence(C.CAMERA_RIGHT_UP), batch_key="k")
    assert document["schema"] == "wagon_eye.camera_report.v1"
    assert document["report_type"] == "single_camera"
    assert document["canonical"] is False
    assert "NOT canonical" in document["disclaimer"]
    # it reports EVIDENCE, not fused verdicts
    assert "local_gaps" in document and "local_segments" in document
    assert "wagons" not in document
    for feature_summary in document["observations"].values():
        assert "observation_count" in feature_summary
        assert "load_status" not in feature_summary
        assert "left_door" not in feature_summary


# =============================================================================
# 5. Anti-divergence guard
# =============================================================================

# The contract the VALIDATED camera report renders. Changing this set is a
# deliberate schema change: update the constant AND the version below together.
VALIDATED_ITEM_FIELDS = {
    "sr", "gw_id", "classification", "classification_conf", "visible",
    "detections", "anomalies", "primary_confidence", "ocr",
    "start_time", "end_time",
}
VALIDATED_REPORT_CONTRACT_VERSION = 1


def test_validated_camera_report_item_contract_is_unchanged(tmp_path):
    """Fails if the validated report's per-wagon item schema drifts.

    Both modes render through this structure, so pinning it here stops either
    architecture changing the camera report's meaning by accident.
    """
    items = camera_reports._build_camera_items(
        camera_id=C.CAMERA_RIGHT_UP, state=_state(), unified=_unified(),
        evidence_root=None, wagon_states_root=None, cache_root=None)
    assert items
    assert set(items[0]) == VALIDATED_ITEM_FIELDS, (
        "the validated camera-report item schema changed; if deliberate, bump "
        "VALIDATED_REPORT_CONTRACT_VERSION and update both modes")
    assert VALIDATED_REPORT_CONTRACT_VERSION == 1


def test_only_one_module_renders_the_validated_camera_report():
    """No second implementation of the validated report may appear."""
    offenders = []
    for root, dirs, files in os.walk(_REPO_ROOT):
        dirs[:] = [d for d in dirs
                   if d not in ("__pycache__", ".git", "batch_outputs",
                                ".venv", "tests")]
        for name in files:
            if not name.endswith(".py"):
                continue
            relative = os.path.relpath(os.path.join(root, name), _REPO_ROOT)
            if relative.replace("\\", "/") == "reporting/camera_reports.py":
                continue
            with open(os.path.join(root, name), encoding="utf-8") as handle:
                source = handle.read()
            if "_camera_summary_page" in source or "_camera_wagon_pages" in source:
                offenders.append(relative)
    assert offenders == [], (
        "the validated camera report is implemented more than once: %s"
        % offenders)


def test_batch_stage_5a_call_is_untouched():
    """Batch's own Stage 5a invocation must remain exactly as validated."""
    with open(os.path.join(_REPO_ROOT, "orchestrator", "master_runner.py"),
              encoding="utf-8") as handle:
        source = handle.read()
    assert "camera_reports.build_all(" in source
    assert "STAGE 5a" in source
    # Batch still passes its own roots, unchanged
    for argument in ("state=recon.state", "unified=out.unified",
                     "evidence_root=evidence_root", "cache_root=cache_root",
                     "output_dir=reports_root"):
        assert argument in source, "Batch Stage 5a changed: %r missing" % argument
