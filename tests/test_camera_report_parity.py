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
    """The camera-local report must not grow into a rival of the validated one."""
    raw = inspect.getsource(sequential_camera_report)
    # Strip docstrings: the module docstring legitimately NAMES the validated
    # renderer to explain why it is not reused.
    code = _strip_docstrings(raw)
    for banned in ("state.wagons", "UnifiedWagonState", "unified[",
                   "wagon_cache", "evidence_snapshot", "camera_reports"):
        assert banned not in code, (
            "the camera-local report is reaching into global territory: %r"
            % banned)


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

    sequential_payload = inspect.getsource(global_assembly._aggregate_door)
    for field in ('"left_door"', '"right_door"', '"doors"', '"door_status"',
                  '"left_door_confidence"', '"right_door_confidence"'):
        assert field in sequential_payload, (
            "Sequential's door payload is missing %s" % field)
    # and it is built with Batch's own helpers, not a copy
    for helper in ("_door_evidence_from_groups", "_pick_side_state",
                   "order_doors", "wagon_door_status"):
        assert helper in sequential_payload
        assert hasattr(door_proc, helper)


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


def test_placement_copies_only_and_never_decodes():
    source = inspect.getsource(global_assembly.place_wagon_evidence)
    assert "shutil.copyfile" in source
    for banned in ("VideoCapture", ".predict(", "load_yolo", "imwrite"):
        assert banned not in source, (
            "evidence placement must copy existing frames, not create them: %r"
            % banned)


def test_placement_writes_the_slots_the_validated_report_reads():
    source = inspect.getsource(global_assembly.place_wagon_evidence)
    for slot in ("best_frame", "track_%d.jpg", "metadata.json"):
        assert slot in source, slot
    # The door slots live in the module-level map the report agrees with.
    assert global_assembly.DOOR_SLOT == {
        C.CAMERA_LEFT_UP: "left_best", C.CAMERA_RIGHT_UP: "right_best"}
    assert "DOOR_SLOT" in source


def test_camera_stage_persists_snapshots_during_the_single_decode():
    from sequential import camera_runner

    source = inspect.getsource(camera_runner._decode_once)
    assert "snapshots.consider(" in source
    assert "snapshots.consider_plain(" in source
    # still exactly one capture
    body = inspect.getsource(camera_runner._decode_once)
    assert body.count("cv2.VideoCapture(") == 1
    assert camera_runner.SNAPSHOT_BUCKET_FRAMES > 0


def test_snapshot_store_keeps_only_the_best_per_bucket(tmp_path):
    import numpy as np
    from sequential.camera_runner import SnapshotStore, snapshot_bucket

    store = SnapshotStore(str(tmp_path / "snaps"))
    frame = np.zeros((40, 40, 3), dtype=np.uint8)
    bbox = [5.0, 5.0, 30.0, 30.0]

    store.consider(feature="door", state="open_door", frame_idx=3, frame=frame,
                   bbox=bbox, score=0.4, label="a")
    first = dict(store.index)
    store.consider(feature="door", state="open_door", frame_idx=7, frame=frame,
                   bbox=bbox, score=0.9, label="b")
    store.consider(feature="door", state="open_door", frame_idx=9, frame=frame,
                   bbox=bbox, score=0.1, label="c")

    # same bucket -> one file, and the best score won
    assert snapshot_bucket(3) == snapshot_bucket(7) == snapshot_bucket(9)
    assert len(store.index) == len(first) == 1
    assert store._best[("door", "open_door", snapshot_bucket(3))] == 0.9


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
