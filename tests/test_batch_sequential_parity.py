"""Batch and Sequential must agree, and here is where that is proved.

The two modes reach a wagon by different routes -- Batch inspects cached JPEGs
per wagon, Sequential inspects frames once and interprets later -- so the only
thing that makes their answers comparable is that every DECISION is made by the
same code. These tests pin each decision point:

    gating       Door `min_conf` and Damage `_filter_detections_for_top`
    aggregation  the same EvidenceAggregator groups -> the same doors
    Load         the same ratio rule, the same `used` denominator
    alignment    the engine's estimate_alignment, forward AND reversed
    projection   canonical RIGHT_UP gaps landing on the right camera frames
    ownership    core.wagon_ownership, one wagon per observation

Where a number is produced by a shared function, the test calls BOTH paths and
compares. Where it is produced by structure, the test asserts the structure.

    python -m pytest tests/test_batch_sequential_parity.py -q
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import pytest

_TEST_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_TEST_DIR)
for path in (_REPO_ROOT, _TEST_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)

from core import constants as C
from core import wagon_ownership
from core.global_state_loader import (
    load_global_train_state, verify_roster_integrity)
from sequential import camera_runner, evidence as ev, global_assembly

from test_sequential_architecture import STUB_ENGINE


# =============================================================================
# helpers
# =============================================================================

WIDTH, HEIGHT = 640, 480


def _frame():
    return np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)


class _Arr:
    def __init__(self, value):
        self._value = value

    def cpu(self):
        return self

    def numpy(self):
        return np.asarray(self._value)


class _Boxes:
    def __init__(self, bboxes, confs, cls_ids):
        self.xyxy = _Arr(bboxes)
        self.conf = _Arr(confs)
        self.cls = _Arr(cls_ids)

    def __len__(self):
        return len(self.conf.numpy())


class _Result:
    def __init__(self, boxes):
        self.boxes = boxes


class _FixedYolo:
    """Returns one fixed detection set for every frame."""

    def __init__(self, bboxes, confs, cls_ids, names):
        self._payload = (bboxes, confs, cls_ids)
        self.names = names

    def __call__(self, frame, **kwargs):
        return [_Result(_Boxes(*self._payload))]


@pytest.fixture
def stub_engine_dir(tmp_path):
    engine_dir = tmp_path / "global_wagon_app"
    engine_dir.mkdir()
    for name, source in STUB_ENGINE.items():
        (engine_dir / name).write_text(source, encoding="utf-8")
    return str(engine_dir)


# =============================================================================
# 1. Door gating parity
# =============================================================================

DOOR_NAMES = {0: "closed_door", 1: "open_door"}


def test_door_gate_value_comes_from_batch_config():
    from features.door.processor import TrackerConfig

    assert camera_runner.door_confidence_floor() == float(
        TrackerConfig().closed_confidence_threshold)


def test_door_gate_keeps_exactly_what_batch_keeps():
    """Same arrays in -> same survivors as Batch's `keep = confs >= min_conf`."""
    from features.door.processor import TrackerConfig

    floor = float(TrackerConfig().closed_confidence_threshold)
    bboxes = [[100.0, 80.0, 200.0, 200.0], [220.0, 80.0, 320.0, 200.0],
              [340.0, 80.0, 440.0, 200.0], [10.0, 10.0, 40.0, 40.0]]
    confs = [floor + 0.10, floor - 0.01, floor, 0.05]     # keep, drop, keep, drop
    cls_ids = [1, 0, 0, 1]

    batch_keep = [index for index, conf in enumerate(confs) if conf >= floor]

    model = _FixedYolo(bboxes, confs, cls_ids, DOOR_NAMES)
    observations = camera_runner._observe(
        "door", model, _frame(), 9, 0.6, WIDTH, HEIGHT)

    assert len(observations) == len(batch_keep) == 2
    assert [round(o.confidence, 6) for o in observations] == \
        [round(confs[i], 6) for i in batch_keep]
    assert [o.bbox for o in observations] == [bboxes[i] for i in batch_keep]
    assert [o.raw_class for o in observations] == \
        [DOOR_NAMES[cls_ids[i]] for i in batch_keep]


def test_door_gate_boundary_is_inclusive_on_both_sides():
    """`>=`, not `>`: a detection exactly at the floor survives in Batch."""
    from features.door.processor import TrackerConfig

    floor = float(TrackerConfig().closed_confidence_threshold)
    model = _FixedYolo([[100.0, 80.0, 200.0, 200.0]], [floor], [0], DOOR_NAMES)
    assert len(camera_runner._observe(
        "door", model, _frame(), 0, 0.0, WIDTH, HEIGHT)) == 1

    model = _FixedYolo([[100.0, 80.0, 200.0, 200.0]],
                       [np.nextafter(floor, 0.0)], [0], DOOR_NAMES)
    assert camera_runner._observe(
        "door", model, _frame(), 0, 0.0, WIDTH, HEIGHT) == []


# =============================================================================
# 2. Damage gating parity
# =============================================================================

DAMAGE_NAMES = {0: "dent", 1: "rust"}


def test_damage_gate_value_comes_from_batch_constant():
    assert camera_runner.damage_confidence_floor() == float(C.CONF_DAMAGE)


def test_damage_gate_output_is_batchs_filter_output():
    """Sequential must produce exactly `_filter_detections_for_top`'s survivors.

    The cases deliberately span every reason Batch drops a detection: below the
    floor, too small, too large, and edge-zone without the confidence bypass.
    """
    from features.damage.processor import _filter_detections_for_top

    floor = float(C.CONF_DAMAGE)
    bboxes = [
        [200.0, 160.0, 440.0, 320.0],   # centred, ~12% area  -> keep
        [200.0, 160.0, 440.0, 320.0],   # same box, below floor -> drop
        [0.0, 0.0, 8.0, 8.0],           # far too small        -> drop
        [0.0, 0.0, 640.0, 480.0],       # whole frame          -> drop
        [0.0, 200.0, 60.0, 300.0],      # left edge, low conf  -> drop
        [0.0, 200.0, 60.0, 300.0],      # left edge, high conf -> keep (bypass)
    ]
    confs = [floor + 0.2, floor - 0.05, 0.95, 0.95, floor + 0.01, 0.95]
    cls_ids = [0, 0, 0, 0, 1, 1]

    keep_boxes, keep_confs, keep_cls = _filter_detections_for_top(
        np.asarray(bboxes, dtype=float), np.asarray(confs, dtype=float),
        np.asarray(cls_ids, dtype=int), DAMAGE_NAMES, WIDTH, HEIGHT, floor)

    model = _FixedYolo(bboxes, confs, cls_ids, DAMAGE_NAMES)
    observations = camera_runner._observe(
        "damage", model, _frame(), 3, 0.2, WIDTH, HEIGHT)

    assert len(observations) == len(keep_confs)
    assert [round(o.confidence, 6) for o in observations] == \
        [round(float(c), 6) for c in keep_confs]
    assert [o.bbox for o in observations] == [
        [float(v) for v in box] for box in keep_boxes]
    assert [o.raw_class for o in observations] == [
        DAMAGE_NAMES[int(c)] for c in keep_cls]
    # the filter really did reject things -- otherwise this proves nothing
    assert len(observations) < len(confs)


def test_sequential_calls_batchs_damage_filter_not_a_copy():
    import inspect

    source = inspect.getsource(camera_runner._observe)
    assert "_filter_detections_for_top" in source
    for banned in ("_EDGE_X_MIN_RATIO", "_AREA_MIN_RATIO", "area_ratio"):
        assert banned not in source, (
            "the damage filter was reimplemented instead of reused: %r" % banned)


def test_load_has_no_detection_gate_in_either_mode():
    """Load votes on every sampled frame in Batch; Sequential must not gate it."""
    class _Classifier:
        def __call__(self, *a, **kw):
            raise AssertionError("load must not go through the detector path")

    from features import _common

    original = _common.run_classification
    try:
        _common.run_classification = lambda model, frame: ("loaded", 0.01)
        observations = camera_runner._observe(
            "load", _Classifier(), _frame(), 4, 0.3, WIDTH, HEIGHT)
    finally:
        _common.run_classification = original

    assert len(observations) == 1, "a low-confidence load vote still counts"
    assert observations[0].raw_class == "loaded"


# =============================================================================
# 3. Load aggregation parity
# =============================================================================

def _load_observations(classes):
    return [ev.FeatureObservation(feature="load", frame_idx=index * 2,
                                  timestamp=0.0, state="", confidence=0.9,
                                  raw_class=name)
            for index, name in enumerate(classes)]


def _batch_load_rule(classes, confidence=0.9):
    """Batch's rule, transcribed here ONLY to compare against."""
    from features.load.processor import _LOADED_RATIO_THRESHOLD, _canonical_load

    loaded = [confidence for name in classes
              if _canonical_load(name) == C.LOAD_LOADED]
    empty = [confidence for name in classes
             if _canonical_load(name) == C.LOAD_EMPTY]
    used = len(classes)
    if used == 0:
        return C.NO_DATA, 0.0
    if (len(loaded) / max(1, used)) > _LOADED_RATIO_THRESHOLD and loaded:
        return C.LOAD_LOADED, sum(loaded) / len(loaded)
    if empty:
        return C.LOAD_EMPTY, sum(empty) / len(empty)
    return C.NO_DATA, 0.0


@pytest.mark.parametrize("classes", [
    ["loaded"] * 5,
    ["empty"] * 5,
    ["loaded", "loaded", "empty", "empty", "empty"],          # ratio 0.40 > 0.35
    ["loaded", "empty", "empty", "empty", "empty", "empty"],  # ratio 0.17
    # a third class dilutes the denominator: 3/9 = 0.33, NOT 3/6 = 0.50
    ["loaded"] * 3 + ["empty"] * 3 + ["unknown"] * 3,
    ["unknown"] * 4,
])
def test_load_verdict_matches_batch(classes):
    expected_state, expected_conf = _batch_load_rule(classes)
    payload = global_assembly._aggregate_load(
        "GW_1", {C.CAMERA_RIGHT_UP_TOP: _load_observations(classes)})
    assert payload["load_status"] == expected_state
    assert payload["load_confidence"] == pytest.approx(
        round(expected_conf, 4), abs=1e-4)


def test_load_denominator_counts_every_sampled_frame():
    """The subtle one: Batch's `total` is `used`, not loaded+empty."""
    classes = ["loaded"] * 3 + ["empty"] * 3 + ["unknown"] * 3
    payload = global_assembly._aggregate_load(
        "GW_1", {C.CAMERA_RIGHT_UP_TOP: _load_observations(classes)})
    assert payload["frames_used"] == 9
    # 3/9 = 0.33 is NOT above 0.35, so this wagon is EMPTY, not LOADED.
    assert payload["load_status"] == C.LOAD_EMPTY


def test_load_uses_batchs_threshold_constant():
    import inspect
    source = inspect.getsource(global_assembly._aggregate_load)
    assert "_LOADED_RATIO_THRESHOLD" in source
    assert "0.35" not in source, "the threshold must not be restated"


# =============================================================================
# 4. Door aggregation parity (multi-door preserved)
# =============================================================================

def _door_observations(per_frame):
    """`[(frame, raw_class, bbox), ...]` -> Sequential observations."""
    from core.frame_quality import snapshot_score

    out = []
    for frame_idx, raw_class, bbox in per_frame:
        out.append(ev.FeatureObservation(
            feature="door", frame_idx=frame_idx, timestamp=frame_idx / 15.0,
            state="", confidence=0.9, bbox=list(bbox), raw_class=raw_class,
            score=float(snapshot_score(list(bbox), 0.9, 1.0, WIDTH, HEIGHT))))
    return out


LEFT_DOOR_BOX = (80.0, 120.0, 220.0, 380.0)
RIGHT_DOOR_BOX = (400.0, 120.0, 540.0, 380.0)


def test_door_aggregation_uses_the_same_helpers_as_batch():
    import inspect
    source = inspect.getsource(global_assembly._aggregate_door)
    for shared in ("_door_evidence_from_groups", "_pick_side_state",
                   "order_doors", "wagon_door_status", "EvidenceAggregator",
                   "DOOR_LABEL_TO_STATE"):
        assert shared in source, (
            "Door aggregation must reuse Batch's %r, not a copy" % shared)


def test_two_distinct_doors_survive_assembly_with_their_own_states():
    """b6f67b5 must hold on the Sequential route too."""
    observations = _door_observations(
        [(f, "closed_door", LEFT_DOOR_BOX) for f in (3, 6, 9)]
        + [(f, "open_door", RIGHT_DOOR_BOX) for f in (3, 6, 9)])

    payload = global_assembly._aggregate_door(
        "GW_1", {C.CAMERA_LEFT_UP: observations}, (WIDTH, HEIGHT))

    states = [door["state"] for door in payload["doors"]]
    assert len(payload["doors"]) == 2, "two physical doors must stay two"
    assert C.DOOR_CLOSED in states and C.DOOR_OPEN in states
    assert payload["door_status"] == C.DOOR_OPEN
    assert [door["door_index"] for door in payload["doors"]] == [1, 2]


def test_many_frames_of_one_door_stay_one_door():
    observations = _door_observations(
        [(f, "open_door", LEFT_DOOR_BOX) for f in range(3, 40, 3)])
    payload = global_assembly._aggregate_door(
        "GW_1", {C.CAMERA_LEFT_UP: observations}, (WIDTH, HEIGHT))
    assert len(payload["doors"]) == 1
    assert payload["doors"][0]["state"] == C.DOOR_OPEN


def test_door_side_state_matches_batchs_picker():
    from features.door.processor import _pick_side_state

    observations = _door_observations(
        [(f, "closed_door", LEFT_DOOR_BOX) for f in (3, 6, 9)]
        + [(f, "open_door", RIGHT_DOOR_BOX) for f in (3, 6, 9)])
    payload = global_assembly._aggregate_door(
        "GW_1", {C.CAMERA_LEFT_UP: observations}, (WIDTH, HEIGHT))

    decisions = [{"state": d["state"], "confidence": d["confidence"],
                  "total_hits": d["total_hits"]} for d in payload["doors"]]
    expected_state, _confidence = _pick_side_state(decisions)
    assert payload["left_door"] == expected_state


def test_door_raw_class_mapping_is_batchs_table():
    """`open_door`/`closed_door` must map through Batch's own table."""
    from features.door import processor as door_proc

    for raw in ("open_door", "closed_door"):
        expected = C.DOOR_LABEL_TO_STATE.get(raw, door_proc._canonical(raw))
        payload = global_assembly._aggregate_door(
            "GW_1",
            {C.CAMERA_LEFT_UP: _door_observations(
                [(f, raw, LEFT_DOOR_BOX) for f in (3, 6, 9)])},
            (WIDTH, HEIGHT))
        assert payload["doors"][0]["state"] == expected


# =============================================================================
# 5. Alignment: forward and reversed support cameras
# =============================================================================

# Irregular spacing: evenly spaced gaps are genuinely un-orientable, so a
# reversal test on them would prove nothing.
CANONICAL = [0.0, 90.0, 210.0, 430.0, 700.0, 1000.0]
REGION_START, REGION_FRAMES = 100, 1501


def _evidence_document(camera_id, positions, *, status=ev.STATUS_SEALED):
    gaps = []
    for index, position in enumerate(positions, start=1):
        frame = REGION_START + int(round(position / 1000.0 * (REGION_FRAMES - 1)))
        gaps.append({
            "local_gap_id": "%s_G%d" % (camera_id, index),
            "confirmation_frame": frame, "first_frame": frame,
            "last_frame": frame, "normalized_position": float(position),
            "max_confidence": 0.9, "average_confidence": 0.9,
            "frame_count": 1, "normalized_duration": 12.0,
        })
    return {
        "camera_id": camera_id, "schema_version": ev.SCHEMA_VERSION,
        "status": status,
        "timing": {"fps": 15.0, "total_frames": REGION_START + REGION_FRAMES + 50,
                   "decoded_frames": REGION_START + REGION_FRAMES + 50,
                   "wagon_region_start_frame": REGION_START,
                   "wagon_region_end_frame": REGION_START + REGION_FRAMES - 1,
                   "wagon_region_frames": REGION_FRAMES,
                   "duration_seconds": 100.0},
        "gaps": gaps, "observations": [],
        "classification_timeline": [], "segments": [],
        "provenance": {"video": {"fingerprint": "x"}, "models": {},
                       "decode_passes": 1, "frame_width": WIDTH,
                       "frame_height": HEIGHT},
        "feature_config": {"features": []}, "diagnostics": {}, "snapshots": {},
    }


def _seal_document(camera_id):
    return {"schema_version": ev.SEAL_SCHEMA_VERSION,
            "evidence_schema_version": ev.SCHEMA_VERSION,
            "camera_id": camera_id, "status": ev.STATUS_SEALED,
            "unique_gap_count": len(CANONICAL), "observation_count": 0,
            "video_fingerprint": {"fingerprint": "x"}, "model_fingerprints": {},
            "config_fingerprint": "c", "feature_config": {"features": []},
            "reports": {}, "timing": {}, "frame_count": 0, "fps": 15.0}


def _write_camera(workspace, camera_id, positions):
    directory = ev.camera_evidence_dir(workspace, camera_id)
    os.makedirs(directory, exist_ok=True)
    with open(ev.evidence_path(workspace, camera_id), "w",
              encoding="utf-8") as handle:
        json.dump(_evidence_document(camera_id, positions), handle)
    with open(ev.seal_path(workspace, camera_id), "w",
              encoding="utf-8") as handle:
        json.dump(_seal_document(camera_id), handle)


def _assemble(workspace, stub_engine_dir, layout):
    for camera_id, positions in layout.items():
        _write_camera(workspace, camera_id, positions)
    return global_assembly.assemble(
        workspace=workspace, repo_root=_REPO_ROOT, batch_key="parity",
        engine_dir=stub_engine_dir, verbose=False)


def _mirrored(positions):
    """The same physical gaps seen by a camera running backwards."""
    return sorted(1000.0 - p for p in positions)


def test_forward_support_camera_is_not_marked_reversed(tmp_path,
                                                       stub_engine_dir):
    workspace = str(tmp_path / "ws")
    result = _assemble(workspace, stub_engine_dir, {
        C.CAMERA_RIGHT_UP: CANONICAL,
        C.CAMERA_LEFT_UP: CANONICAL,
    })
    assert result.ready, result.reason
    assert result.diagnostics["alignments"][C.CAMERA_LEFT_UP]["is_reversed"] \
        is False
    state = load_global_train_state(result.state_json_path)
    for wagon in state.wagons:
        assert wagon.camera_frame_ranges[C.CAMERA_LEFT_UP][
            "timeline_reversed"] is False


def test_reversed_support_camera_is_detected(tmp_path, stub_engine_dir):
    """A camera whose gap sequence is mirrored must be adopted as reversed."""
    workspace = str(tmp_path / "ws")
    result = _assemble(workspace, stub_engine_dir, {
        C.CAMERA_RIGHT_UP: CANONICAL,
        C.CAMERA_LEFT_UP_TOP: _mirrored(CANONICAL),
    })
    assert result.ready, result.reason
    alignment = result.diagnostics["alignments"][C.CAMERA_LEFT_UP_TOP]
    assert alignment["is_reversed"] is True, (
        "the mirrored gap sequence was not recognised as reversed")
    assert alignment["matched_count"] == len(CANONICAL)


def test_reversed_camera_windows_run_backwards(tmp_path, stub_engine_dir):
    """As the master advances, a reversed camera's frames must DECREASE."""
    workspace = str(tmp_path / "ws")
    result = _assemble(workspace, stub_engine_dir, {
        C.CAMERA_RIGHT_UP: CANONICAL,
        C.CAMERA_LEFT_UP_TOP: _mirrored(CANONICAL),
    })
    state = load_global_train_state(result.state_json_path)

    master_starts, reversed_starts = [], []
    for wagon in state.wagons:
        master_starts.append(
            wagon.camera_frame_ranges[C.CAMERA_RIGHT_UP]["start_frame"])
        reversed_starts.append(
            wagon.camera_frame_ranges[C.CAMERA_LEFT_UP_TOP]["start_frame"])

    assert master_starts == sorted(master_starts)
    assert reversed_starts == sorted(reversed_starts, reverse=True), (
        "a reversed camera's windows must descend against the master")
    for wagon in state.wagons:
        entry = wagon.camera_frame_ranges[C.CAMERA_LEFT_UP_TOP]
        assert entry["timeline_reversed"] is True
        assert entry["start_frame"] <= entry["end_frame"]


def test_canonical_gaps_project_onto_the_right_reversed_frames(
        tmp_path, stub_engine_dir):
    """Canonical gap k must land where the reversed camera actually saw it."""
    workspace = str(tmp_path / "ws")
    mirrored = _mirrored(CANONICAL)
    result = _assemble(workspace, stub_engine_dir, {
        C.CAMERA_RIGHT_UP: CANONICAL,
        C.CAMERA_LEFT_UP_TOP: mirrored,
    })
    state = load_global_train_state(result.state_json_path)

    # The camera's own frame for the mirror of canonical position p.
    def expected_frame(position):
        mirror = 1000.0 - position
        return REGION_START + int(round(mirror / 1000.0 * (REGION_FRAMES - 1)))

    for index, gap in enumerate(state.global_gaps):
        entry = gap["cameras"][C.CAMERA_LEFT_UP_TOP]
        assert entry["frame"] == pytest.approx(
            expected_frame(CANONICAL[index]), abs=2), (
            "canonical gap %d mis-projected into the reversed camera" % index)


def test_reversal_is_not_adopted_without_proof(tmp_path, stub_engine_dir):
    """A forward camera must never be flipped just because it could be."""
    workspace = str(tmp_path / "ws")
    result = _assemble(workspace, stub_engine_dir, {
        C.CAMERA_RIGHT_UP: CANONICAL,
        C.CAMERA_LEFT_UP: CANONICAL,
        C.CAMERA_RIGHT_UP_TOP: CANONICAL,
    })
    for camera_id in (C.CAMERA_LEFT_UP, C.CAMERA_RIGHT_UP_TOP):
        assert result.diagnostics["alignments"][camera_id]["is_reversed"] is False


def test_assembly_uses_the_engine_estimator_not_a_second_algorithm():
    import inspect
    source = inspect.getsource(global_assembly.align_camera)
    assert "estimate_alignment" in source
    assert "project_to_camera" in source
    for banned in ("np.polyfit", "for i, m in enumerate", "tolerance"):
        assert banned not in source, (
            "a second alignment algorithm appeared in assembly: %r" % banned)


# =============================================================================
# 6. Canonical roster + ownership parity
# =============================================================================

def test_canonical_roster_is_the_master_gaps_minus_one(tmp_path,
                                                       stub_engine_dir):
    workspace = str(tmp_path / "ws")
    result = _assemble(workspace, stub_engine_dir, {
        C.CAMERA_RIGHT_UP: CANONICAL,
        C.CAMERA_LEFT_UP: CANONICAL,
        C.CAMERA_RIGHT_UP_TOP: CANONICAL,
        C.CAMERA_LEFT_UP_TOP: _mirrored(CANONICAL),
    })
    assert result.global_gap_count == len(CANONICAL)
    assert result.global_wagon_count == len(CANONICAL) - 1

    state = load_global_train_state(result.state_json_path)
    assert [w.global_id for w in state.wagons] == [
        "GW_%d" % n for n in range(1, len(CANONICAL))]
    assert verify_roster_integrity(state) == []
    assert state.master_camera == C.CAMERA_RIGHT_UP


def test_ownership_rule_holds_on_the_assembled_state(tmp_path,
                                                     stub_engine_dir):
    """ef2868f: before -> previous, at/after -> next, never two owners."""
    workspace = str(tmp_path / "ws")
    result = _assemble(workspace, stub_engine_dir, {
        C.CAMERA_RIGHT_UP: CANONICAL,
        C.CAMERA_LEFT_UP_TOP: _mirrored(CANONICAL),
    })
    state = load_global_train_state(result.state_json_path)
    ownership = wagon_ownership.for_state(state)
    assert ownership is not None
    assert wagon_ownership.BOUNDARY_GOES_TO == "next_wagon"

    for camera_id in (C.CAMERA_RIGHT_UP, C.CAMERA_LEFT_UP_TOP):
        bounds = ownership.camera_boundaries(camera_id)
        assert bounds, camera_id
        step = -1 if camera_id == C.CAMERA_LEFT_UP_TOP else 1
        for index in range(1, len(bounds) - 1):
            boundary = bounds[index]
            assert ownership.owner_of_camera_frame(camera_id, boundary) == \
                ownership.wagon_ids[index]
            assert ownership.owner_of_camera_frame(
                camera_id, boundary - step) == ownership.wagon_ids[index - 1]

        low, high = min(bounds), max(bounds)
        doubles = [frame for frame in range(low, high + 1)
                   if sum(1 for gw in ownership.wagon_ids
                          if ownership.owner_of_camera_frame(camera_id, frame)
                          == gw) > 1]
        assert doubles == [], "%s: double-owned frames %s" % (camera_id,
                                                              doubles[:5])


def test_observation_assignment_is_single_owner(tmp_path, stub_engine_dir):
    """A persisted observation may reach exactly one wagon."""
    workspace = str(tmp_path / "ws")
    for camera_id, positions in ((C.CAMERA_RIGHT_UP, CANONICAL),
                                 (C.CAMERA_LEFT_UP, CANONICAL)):
        _write_camera(workspace, camera_id, positions)

    # Add door observations across the whole region, including exactly on gaps.
    document = json.loads(open(ev.evidence_path(workspace, C.CAMERA_LEFT_UP),
                               encoding="utf-8").read())
    frames = [gap["confirmation_frame"] for gap in document["gaps"]]
    frames += [f + 1 for f in frames] + [f - 1 for f in frames]
    document["observations"] = [
        {"feature": "door", "frame_idx": int(frame), "timestamp": 0.0,
         "state": "", "confidence": 0.9, "bbox": list(LEFT_DOOR_BOX),
         "raw_class": "open_door", "score": 1.0, "extra": {}}
        for frame in sorted(set(frames)) if frame >= 0]
    with open(ev.evidence_path(workspace, C.CAMERA_LEFT_UP), "w",
              encoding="utf-8") as handle:
        json.dump(document, handle)

    result = global_assembly.assemble(
        workspace=workspace, repo_root=_REPO_ROOT, batch_key="parity",
        engine_dir=stub_engine_dir, verbose=False)
    state = load_global_train_state(result.state_json_path)
    evidences = {camera_id: ev.load_evidence(workspace, camera_id)
                 for camera_id in (C.CAMERA_RIGHT_UP, C.CAMERA_LEFT_UP)}
    assigned = global_assembly.assign_observations(state, evidences)

    owners = {}
    for gw_id, features in assigned.items():
        for per_camera in features.values():
            for camera_id, observations in per_camera.items():
                for observation in observations:
                    owners.setdefault((camera_id, observation.frame_idx),
                                      set()).add(gw_id)
    doubles = {key: value for key, value in owners.items() if len(value) > 1}
    assert doubles == {}, "observation owned twice: %s" % list(doubles)[:3]
    assert owners, "no observation was assigned at all"


def test_final_wagon_assignment_produces_feature_payloads(tmp_path,
                                                          stub_engine_dir):
    """End of the chain: assigned evidence becomes per-wagon feature JSON."""
    workspace = str(tmp_path / "ws")
    _write_camera(workspace, C.CAMERA_RIGHT_UP, CANONICAL)
    _write_camera(workspace, C.CAMERA_RIGHT_UP_TOP, CANONICAL)

    document = json.loads(open(
        ev.evidence_path(workspace, C.CAMERA_RIGHT_UP_TOP),
        encoding="utf-8").read())
    centre = REGION_START + REGION_FRAMES // 2
    document["observations"] = [
        {"feature": "load", "frame_idx": centre + offset, "timestamp": 0.0,
         "state": "", "confidence": 0.9, "bbox": None,
         "raw_class": "loaded", "score": 0.0, "extra": {}}
        for offset in range(0, 20, 2)]
    with open(ev.evidence_path(workspace, C.CAMERA_RIGHT_UP_TOP), "w",
              encoding="utf-8") as handle:
        json.dump(document, handle)

    result = global_assembly.assemble(
        workspace=workspace, repo_root=_REPO_ROOT, batch_key="parity",
        engine_dir=stub_engine_dir, verbose=False)
    assert result.ready

    load_dir = os.path.join(workspace, "wagon_states", "load")
    payloads = [json.load(open(os.path.join(load_dir, name), encoding="utf-8"))
                for name in sorted(os.listdir(load_dir))]
    assert payloads, "no load payload was written"
    assert all(p["load_status"] == C.LOAD_LOADED for p in payloads)
    assert all(p["source"] == "global_assembly_from_persisted_evidence"
               for p in payloads)
