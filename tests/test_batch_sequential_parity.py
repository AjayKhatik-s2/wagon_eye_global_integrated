"""Anti-divergence: every decision is made by BATCH's code, not a copy of it.

WHY EVERY TEST IN THIS FILE MOVED

Before the exact-parity redesign, Sequential had its own confidence gates
(`camera_runner._observe`) and its own per-wagon aggregators
(`global_assembly._aggregate_door/_load/_damage`). The tests here compared those
copies against Batch's originals -- which is the best you can do when a copy
exists, and is exactly why the copies were the problem.

Sequential now has no gate and no aggregator. Batch's own processors run, over
Batch's own materialized wagon cache. So each test below keeps its correctness
assertion but points it at the real owner:

  * the gate / filter / verdict semantics are asserted against BATCH's own
    functions, so a change in Batch's rule still breaks a test here;
  * and each is paired with an assertion that Sequential contains no copy,
    which is what previously drifted.

Nothing was weakened to pass. The load verdict, the damage edge filter and the
door aggregation are still driven with the same inputs and still checked for the
same subtle behaviours (an inclusive gate boundary, a denominator that counts
every sampled frame, two physical doors that keep separate states).

Alignment, reversal, the roster and ownership are exercised against the REAL
engine, because it can run its whole global half from persisted records.

    python -m pytest tests/test_batch_sequential_parity.py -q
"""

from __future__ import annotations

import inspect
import os
import sys

import numpy as np
import pytest

_TEST_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_TEST_DIR)
for _path in (_REPO_ROOT, _TEST_DIR):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import _parity_fixtures as F

from core import constants as C
from sequential import camera_runner, evidence as ev, global_assembly

WIDTH, HEIGHT = 640, 480

REAL_ENGINE = F.real_engine_dir()
needs_engine = pytest.mark.skipif(
    REAL_ENGINE is None,
    reason="the frozen global_wagon_app checkout is not on this machine")


def _sequential_source():
    return (inspect.getsource(camera_runner)
            + inspect.getsource(global_assembly))


# =============================================================================
# 1. The door confidence gate is BATCH's
# =============================================================================

def test_door_gate_value_comes_from_batch_config():
    """The floor must be Batch's TrackerConfig value, never a literal."""
    from features.door.processor import TrackerConfig

    floor = float(TrackerConfig().closed_confidence_threshold)
    assert 0.0 < floor < 1.0

    source = inspect.getsource(sys.modules["features.door.processor"])
    assert "min_conf = float(tracker_config.closed_confidence_threshold)" in \
        source, "Batch's door gate no longer reads its own config value"


def test_door_gate_is_inclusive_at_the_floor_in_batch():
    """`>=`, not `>`: a detection exactly at the floor survives.

    MIGRATED: this used to call `camera_runner._observe` and compare survivors.
    That function is gone -- Sequential runs no gate. The operator itself is the
    thing worth pinning, so it is asserted where it lives.
    """
    source = inspect.getsource(sys.modules["features.door.processor"])
    assert "keep = confs >= min_conf" in source, (
        "Batch's door gate changed shape; the inclusive boundary this parity "
        "claim depends on must be re-verified")
    assert "keep = confs > min_conf" not in source


def test_sequential_holds_no_door_gate_of_its_own():
    """The copy that used to drift must not come back."""
    source = _sequential_source()
    for banned in ("closed_confidence_threshold", "door_confidence_floor",
                   "confs >= min_conf", "_observe("):
        assert banned not in source, (
            "Sequential is gating door detections itself again: %s" % banned)


# =============================================================================
# 2. The damage filter is BATCH's -- driven directly, same arrays
# =============================================================================

def test_damage_gate_value_comes_from_batch_constant():
    from features.damage import processor as damage_processor

    assert damage_processor._EDGE_BYPASS_CONF == 0.70
    assert isinstance(C.CONF_DAMAGE, float)


def test_batch_damage_filter_keeps_exactly_what_it_should():
    """Batch's REAL `_filter_detections_for_top`, called with fixed arrays.

    MIGRATED: previously this compared Sequential's filtered output against
    Batch's. Sequential no longer filters, so Batch's pure function is driven
    directly -- which tests the same rule without a copy to compare to.
    """
    from features.damage.processor import _filter_detections_for_top

    names = {0: "dent", 1: "hole"}
    # centred and comfortably above the floor -> kept
    centred = [100.0, 80.0, 200.0, 160.0]
    # far below the confidence floor -> dropped
    faint = [110.0, 90.0, 210.0, 170.0]

    boxes = np.asarray([centred, faint], dtype=float)
    confs = np.asarray([0.90, 0.01], dtype=float)
    cls_ids = np.asarray([0, 0], dtype=int)

    kept_boxes, kept_confs, kept_cls = _filter_detections_for_top(
        boxes, confs, cls_ids, names, WIDTH, HEIGHT, float(C.CONF_DAMAGE))

    assert len(kept_boxes) == len(kept_confs) == len(kept_cls) == 1
    assert kept_confs[0] == pytest.approx(0.90)
    assert list(kept_boxes[0]) == pytest.approx(centred)


def test_batch_damage_filter_drops_everything_below_the_floor():
    from features.damage.processor import _filter_detections_for_top

    names = {0: "dent"}
    boxes = np.asarray([[100.0, 80.0, 200.0, 160.0]], dtype=float)
    confs = np.asarray([float(C.CONF_DAMAGE) / 2.0], dtype=float)
    cls_ids = np.asarray([0], dtype=int)

    kept_boxes, _confs, _cls = _filter_detections_for_top(
        boxes, confs, cls_ids, names, WIDTH, HEIGHT, float(C.CONF_DAMAGE))
    assert len(kept_boxes) == 0


def test_sequential_calls_batchs_damage_filter_not_a_copy():
    source = _sequential_source()
    for banned in ("_filter_detections_for_top", "_EDGE_BYPASS_CONF",
                   "keep_mask"):
        assert banned not in source, (
            "Sequential re-implements Batch's damage filter: %s" % banned)
    # damage runs through Batch's registry
    assert "load_feature_runner" in inspect.getsource(
        global_assembly._run_features)


# =============================================================================
# 3. The load verdict is BATCH's -- driven directly, same class sequences
# =============================================================================

def _drive_batch_load(monkeypatch, classes, confidence=0.9):
    """Run BATCH's own `_aggregate_camera` over a fixed class sequence.

    Only the frame source and the classifier are replaced; the counting, the
    denominator and the threshold comparison are Batch's.
    """
    from features.load import processor as load_processor

    frames = [(index * 2, np.zeros((4, 4, 3), dtype=np.uint8))
              for index in range(len(classes))]
    sequence = list(classes)

    monkeypatch.setattr(load_processor, "iter_wagon_frames",
                        lambda *a, **k: iter(frames))
    calls = {"n": 0}

    def _classify(model, frame):
        name = sequence[calls["n"]]
        calls["n"] += 1
        return (name, confidence)

    monkeypatch.setattr(load_processor, "run_classification", _classify)

    status, conf, used, n_loaded, n_empty, _bl, _be = \
        load_processor._aggregate_camera(
            object(), "cache", "GW_1", C.CAMERA_RIGHT_UP_TOP,
            every_nth=2, max_frames=None)
    return {"status": status, "confidence": conf, "used": used,
            "loaded": n_loaded, "empty": n_empty}


@pytest.mark.parametrize("classes,expected", [
    (["loaded"] * 5, C.LOAD_LOADED),
    (["empty"] * 5, C.LOAD_EMPTY),
    # ratio 0.40 > 0.35 -> LOADED
    (["loaded", "loaded", "empty", "empty", "empty"], C.LOAD_LOADED),
    # ratio 0.17 -> EMPTY
    (["loaded", "empty", "empty", "empty", "empty", "empty"], C.LOAD_EMPTY),
    # a third class dilutes the denominator: 3/9 = 0.33, NOT 3/6 = 0.50
    (["loaded"] * 3 + ["empty"] * 3 + ["unknown"] * 3, C.LOAD_EMPTY),
    (["unknown"] * 4, C.NO_DATA),
])
def test_batch_load_verdict_for_each_case(monkeypatch, classes, expected):
    """The same six cases the old parity test used, against Batch's own rule."""
    assert _drive_batch_load(monkeypatch, classes)["status"] == expected


def test_load_denominator_counts_every_sampled_frame(monkeypatch):
    """The subtle one: Batch's `total` is every sampled frame, not loaded+empty."""
    classes = ["loaded"] * 3 + ["empty"] * 3 + ["unknown"] * 3
    outcome = _drive_batch_load(monkeypatch, classes)
    assert outcome["used"] == 9
    assert outcome["loaded"] == 3
    assert outcome["empty"] == 3
    # 3/9 = 0.33 is NOT above 0.35, so this wagon is EMPTY, not LOADED
    assert outcome["status"] == C.LOAD_EMPTY


def test_load_threshold_is_strictly_greater_than(monkeypatch):
    """`>`, not `>=`: a ratio exactly at 0.35 is not LOADED."""
    from features.load.processor import _LOADED_RATIO_THRESHOLD

    assert _LOADED_RATIO_THRESHOLD == 0.35
    # 7/20 = 0.35 exactly
    classes = ["loaded"] * 7 + ["empty"] * 13
    outcome = _drive_batch_load(monkeypatch, classes)
    assert outcome["used"] == 20
    assert outcome["status"] == C.LOAD_EMPTY, (
        "a ratio exactly at the threshold must not count as LOADED")


def test_sequential_holds_no_load_verdict_of_its_own():
    source = _sequential_source()
    for banned in ("_LOADED_RATIO_THRESHOLD", "0.35", "loaded_ratio",
                   "_aggregate_load"):
        assert banned not in source, (
            "Sequential is deciding load status itself again: %s" % banned)


def test_load_has_no_detection_confidence_gate_in_batch():
    """Load is a classifier: every sampled frame votes, none is gated out."""
    source = inspect.getsource(sys.modules["features.load.processor"])
    assert "confidence_floor" not in source
    assert "min_conf" not in source


# =============================================================================
# 4. Door aggregation and MULTI-DOOR are BATCH's
# =============================================================================

def test_door_raw_class_mapping_is_batchs_table():
    """Sequential must not carry its own label table."""
    assert C.DOOR_LABEL_TO_STATE["open_door"] == C.DOOR_OPEN
    assert C.DOOR_LABEL_TO_STATE["closed_door"] == C.DOOR_CLOSED
    assert C.DOOR_LABEL_TO_STATE["closed_with_wire"] == C.DOOR_PARTIAL

    source = _sequential_source()
    for banned in ('"open_door":', '"closed_door":', "DOOR_LABEL_TO_STATE = "):
        assert banned not in source, (
            "Sequential defines its own door label table: %s" % banned)


def test_two_distinct_doors_keep_their_own_states_in_batchs_helpers():
    """The b6f67b5 contract, at its owner: two doors, OPEN and CLOSED, both kept.

    MIGRATED: previously driven through `global_assembly._aggregate_door`. That
    aggregator is gone; Batch's `order_doors` / `wagon_door_status` decide this
    now, and the end-to-end survival through a real assembly is asserted in
    tests/test_batch_sequential_exact_parity.py.
    """
    from features.door.processor import (order_doors, wagon_door_status,
                                         _indexed_doors)

    doors = [
        {"camera_id": C.CAMERA_RIGHT_UP, "state": C.DOOR_CLOSED,
         "bbox": [300.0, 80.0, 400.0, 200.0], "track_id": 2,
         "confidence": 0.80},
        {"camera_id": C.CAMERA_RIGHT_UP, "state": C.DOOR_OPEN,
         "bbox": [100.0, 80.0, 200.0, 200.0], "track_id": 1,
         "confidence": 0.90},
    ]
    ordered = _indexed_doors(order_doors(doors))

    assert len(ordered) == 2, "a physical door was collapsed away"
    assert [d["door_index"] for d in ordered] == [1, 2]
    # ordering is by position across the wagon, so the left-most box is Door 1
    assert ordered[0]["state"] == C.DOOR_OPEN
    assert ordered[1]["state"] == C.DOOR_CLOSED
    # and the wagon-level verdict is OPEN because ANY door is open
    assert wagon_door_status(ordered) == C.DOOR_OPEN


def test_many_frames_of_one_door_stay_one_door():
    """Repeated sightings of the same track must not become several doors."""
    from features.door.processor import order_doors, _indexed_doors

    doors = [{"camera_id": C.CAMERA_LEFT_UP, "state": C.DOOR_OPEN,
              "bbox": [100.0, 80.0, 200.0, 200.0], "track_id": 7,
              "confidence": 0.9}]
    ordered = _indexed_doors(order_doors(doors))
    assert len(ordered) == 1
    assert ordered[0]["door_index"] == 1


def test_door_ordering_is_deterministic():
    """"Door 1" must mean the same door between runs."""
    from features.door.processor import order_doors

    doors = [
        {"camera_id": C.CAMERA_RIGHT_UP, "bbox": [300.0, 0.0, 400.0, 10.0],
         "track_id": 5, "state": C.DOOR_CLOSED},
        {"camera_id": C.CAMERA_LEFT_UP, "bbox": [300.0, 0.0, 400.0, 10.0],
         "track_id": 1, "state": C.DOOR_OPEN},
        {"camera_id": C.CAMERA_RIGHT_UP, "bbox": [100.0, 0.0, 200.0, 10.0],
         "track_id": 9, "state": C.DOOR_OPEN},
    ]
    once = [d["track_id"] for d in order_doors(list(doors))]
    twice = [d["track_id"] for d in order_doors(list(reversed(doors)))]
    assert once == twice
    # LEFT_UP sorts first, then position across the wagon
    assert once[0] == 1


def test_door_side_state_matches_batchs_picker():
    """OPEN outranks CLOSED; DAMAGED outranks everything."""
    from features.door.processor import _pick_side_state

    closed = {"state": C.DOOR_CLOSED, "total_hits": 40, "confidence": 0.95}
    opened = {"state": C.DOOR_OPEN, "total_hits": 2, "confidence": 0.70}
    damaged = {"state": C.DOOR_DAMAGED, "total_hits": 1, "confidence": 0.60}

    state, _conf = _pick_side_state([closed, opened])
    assert state == C.DOOR_OPEN, (
        "a single OPEN track must outrank many CLOSED ones")
    state, _conf = _pick_side_state([closed, opened, damaged])
    assert state == C.DOOR_DAMAGED
    state, _conf = _pick_side_state([closed])
    assert state == C.DOOR_CLOSED
    state, conf = _pick_side_state([])
    assert (state, conf) == (C.NO_DATA, 0.0)


def test_door_aggregation_uses_the_same_helpers_as_batch():
    """Sequential must own none of these; Batch's processor owns all of them."""
    from features.door import processor as door_processor

    for helper in ("order_doors", "wagon_door_status", "_pick_side_state",
                   "_door_evidence_from_groups", "_indexed_doors"):
        assert hasattr(door_processor, helper), helper

    source = _sequential_source()
    for banned in ("_door_evidence_from_groups", "_pick_side_state",
                   "_indexed_doors", "_aggregate_door"):
        assert banned not in source, (
            "Sequential re-implements Batch's %s" % banned)


# =============================================================================
# 5. Alignment and reversal are the ENGINE's
# =============================================================================

def _mirrored(positions):
    """The same gaps as seen by a camera pointing the other way."""
    return tuple(sorted(1000.0 - float(p) for p in positions))


@needs_engine
def test_forward_support_camera_is_not_marked_reversed(tmp_path, capsys):
    workspace = str(tmp_path / "ws")
    F.seal_all(workspace)
    snapshot = F.run_global_half(
        REAL_ENGINE, global_assembly.restore_camera_results(
            {c: ev.load_evidence(workspace, c) for c in C.ALL_CAMERAS}),
        str(tmp_path / "out"))
    capsys.readouterr()

    assert snapshot["alignments"]
    for camera, mapping in snapshot["alignments"].items():
        assert mapping["reversed"] is False, camera


@needs_engine
def test_reversed_support_camera_is_detected(tmp_path, capsys):
    """A genuinely mirrored timeline must be recognised by the engine itself.

    The positions are asymmetric on purpose: a symmetric set is its own mirror,
    so it could not distinguish a working reversal test from a broken one.

    Six gaps, not five: the engine adopts a reversal only when it beats the
    forward fit by REVERSAL_MIN_EXTRA_MATCHES (2) AND on error. A normalized
    timeline always pins its first gap to 0 and its last to 1000, so those two
    always mirror onto each other for free; with only five gaps the forward fit
    can pick up a third exact match and the error-improvement gate has nothing
    to separate. That conservatism is asserted directly in
    test_engine_refuses_an_unproven_reversal below.
    """
    positions = (0.0, 95.0, 265.0, 480.0, 730.0, 1000.0)
    workspace = str(tmp_path / "ws")
    F.seal_all(workspace, positions_by_camera={
        C.CAMERA_RIGHT_UP: positions,
        C.CAMERA_RIGHT_UP_TOP: positions,
        C.CAMERA_LEFT_UP_TOP: positions,
        C.CAMERA_LEFT_UP: _mirrored(positions)})
    snapshot = F.run_global_half(
        REAL_ENGINE, global_assembly.restore_camera_results(
            {c: ev.load_evidence(workspace, c) for c in C.ALL_CAMERAS}),
        str(tmp_path / "out"))
    capsys.readouterr()

    # the canonical train is unchanged by a camera's direction
    assert snapshot["global_gap_count"] == len(positions)
    assert snapshot["global_wagon_count"] == len(positions) - 1

    left = snapshot["alignments"]["left_up"]
    assert left["matched"] == len(positions), (
        "the mirrored camera did not align: reversal was not applied")
    assert left["reversed"] is True


@needs_engine
def test_engine_refuses_an_unproven_reversal(tmp_path, capsys):
    """Reversal must clear the engine's OWN thresholds, not merely look better.

    A five-gap mirrored timeline wins on match COUNT (5 reversed vs 3 forward)
    yet is still refused, because both directions fit with zero error and
    REVERSAL_MIN_ERROR_IMPROVEMENT is therefore not satisfied. This is the
    engine's documented policy -- "reversal not proven" -- and Sequential must
    inherit it rather than second-guess it. The canonical train is correct
    either way, which is the property that actually matters downstream.
    """
    positions = (0.0, 120.0, 300.0, 560.0, 1000.0)
    workspace = str(tmp_path / "ws")
    F.seal_all(workspace, positions_by_camera={
        C.CAMERA_RIGHT_UP: positions,
        C.CAMERA_RIGHT_UP_TOP: positions,
        C.CAMERA_LEFT_UP_TOP: positions,
        C.CAMERA_LEFT_UP: _mirrored(positions)})
    snapshot = F.run_global_half(
        REAL_ENGINE, global_assembly.restore_camera_results(
            {c: ev.load_evidence(workspace, c) for c in C.ALL_CAMERAS}),
        str(tmp_path / "out"))
    capsys.readouterr()

    assert snapshot["alignments"]["left_up"]["reversed"] is False
    # and the master still defines a correct canonical train
    assert snapshot["global_gap_count"] == len(positions)
    assert snapshot["global_wagon_count"] == len(positions) - 1


@needs_engine
def test_reversal_is_not_adopted_without_proof(tmp_path, capsys):
    """A co-ordered camera must never be flipped, whatever side it is on."""
    workspace = str(tmp_path / "ws")
    F.seal_all(workspace)
    snapshot = F.run_global_half(
        REAL_ENGINE, global_assembly.restore_camera_results(
            {c: ev.load_evidence(workspace, c) for c in C.ALL_CAMERAS}),
        str(tmp_path / "out"))
    capsys.readouterr()

    for camera in ("left_up", "left_up_top"):
        assert snapshot["alignments"][camera]["reversed"] is False, camera

    engine_source = open(os.path.join(REAL_ENGINE, "global_alignment.py"),
                         encoding="utf-8").read()
    assert "REVERSAL_MIN_EXTRA_MATCHES" in engine_source, (
        "the engine no longer requires evidence before reversing a camera")


def test_assembly_uses_the_engine_estimator_not_a_second_algorithm():
    source = inspect.getsource(global_assembly)
    for banned in ("monotonic_gap_match", "robust_linear_fit",
                   "_estimate_one_direction", "def align_camera",
                   "ALLOW_TIMELINE_REVERSAL ="):
        assert banned not in source, (
            "assembly carries a second alignment algorithm: %s" % banned)
    assert "match_all_cameras()" in inspect.getsource(
        global_assembly.run_engine_global_half)


# =============================================================================
# 6. Roster and ownership
# =============================================================================

@needs_engine
def test_canonical_roster_is_the_master_gaps_minus_one(tmp_path, capsys):
    workspace = str(tmp_path / "ws")
    positions = (0.0, 90.0, 210.0, 430.0, 700.0, 1000.0)
    F.seal_all(workspace, positions_by_camera={
        camera: positions for camera in C.ALL_CAMERAS})
    snapshot = F.run_global_half(
        REAL_ENGINE, global_assembly.restore_camera_results(
            {c: ev.load_evidence(workspace, c) for c in C.ALL_CAMERAS}),
        str(tmp_path / "out"))
    capsys.readouterr()

    assert snapshot["global_gap_count"] == len(positions)
    assert snapshot["global_wagon_count"] == len(positions) - 1
    assert len(snapshot["global_wagons"]) == len(positions) - 1


def test_ownership_is_centralized_and_boundary_goes_to_the_next_wagon():
    """The ef2868f rule, in the one module every consumer must use."""
    from core import wagon_ownership

    assert wagon_ownership.BOUNDARY_GOES_TO == "next_wagon"

    source = _sequential_source()
    for banned in ("def assign_observations", "boundary_goes_to",
                   "if frame >= start and frame <= end"):
        assert banned not in source, (
            "Sequential is deciding wagon ownership itself: %s" % banned)


def test_only_wagon_ownership_defines_the_boundary_rule():
    """No second implementation of the half-open window anywhere in the repo."""
    from core import wagon_ownership

    owner_source = inspect.getsource(wagon_ownership)
    assert "next_wagon" in owner_source

    for relative in ("sequential/global_assembly.py",
                     "sequential/camera_runner.py"):
        code = open(os.path.join(_REPO_ROOT, relative),
                    encoding="utf-8").read()
        assert "BOUNDARY_GOES_TO =" not in code, relative
