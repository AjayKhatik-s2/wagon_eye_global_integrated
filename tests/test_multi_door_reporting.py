"""A wagon can show several DISTINCT doors; the report must show them all.

Door 1 CLOSED and Door 2 OPEN on the same wagon are two observations, not one.
The old contract could only carry `left_door` / `right_door` -- one state and
one snapshot per camera SIDE -- so a second door was discarded and the reader
could not see that one door was shut and another was not.

Door identity was NOT invented for this: the tracker already groups a door's
frames into one track and `DoorIdentityMerger` collapses fragmented tracks of
the same physical door, while the sampled path (the production default) groups
observations into `EvidenceAggregator` candidates. Both already yield one record
per physical door. `doors[]` stops discarding that identity and gives each door
its own snapshot.

Also covers the Ultralytics `half` -> `quantize` deprecation cleanup.

    python -m pytest tests/test_multi_door_reporting.py -q
"""

from __future__ import annotations

import json
import os
import re
import sys

import pytest

_TEST_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_TEST_DIR)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from core import constants as C
from core.global_state_loader import GlobalTrainState, GlobalWagon
from core.unified_wagon_state import UnifiedWagonState, summarize_wagons
from features.door import processor as door_proc
from features.evidence_aggregator import Observation


# -----------------------------------------------------------------------------
# helpers: aggregator groups exactly as EvidenceAggregator.finalize() emits them
# -----------------------------------------------------------------------------

def _group(candidate_id, state, *, frames, bbox, confidence=0.9, best_frame=None):
    """One accepted candidate == one distinct physical door."""
    best_frame = frames[len(frames) // 2] if best_frame is None else best_frame
    return {
        "candidate_id": candidate_id,
        "state": state,
        "confidence": confidence,
        "frame_support": len(frames),
        "first_frame": frames[0],
        "last_frame": frames[-1],
        "accepted": True,
        "best": Observation(frame_idx=best_frame, state=state,
                            confidence=confidence, bbox=bbox, score=0.8),
    }


def _snapshots(*frame_indices):
    """Stand-in frame images; only identity matters here."""
    return {int(f): object() for f in frame_indices}


LEFT_BBOX = (100.0, 50.0, 200.0, 300.0)
RIGHT_BBOX = (400.0, 50.0, 500.0, 300.0)


# -----------------------------------------------------------------------------
# 1-4. one closed / one open / closed+open / two open
# -----------------------------------------------------------------------------

def test_1_single_closed_door():
    groups = [_group(1, C.DOOR_CLOSED, frames=[3, 6, 9], bbox=LEFT_BBOX)]
    doors = door_proc._door_evidence_from_groups(
        C.CAMERA_LEFT_UP, groups, _snapshots(6), 640, 480)
    assert len(doors) == 1
    assert doors[0]["state"] == C.DOOR_CLOSED
    assert doors[0]["_snapshot"] is not None
    assert door_proc.wagon_door_status(doors) == C.DOOR_CLOSED


def test_2_single_open_door():
    groups = [_group(1, C.DOOR_OPEN, frames=[3, 6, 9], bbox=LEFT_BBOX)]
    doors = door_proc._door_evidence_from_groups(
        C.CAMERA_LEFT_UP, groups, _snapshots(6), 640, 480)
    assert len(doors) == 1
    assert doors[0]["state"] == C.DOOR_OPEN
    assert door_proc.wagon_door_status(doors) == C.DOOR_OPEN


def test_3_one_closed_and_one_open_are_both_kept():
    """The headline case: two doors, two states, two snapshots."""
    groups = [
        _group(1, C.DOOR_CLOSED, frames=[3, 6], bbox=LEFT_BBOX, best_frame=3),
        _group(2, C.DOOR_OPEN, frames=[9, 12], bbox=RIGHT_BBOX, best_frame=9),
    ]
    doors = door_proc._door_evidence_from_groups(
        C.CAMERA_LEFT_UP, groups, _snapshots(3, 9), 640, 480)

    assert len(doors) == 2, "a second door must not be collapsed away"
    assert {d["state"] for d in doors} == {C.DOOR_CLOSED, C.DOOR_OPEN}
    # each door has its OWN snapshot, not a shared one
    snaps = [id(d["_snapshot"]) for d in doors]
    assert len(set(snaps)) == 2, "the two doors share one snapshot"
    assert len({d["track_id"] for d in doors}) == 2


def test_4_two_open_doors_stay_two_entries():
    """Same state on both doors must NOT merge them (a state-keyed bucket would)."""
    groups = [
        _group(1, C.DOOR_OPEN, frames=[3], bbox=LEFT_BBOX, best_frame=3),
        _group(2, C.DOOR_OPEN, frames=[9], bbox=RIGHT_BBOX, best_frame=9),
    ]
    doors = door_proc._door_evidence_from_groups(
        C.CAMERA_LEFT_UP, groups, _snapshots(3, 9), 640, 480)
    assert len(doors) == 2
    assert all(d["state"] == C.DOOR_OPEN for d in doors)
    assert len({id(d["_snapshot"]) for d in doors}) == 2


# -----------------------------------------------------------------------------
# 5. many frames of ONE door -> one entry, one snapshot
# -----------------------------------------------------------------------------

def test_5_many_frames_of_the_same_door_yield_one_entry():
    """De-duplication is the aggregator's job and is already done upstream."""
    frames = list(range(3, 61, 3))            # 20 sampled frames, stride 3
    groups = [_group(1, C.DOOR_OPEN, frames=frames, bbox=LEFT_BBOX)]
    doors = door_proc._door_evidence_from_groups(
        C.CAMERA_LEFT_UP, groups, _snapshots(*frames), 640, 480)

    assert len(doors) == 1, "one physical door must not become 20 entries"
    assert doors[0]["total_hits"] == len(frames)
    assert doors[0]["best_frame_idx"] in frames


def test_5_dedup_is_not_reimplemented_in_the_processor():
    """The processor must lean on the existing identity, not invent its own."""
    import inspect
    source = inspect.getsource(door_proc._door_evidence_from_groups)
    assert "candidate_id" in source
    for banned in ("iou(", "merge", "dedup"):
        assert banned not in source, (
            "a second de-duplication mechanism appeared: %r" % banned)


# -----------------------------------------------------------------------------
# 6 + 7. distinct doors are separate entries; wagon status is OPEN if any is
# -----------------------------------------------------------------------------

def test_6_ordering_is_deterministic_left_then_position():
    doors = [
        {"camera_id": C.CAMERA_RIGHT_UP, "track_id": 9, "bbox": [400, 0, 500, 10]},
        {"camera_id": C.CAMERA_LEFT_UP, "track_id": 4, "bbox": [300, 0, 400, 10]},
        {"camera_id": C.CAMERA_LEFT_UP, "track_id": 2, "bbox": [100, 0, 200, 10]},
    ]
    ordered = door_proc.order_doors(doors)
    assert [d["track_id"] for d in ordered] == [2, 4, 9]
    assert door_proc.order_doors(list(reversed(doors))) == ordered


def test_7_wagon_status_is_open_when_any_door_is_open():
    assert door_proc.wagon_door_status(
        [{"state": C.DOOR_CLOSED}, {"state": C.DOOR_OPEN}]) == C.DOOR_OPEN
    assert door_proc.wagon_door_status(
        [{"state": C.DOOR_CLOSED}, {"state": C.DOOR_CLOSED}]) == C.DOOR_CLOSED
    assert door_proc.wagon_door_status([]) == C.NO_DATA
    # DAMAGED still outranks OPEN, as the per-side picker has always done
    assert door_proc.wagon_door_status(
        [{"state": C.DOOR_OPEN}, {"state": C.DOOR_DAMAGED}]) == C.DOOR_DAMAGED


def test_7_unified_state_exposes_the_same_rule():
    wagon = UnifiedWagonState(
        global_id="GW_1", wagon_index=1,
        doors=[{"state": C.DOOR_CLOSED, "side": "left", "door_index": 1},
               {"state": C.DOOR_OPEN, "side": "left", "door_index": 2}])
    assert wagon.door_status == C.DOOR_OPEN
    assert wagon.has_open_door is True


def test_7_unified_state_falls_back_to_the_per_side_fields():
    """A door payload without `doors` must behave exactly as before."""
    wagon = UnifiedWagonState(
        global_id="GW_1", wagon_index=1,
        left_door=C.DOOR_CLOSED, right_door=C.DOOR_OPEN)
    assert wagon.doors == []
    assert wagon.door_status == C.DOOR_OPEN
    assert wagon.has_open_door is True

    shut = UnifiedWagonState(global_id="GW_2", wagon_index=2,
                             left_door=C.DOOR_CLOSED, right_door=C.DOOR_CLOSED)
    assert shut.door_status == C.DOOR_CLOSED
    assert shut.has_open_door is False


# -----------------------------------------------------------------------------
# fusion carries the doors through
# -----------------------------------------------------------------------------

def _state(gw_ids):
    return GlobalTrainState(
        total_wagons=len(gw_ids),
        wagons=tuple(
            GlobalWagon(global_id=gw_id, wagon_index=index + 1,
                        start_frame_master=index * 100,
                        end_frame_master=index * 100 + 99,
                        start_time=float(index), end_time=float(index + 1),
                        classification=C.CLASS_WAGON)
            for index, gw_id in enumerate(gw_ids)),
        master_camera=C.CAMERA_RIGHT_UP_TOP, master_fps=15.0)


def _write_door_json(root, gw_id, doors, left, right):
    directory = os.path.join(root, "door")
    os.makedirs(directory, exist_ok=True)
    with open(os.path.join(directory, "%s.json" % gw_id), "w",
              encoding="utf-8") as handle:
        json.dump({
            "global_id": gw_id, "feature": "door", "status": C.STATUS_OK,
            "left_door": left, "left_door_confidence": 0.9,
            "right_door": right, "right_door_confidence": 0.9,
            "doors": doors,
            "door_status": door_proc.wagon_door_status(doors),
        }, handle)


TWO_DOORS = [
    {"door_index": 1, "side": "left", "camera_id": C.CAMERA_LEFT_UP,
     "track_id": 1, "state": C.DOOR_CLOSED, "confidence": 0.88},
    {"door_index": 2, "side": "left", "camera_id": C.CAMERA_LEFT_UP,
     "track_id": 2, "state": C.DOOR_OPEN, "confidence": 0.93},
]


def test_fusion_carries_every_door(tmp_path):
    from fusion import wagon_state_builder

    root = str(tmp_path / "wagon_states")
    state = _state(["GW_1"])
    _write_door_json(root, "GW_1", TWO_DOORS, C.DOOR_OPEN, C.NO_DATA)

    unified = wagon_state_builder.build(
        state=state, wagon_states_root=root, write_per_wagon_json=False,
        verbose=False)

    wagon = unified["GW_1"]
    assert len(wagon.doors) == 2
    assert [d["state"] for d in wagon.doors] == [C.DOOR_CLOSED, C.DOOR_OPEN]
    assert wagon.door_status == C.DOOR_OPEN
    # the per-side contract is untouched
    assert wagon.left_door == C.DOOR_OPEN


def test_fusion_without_doors_field_is_unchanged(tmp_path):
    from fusion import wagon_state_builder

    root = str(tmp_path / "wagon_states")
    directory = os.path.join(root, "door")
    os.makedirs(directory)
    with open(os.path.join(directory, "GW_1.json"), "w", encoding="utf-8") as h:
        json.dump({"global_id": "GW_1", "feature": "door",
                   "status": C.STATUS_OK, "left_door": C.DOOR_OPEN,
                   "left_door_confidence": 0.7}, h)

    unified = wagon_state_builder.build(
        state=_state(["GW_1"]), wagon_states_root=root,
        write_per_wagon_json=False, verbose=False)
    assert unified["GW_1"].doors == []
    assert unified["GW_1"].left_door == C.DOOR_OPEN


# -----------------------------------------------------------------------------
# 8 + 9. report entries: both doors, each with its own snapshot
# -----------------------------------------------------------------------------

def _evidence_tree(root, gw_id, names):
    directory = os.path.join(root, gw_id, "door")
    os.makedirs(directory, exist_ok=True)
    for name in names:
        path = os.path.join(directory, "%s.jpg" % name)
        # a 1x1 JPEG so reportlab can actually load it
        with open(path, "wb") as handle:
            handle.write(bytes.fromhex(
                "ffd8ffe000104a46494600010100000100010000ffdb004300"
                "08060607060508070707090908080a0c140d0c0b0b0c191213"
                "0f141d1a1f1e1d1a1c1c20242e2720222c231c1c2837292c30"
                "31343434371f27393d38323c2e333432ffc0000b0801000100"
                "01011100ffc40014000100000000000000000000000000000009"
                "ffc40014100100000000000000000000000000000000ffda0008"
                "010100003f00d2cf20ffd9"))
    return directory


def test_8_report_emits_one_entry_per_door_with_its_own_snapshot(tmp_path):
    from reporting import _adapter

    evidence_root = str(tmp_path / "evidence")
    _evidence_tree(evidence_root, "GW_1", ["door_1", "door_2", "left_best"])

    wagon = UnifiedWagonState(global_id="GW_1", wagon_index=1, doors=TWO_DOORS,
                              left_door=C.DOOR_OPEN)
    entries = _adapter._door_entries(
        evidence_root=evidence_root, gw_id="GW_1", wagon_number=25, u=wagon)

    left = entries["left"]
    assert len(left) == 2, "both doors must appear under the wagon"
    assert [d["door_number"] for d in left] == [1, 2]
    assert [d["state"] for d in left] == [C.DOOR_CLOSED, C.DOOR_OPEN]
    paths = [d["local_snapshot_path"] for d in left]
    assert all(paths), "every door needs a snapshot"
    assert len(set(paths)) == 2, "doors must not share one snapshot"
    assert paths[0].endswith("door_1.jpg") and paths[1].endswith("door_2.jpg")


def test_8_per_door_snapshot_falls_back_to_the_side_frame(tmp_path):
    """The tracker path takes no per-door image; show the side frame, not none."""
    from reporting import _adapter

    evidence_root = str(tmp_path / "evidence")
    _evidence_tree(evidence_root, "GW_1", ["left_best"])
    wagon = UnifiedWagonState(global_id="GW_1", wagon_index=1, doors=TWO_DOORS,
                              left_door=C.DOOR_OPEN)
    entries = _adapter._door_entries(
        evidence_root=evidence_root, gw_id="GW_1", wagon_number=1, u=wagon)
    assert len(entries["left"]) == 2
    assert all(d["local_snapshot_path"].endswith("left_best.jpg")
               for d in entries["left"])


def test_8_all_closed_wagon_reports_no_door_anomaly(tmp_path):
    """Unchanged behaviour: a wagon with nothing open contributes no entries."""
    from reporting import _adapter

    evidence_root = str(tmp_path / "evidence")
    _evidence_tree(evidence_root, "GW_1", ["door_1"])
    wagon = UnifiedWagonState(
        global_id="GW_1", wagon_index=1,
        doors=[{"door_index": 1, "side": "left", "state": C.DOOR_CLOSED,
                "camera_id": C.CAMERA_LEFT_UP, "confidence": 0.9}])
    entries = _adapter._door_entries(
        evidence_root=evidence_root, gw_id="GW_1", wagon_number=1, u=wagon)
    assert entries == {"left": [], "right": []}


def test_9_legacy_per_side_path_is_untouched(tmp_path):
    from reporting import _adapter

    evidence_root = str(tmp_path / "evidence")
    _evidence_tree(evidence_root, "GW_1", ["left_best", "right_best"])
    wagon = UnifiedWagonState(global_id="GW_1", wagon_index=1,
                              left_door=C.DOOR_OPEN, right_door=C.DOOR_CLOSED)
    entries = _adapter._door_entries(
        evidence_root=evidence_root, gw_id="GW_1", wagon_number=1, u=wagon)
    assert len(entries["left"]) == 1
    assert entries["left"][0]["door_number"] == 1
    assert entries["right"] == []          # CLOSED side is not an anomaly


# -----------------------------------------------------------------------------
# 10. the report still builds, and names each door
# -----------------------------------------------------------------------------

def test_10_report_json_and_pdf_build_with_two_doors(tmp_path):
    from reporting import combined_train_report

    state = _state(["GW_1", "GW_2"])
    evidence_root = str(tmp_path / "evidence")
    _evidence_tree(evidence_root, "GW_1", ["door_1", "door_2", "left_best"])

    unified = {
        "GW_1": UnifiedWagonState(global_id="GW_1", wagon_index=1,
                                  doors=TWO_DOORS, left_door=C.DOOR_OPEN,
                                  classification=C.CLASS_WAGON),
        "GW_2": UnifiedWagonState(global_id="GW_2", wagon_index=2,
                                  classification=C.CLASS_WAGON,
                                  left_door=C.DOOR_CLOSED,
                                  right_door=C.DOOR_CLOSED),
    }

    result = combined_train_report.build(
        state=state, unified=unified, output_dir=str(tmp_path / "reports"),
        batch_key="multidoor", evidence_root=evidence_root, verbose=False)

    assert result.get("json_path") and os.path.isfile(result["json_path"])
    with open(result["json_path"], "r", encoding="utf-8") as handle:
        document = json.load(handle)
    text = json.dumps(document)
    assert "DOOR 1" in text and "DOOR 2" in text, (
        "the summary text must name each door")

    try:
        import reportlab                                      # noqa: F401
    except ImportError:
        pytest.skip("reportlab not installed")
    assert result.get("pdf_path") and os.path.isfile(result["pdf_path"])
    assert os.path.getsize(result["pdf_path"]) > 1000


def test_10_summary_counts_are_unaffected():
    wagons = [
        UnifiedWagonState(global_id="GW_1", wagon_index=1, doors=TWO_DOORS,
                          left_door=C.DOOR_OPEN),
        UnifiedWagonState(global_id="GW_2", wagon_index=2,
                          left_door=C.DOOR_CLOSED, right_door=C.DOOR_CLOSED),
    ]
    summary = summarize_wagons(wagons)
    assert summary["total_wagons"] == 2
    assert summary["left_doors_open"] == 1


# -----------------------------------------------------------------------------
# invariants that must survive this change
# -----------------------------------------------------------------------------

def test_door_sampling_stride_is_still_three():
    from orchestrator.master_runner import _build_parser

    args = _build_parser().parse_args(["--once"])
    assert (args.door_inference_mode, args.door_sample_stride) == ("sampled", 3)
    assert (args.damage_inference_mode, args.damage_sample_stride) == ("sampled", 3)
    assert (args.load_inference_mode, args.load_sample_stride) == ("sampled", 2)


def test_wagon_ids_and_ownership_fix_are_intact(tmp_path):
    """The ef2868f ownership rule must still hold."""
    from core import wagon_ownership

    state = _state(["GW_1", "GW_2", "GW_3"])
    assert [w.global_id for w in state.wagons] == ["GW_1", "GW_2", "GW_3"]
    own = wagon_ownership.WagonOwnership(state)
    assert own.wagon_ids == ("GW_1", "GW_2", "GW_3")
    assert wagon_ownership.BOUNDARY_GOES_TO == "next_wagon"


def test_a_door_observation_belongs_to_exactly_one_wagon():
    """Door frames go through the same ownership filter as every feature."""
    import inspect
    source = inspect.getsource(door_proc)
    assert "wagon_ownership.for_state" in source
    assert "ownership=ownership" in source
    # and the shared enumeration point is what applies it
    from features._common import list_wagon_frames
    assert "ownership" in inspect.signature(list_wagon_frames).parameters


# -----------------------------------------------------------------------------
# Ultralytics `half` -> `quantize` deprecation
# -----------------------------------------------------------------------------

_PREDICT_HALF = re.compile(r"\bhalf\s*=\s*(True|False)\b")


def _source(path):
    with open(os.path.join(_REPO_ROOT, path), "r", encoding="utf-8") as handle:
        return handle.read()


def _code_lines(text):
    """Non-comment, non-docstring-ish lines -- enough to spot a real call."""
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#") or not stripped:
            continue
        yield line


def test_door_does_not_pass_the_deprecated_half_argument():
    """Ultralytics warns for every call that MENTIONS `half`, even half=False."""
    offenders = [line.strip() for line in _code_lines(
        _source("features/door/processor.py")) if _PREDICT_HALF.search(line)]
    assert offenders == [], offenders


def test_shared_predict_helper_omits_half_by_default():
    from features._common import _predict_kwargs

    assert _predict_kwargs(False) == {"verbose": False}, (
        "half must not be mentioned when fp32 is wanted -- that is the default")
    # a genuine fp16 request is still honoured: dropping it would change
    # inference precision, which this cleanup must not do.
    assert _predict_kwargs(True) == {"verbose": False, "half": True}


def test_load_and_classification_never_requested_half():
    """Load goes through run_classification, which never passed `half`."""
    offenders = [line.strip() for line in _code_lines(
        _source("features/load/processor.py")) if _PREDICT_HALF.search(line)]
    assert offenders == []


def test_only_ocr_still_requests_fp16_and_it_is_documented():
    """OCR asks for half=True deliberately; it is disabled in production.

    Silently dropping it would change OCR's inference precision, which is out
    of scope for a warning cleanup -- so it stays, and stays visible here.
    """
    offenders = [line.strip() for line in _code_lines(
        _source("features/ocr/processor.py")) if _PREDICT_HALF.search(line)]
    assert offenders == ["results = yolo_model(frame, verbose=False, half=True)[0]"], (
        "the OCR fp16 request changed; update this documented exception")


def test_the_warning_text_belongs_to_ultralytics_not_to_us():
    """Provenance: the message is emitted inside Ultralytics, not our code."""
    import inspect

    from ultralytics.utils import deprecation_warn

    source = inspect.getsource(deprecation_warn)
    assert "is deprecated and will be removed in the future" in source
    assert "Use '" in source
    # nothing in this repository emits that text itself
    for root, _dirs, files in os.walk(_REPO_ROOT):
        if any(part in root for part in
               ("__pycache__", ".git", "batch_outputs", ".venv")):
            continue
        for name in files:
            if not name.endswith(".py") or name == os.path.basename(__file__):
                continue
            with open(os.path.join(root, name), "r", encoding="utf-8",
                      errors="replace") as handle:
                assert "is deprecated and will be removed" not in handle.read(), (
                    "%s emits the deprecation text itself" % name)
