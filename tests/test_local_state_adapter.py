"""A camera-local GlobalTrainState, built from one camera's sealed evidence.

This is the shape Batch's cache builder, feature processors, fusion builder and
camera-report renderer all require -- they take `state: GlobalTrainState` and
address everything by `wagon.global_id`. Supplying it camera-locally is what
lets Phase 1 run Batch's own stages without a canonical roster.

The invariant that matters most: no `GW_n` is ever fabricated.
"""

from __future__ import annotations

import pytest

from core import constants as C
from materializer.wagon_cache_builder import _wagon_local_range
from sequential import evidence as ev
from sequential import local_state_adapter as A

CAM = C.CAMERA_LEFT_UP


def _gap(idx, frame):
    return ev.GapObservation(local_gap_id="%s_G%d" % (CAM, idx),
                             confirmation_frame=frame, first_frame=frame - 2,
                             last_frame=frame + 2,
                             normalized_position=float(idx * 100),
                             max_confidence=0.9)


def _evidence(frames=(100, 200, 300, 400), fps=15.0, timeline=None,
              camera_id=CAM):
    gaps = [_gap(i, f) for i, f in enumerate(frames, start=1)]
    segments = [
        {"segment_id": ev.SEGMENT_ID_FORMAT % (camera_id, i),
         "segment_index": i,
         "start_frame": a.confirmation_frame, "end_frame": b.confirmation_frame,
         "opening_gap": a.local_gap_id, "closing_gap": b.local_gap_id,
         "canonical": False}
        for i, (a, b) in enumerate(zip(gaps, gaps[1:]), start=1)
    ]
    return ev.CameraEvidence(
        camera_id=camera_id, status=ev.STATUS_SEALED,
        timing=ev.CameraTiming(fps=fps, total_frames=1000),
        gaps=gaps, segments=segments,
        classification_timeline=timeline or [],
        engine_result={"video_info": {"width": 960, "height": 540}},
    )


# -----------------------------------------------------------------------------
# Shape
# -----------------------------------------------------------------------------

def test_n_gaps_yield_n_minus_one_wagons():
    state = A.build_local_state(_evidence(frames=(100, 200, 300, 400)))
    assert state.total_wagons == 3
    assert len(state.wagons) == 3


def test_ids_are_camera_local_never_canonical():
    state = A.build_local_state(_evidence())
    ids = [w.global_id for w in state.wagons]
    assert ids == ["LEFT_UP_W1", "LEFT_UP_W2", "LEFT_UP_W3"]
    for wagon_id in ids:
        assert not wagon_id.startswith("GW_")


def test_the_canonical_id_guard_accepts_this_state():
    """`assert_no_canonical_ids` is the existing invariant; it must still pass."""
    state = A.build_local_state(_evidence())
    doc = {"wagons": [{"id": w.global_id} for w in state.wagons]}
    ev.assert_no_canonical_ids(doc, where="local state test")


def test_one_gap_bounds_no_wagon():
    """N wagons need N+1 gaps. One gap must return None, not an empty state."""
    assert A.build_local_state(_evidence(frames=(100,))) is None


def test_no_gaps_returns_none():
    assert A.build_local_state(_evidence(frames=())) is None


# -----------------------------------------------------------------------------
# The window -- the reason this works at all
# -----------------------------------------------------------------------------

def test_camera_frame_ranges_carry_this_cameras_own_frames():
    state = A.build_local_state(_evidence(frames=(100, 200, 300)))
    first = state.wagons[0]
    assert first.local_range(CAM) == (100, 200)
    assert list(first.camera_frame_ranges) == [CAM]


def test_the_cache_builder_prefers_that_window():
    """`_wagon_local_range` must resolve the LOCAL window, not a projection.

    If it fell back to master-time projection the cache would extract the wrong
    frames and every feature would run on the wrong pixels.
    """
    state = A.build_local_state(_evidence(frames=(100, 200, 300)))
    got = _wagon_local_range(state.wagons[0], 15.0, 1000, 0.0, CAM)
    assert got == (100, 200)


def test_a_reversed_window_is_normalized():
    e = _evidence(frames=(300, 100))
    e.segments[0]["start_frame"], e.segments[0]["end_frame"] = 300, 100
    state = A.build_local_state(e)
    assert state.wagons[0].local_range(CAM) == (100, 300)


def test_a_segment_with_an_unusable_window_is_skipped_not_faked():
    e = _evidence(frames=(100, 200, 300))
    e.segments[0]["start_frame"] = None
    state = A.build_local_state(e)
    assert state.total_wagons == 1
    # Indices stay contiguous rather than leaving a hole at W1.
    assert [w.global_id for w in state.wagons] == ["LEFT_UP_W1"]
    assert [w.wagon_index for w in state.wagons] == [1]


# -----------------------------------------------------------------------------
# No global claims
# -----------------------------------------------------------------------------

def test_master_camera_is_this_camera_and_is_not_a_selection():
    """A single-camera state has no other candidate; the note says so."""
    state = A.build_local_state(_evidence())
    assert state.master_camera == CAM
    assert state.notes and "Not canonical" in state.notes[0]
    assert "no master selection" in state.notes[0]


def test_only_this_camera_appears_anywhere_in_the_state():
    state = A.build_local_state(_evidence())
    assert list(state.per_camera_local_counts) == [CAM]
    assert list(state.per_camera_gap_counts) == [CAM]
    for wagon in state.wagons:
        assert wagon.supporting_cameras == (CAM,)


def test_no_global_gap_timeline_is_invented():
    state = A.build_local_state(_evidence())
    assert state.global_gaps == []
    assert state.global_gap_count == 0


def test_two_cameras_may_disagree_and_nothing_reconciles_them():
    """A camera that missed a gap sees one fewer wagon. That is an observation."""
    right = A.build_local_state(_evidence(frames=(100, 200, 300, 400),
                                          camera_id=C.CAMERA_RIGHT_UP))
    left = A.build_local_state(_evidence(frames=(100, 300, 400),
                                         camera_id=C.CAMERA_LEFT_UP))
    assert right.total_wagons == 3
    assert left.total_wagons == 2
    assert right.wagons[0].global_id == "RIGHT_UP_W1"
    assert left.wagons[0].global_id == "LEFT_UP_W1"


# -----------------------------------------------------------------------------
# Classification comes from the persisted timeline, not a new classifier run
# -----------------------------------------------------------------------------

def test_classification_is_the_majority_of_the_persisted_timeline():
    timeline = ([{"frame_idx": f, "label": "wagon", "confidence": 0.9}
                 for f in range(100, 160, 10)]
                + [{"frame_idx": f, "label": "engine", "confidence": 0.5}
                   for f in range(160, 180, 10)])
    state = A.build_local_state(_evidence(frames=(100, 200), timeline=timeline))
    assert state.wagons[0].classification == "WAGON"
    assert state.wagons[0].classification_confidence == pytest.approx(0.9)


def test_an_empty_timeline_gives_unknown_not_a_guess():
    state = A.build_local_state(_evidence(frames=(100, 200), timeline=[]))
    assert state.wagons[0].classification == "UNKNOWN"
    assert state.wagons[0].classification_confidence == 0.0


def test_frames_outside_the_window_do_not_vote():
    timeline = [{"frame_idx": 900, "label": "brakevan", "confidence": 1.0}]
    state = A.build_local_state(_evidence(frames=(100, 200), timeline=timeline))
    assert state.wagons[0].classification == "UNKNOWN"


# -----------------------------------------------------------------------------
# Tracking document for Batch's renderer
# -----------------------------------------------------------------------------

def test_tracking_document_has_the_four_keys_the_renderer_reads():
    doc = A.per_camera_tracking_document(_evidence())
    assert list(doc) == [CAM]
    assert set(doc[CAM]) == {"fps", "total_frames", "width", "height"}
    assert doc[CAM]["fps"] == 15.0
    assert doc[CAM]["width"] == 960


def test_tracking_document_round_trips_through_the_real_loader(tmp_path):
    import json
    from reporting import _evidence_lookup as lookup
    p = tmp_path / "per_camera_tracking.json"
    p.write_text(json.dumps(A.per_camera_tracking_document(_evidence())))
    meta = lookup.load_per_camera_meta(str(p))
    assert meta[CAM]["fps"] == 15.0
    assert meta[CAM]["total_frames"] == 1000
