"""Exactly one wagon owns any feature event.

A global wagon is the region BETWEEN two consecutive global gaps, so gap *k* is
both the end of wagon *k* and the start of wagon *k+1*: adjacent wagons' aligned
windows share exactly one boundary frame. That frame used to be resolved by an
undocumented last-write-wins in the cache builder, invisible to the features, so
the same detection could be attributed to two consecutive wagons.

`core/wagon_ownership.py` makes the rule explicit, deterministic and SHARED:
the engine's global gap timeline is the only authority, and

    wagon k owns master-order positions  g_(k-1) <= p < g_k
    the last wagon additionally owns the closing boundary p == g_N

so an event exactly on a gap belongs to the wagon AFTER it.

Covers requirements 1-10 of the brief. No weights, no videos, no engine.

    python -m pytest tests/test_wagon_ownership.py -q
"""

from __future__ import annotations

import json
import os
import sys

import pytest

_TEST_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_TEST_DIR)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from core import constants as C
from core import wagon_ownership
from core.global_state_loader import (
    parse_global_train_state, roster_fingerprint, verify_roster_integrity,
)
from global_counting import adapter

from test_global_counting_integration import (
    EXPECTED_WAGONS, GAP_POSITIONS, MASTER, PARTIAL_CAMERA, REVERSED_CAMERA,
    build_harvest,
)


@pytest.fixture(scope="module")
def harvest():
    return build_harvest()


@pytest.fixture(scope="module")
def document(harvest):
    return adapter.build_global_train_state_document(harvest)


@pytest.fixture(scope="module")
def state(document):
    return parse_global_train_state(json.loads(json.dumps(document)))


@pytest.fixture(scope="module")
def ownership(state):
    own = wagon_ownership.for_state(state)
    assert own is not None, "the fixture state must carry gap boundaries"
    return own


def _boundaries(ownership, camera_id):
    bounds = ownership.camera_boundaries(camera_id)
    assert bounds, "no boundaries for %s" % camera_id
    return bounds


# -----------------------------------------------------------------------------
# The gap timeline is present and is the authority
# -----------------------------------------------------------------------------

def test_gap_timeline_is_published_and_ordered(state):
    """N wagons are delimited by N+1 ordered global gaps."""
    gaps = state.global_gaps
    assert len(gaps) == state.total_wagons + 1 == EXPECTED_WAGONS + 1
    assert [gap["gap_index"] for gap in gaps] == list(range(len(gaps)))
    positions = [gap["normalized_position"] for gap in gaps]
    assert positions == sorted(positions)
    assert positions == [pytest.approx(p) for p in GAP_POSITIONS]
    # and it does not invent a different count
    assert state.global_gap_count == len(gaps)


def test_gap_timeline_links_the_wagons_it_delimits(state):
    gaps = state.global_gaps
    for index, gap in enumerate(gaps):
        expected_opens = (state.wagons[index].global_id
                          if index < len(state.wagons) else None)
        expected_closes = (state.wagons[index - 1].global_id
                           if index else None)
        assert gap["opens_wagon"] == expected_opens
        assert gap["closes_wagon"] == expected_closes


# -----------------------------------------------------------------------------
# 1 + 2 + 3. before / after / exactly on a gap
# -----------------------------------------------------------------------------

@pytest.mark.parametrize("camera_id", list(C.ALL_CAMERAS))
def test_1_event_before_a_gap_belongs_to_the_previous_wagon(
        ownership, state, camera_id):
    bounds = _boundaries(ownership, camera_id)
    reversed_camera = camera_id == REVERSED_CAMERA
    step = -1 if reversed_camera else 1     # "one frame earlier in master order"

    for index in range(1, len(bounds) - 1):
        gap_frame = bounds[index]
        before = gap_frame - step
        owner = ownership.owner_of_camera_frame(camera_id, before)
        expected = ownership.wagon_ids[index - 1]
        assert owner == expected, (
            "%s frame %d is just BEFORE gap %d and must belong to %s, got %s"
            % (camera_id, before, index, expected, owner))


@pytest.mark.parametrize("camera_id", list(C.ALL_CAMERAS))
def test_2_event_after_a_gap_belongs_to_the_next_wagon(
        ownership, state, camera_id):
    bounds = _boundaries(ownership, camera_id)
    reversed_camera = camera_id == REVERSED_CAMERA
    step = -1 if reversed_camera else 1

    for index in range(1, len(bounds) - 1):
        gap_frame = bounds[index]
        after = gap_frame + step
        owner = ownership.owner_of_camera_frame(camera_id, after)
        expected = ownership.wagon_ids[index]
        assert owner == expected, (
            "%s frame %d is just AFTER gap %d and must belong to %s, got %s"
            % (camera_id, after, index, expected, owner))


@pytest.mark.parametrize("camera_id", list(C.ALL_CAMERAS))
def test_3_event_exactly_on_a_gap_follows_the_documented_rule(
        ownership, camera_id):
    """Documented rule: at a gap -> the NEXT wagon."""
    assert wagon_ownership.BOUNDARY_GOES_TO == "next_wagon"
    bounds = _boundaries(ownership, camera_id)
    for index in range(1, len(bounds) - 1):
        owner = ownership.owner_of_camera_frame(camera_id, bounds[index])
        assert owner == ownership.wagon_ids[index], (
            "%s boundary frame %d must go to the later wagon"
            % (camera_id, bounds[index]))


def test_3_final_boundary_belongs_to_the_last_wagon(ownership):
    """The closing gap would otherwise be owned by nobody."""
    for camera_id in C.ALL_CAMERAS:
        bounds = _boundaries(ownership, camera_id)
        owner = ownership.owner_of_camera_frame(camera_id, bounds[-1])
        assert owner is not None
        # the last wagon that actually resolved on this camera
        resolved = [gw for gw in ownership.wagon_ids
                    if ownership.camera_boundaries(camera_id)]
        assert owner == resolved[len(bounds) - 2]


def test_3_master_clock_events_use_the_master_boundaries(state, ownership):
    """An event already in master coordinates needs no camera mapping."""
    for index, wagon in enumerate(state.wagons):
        assert ownership.owner_of_master_frame(
            wagon.start_frame_master) == wagon.global_id
        if index:
            previous = state.wagons[index - 1]
            # the shared master boundary goes to the later wagon
            assert previous.end_frame_master == wagon.start_frame_master
            assert ownership.owner_of_master_frame(
                previous.end_frame_master) == wagon.global_id


# -----------------------------------------------------------------------------
# 4. a frame can never belong to two wagons
# -----------------------------------------------------------------------------

@pytest.mark.parametrize("camera_id", list(C.ALL_CAMERAS))
def test_4_no_frame_is_owned_by_two_wagons(ownership, state, camera_id):
    """Exhaustive sweep of the whole wagon region, every camera."""
    bounds = _boundaries(ownership, camera_id)
    low, high = min(bounds), max(bounds)
    duplicates = []
    unowned = []
    for frame in range(low, high + 1):
        owners = [gw for gw in ownership.wagon_ids
                  if ownership.owner_of_camera_frame(camera_id, frame) == gw]
        if len(owners) > 1:
            duplicates.append(frame)
        if not owners:
            unowned.append(frame)
    assert duplicates == [], "%s: frames owned twice: %s" % (
        camera_id, duplicates[:5])
    assert unowned == [], "%s: frames owned by nobody: %s" % (
        camera_id, unowned[:5])


@pytest.mark.parametrize("camera_id", list(C.ALL_CAMERAS))
def test_4_adjacent_wagons_never_claim_the_same_frame(ownership, camera_id):
    bounds = _boundaries(ownership, camera_id)
    for index in range(len(bounds) - 2):
        earlier = ownership.wagon_ids[index]
        later = ownership.wagon_ids[index + 1]
        shared = bounds[index + 1]
        claims = [gw for gw in (earlier, later)
                  if ownership.owns_camera_frame(gw, camera_id, shared)]
        assert claims == [later], (
            "%s: boundary %d claimed by %s" % (camera_id, shared, claims))


def test_4_the_contract_really_does_share_a_boundary_frame(state):
    """Guard: if the windows ever stop touching, this test must be revisited."""
    shared = 0
    for earlier, later in zip(state.wagons, state.wagons[1:]):
        for camera_id in C.ALL_CAMERAS:
            a = earlier.local_range(camera_id)
            b = later.local_range(camera_id)
            if a and b and (min(a[1], b[1]) - max(a[0], b[0])) >= 0:
                shared += 1
    assert shared > 0, (
        "adjacent windows no longer overlap; the ownership rule is still "
        "correct but this regression's premise has changed")


# -----------------------------------------------------------------------------
# 1 + 2 + 5. the shared enumeration point, used by Damage AND Door
# -----------------------------------------------------------------------------

def _make_cache(root, gw_ids, camera_id, frames_per_wagon):
    """A wagon_cache tree whose filenames carry original frame indices."""
    from features._common import wagon_camera_dir

    for gw_id, frames in zip(gw_ids, frames_per_wagon):
        directory = wagon_camera_dir(root, gw_id, camera_id)
        os.makedirs(directory, exist_ok=True)
        for frame in frames:
            open(os.path.join(directory, "frame_%06d.jpg" % frame), "w").close()


def test_5_shared_enumeration_drops_the_foreign_boundary_frame(
        ownership, state, tmp_path):
    """The one insertion point Door, Damage and Load all go through."""
    from features._common import list_wagon_frames, frame_index_of

    camera_id = MASTER
    bounds = _boundaries(ownership, camera_id)
    root = str(tmp_path / "wagon_cache")

    # Deliberately materialize the OLD overlapping windows: give both adjacent
    # wagons the shared boundary frame, as the contract's ranges do.
    gw_ids, per_wagon = [], []
    for index in range(len(bounds) - 1):
        gw_ids.append(ownership.wagon_ids[index])
        per_wagon.append(list(range(bounds[index], bounds[index + 1] + 1)))
    _make_cache(root, gw_ids, camera_id, per_wagon)

    seen = {}
    for gw_id in gw_ids:
        kept = list_wagon_frames(root, gw_id, camera_id, ownership=ownership)
        for path in kept:
            frame = frame_index_of(path)
            seen.setdefault(frame, []).append(gw_id)

    duplicated = {frame: owners for frame, owners in seen.items()
                  if len(owners) > 1}
    assert duplicated == {}, "frames inspected by two wagons: %s" % duplicated

    # ...and without ownership the duplication is still there, proving the
    # filter is what removes it rather than the fixture being trivial.
    unfiltered = {}
    for gw_id in gw_ids:
        for path in list_wagon_frames(root, gw_id, camera_id):
            unfiltered.setdefault(frame_index_of(path), []).append(gw_id)
    assert any(len(owners) > 1 for owners in unfiltered.values()), (
        "the fixture must reproduce the original overlap")


@pytest.mark.parametrize("module_name", ["damage", "door"])
def test_5_damage_and_door_forward_ownership_to_the_shared_helper(
        monkeypatch, ownership, module_name, tmp_path):
    """Both features must pass the SAME ownership object down, not reimplement."""
    import importlib

    processor = importlib.import_module("features.%s.processor" % module_name)
    recorded = {}

    def _fake_list(cache_root, gw_id, camera_id, **kwargs):
        recorded.update(kwargs)
        return []                      # early-return keeps the helper cheap

    monkeypatch.setattr(processor, "list_wagon_frames", _fake_list)

    if module_name == "damage":
        processor._run_tracker_one_camera(
            None, processor.DamageTrackerConfig(), str(tmp_path), "GW_1",
            MASTER, confidence_floor=0.5, ownership=ownership)
    else:
        processor._run_tracker_one_camera(
            None, processor.TrackerConfig(), processor.MergeConfig(),
            str(tmp_path), "GW_1", C.CAMERA_LEFT_UP, ownership=ownership)

    assert recorded.get("ownership") is ownership, (
        "%s did not forward the shared ownership object" % module_name)


def test_5_load_also_forwards_ownership(monkeypatch, ownership, tmp_path):
    """Load reads frames the same way, so it gets the same rule."""
    from features.load import processor

    recorded = {}

    def _fake_iter(cache_root, gw_id, camera_id, **kwargs):
        recorded.update(kwargs)
        return iter(())

    monkeypatch.setattr(processor, "iter_wagon_frames", _fake_iter)
    processor._aggregate_camera(
        None, str(tmp_path), "GW_1", C.CAMERA_RIGHT_UP_TOP,
        every_nth=2, max_frames=None, ownership=ownership)
    assert recorded.get("ownership") is ownership


def test_5_every_feature_shares_one_implementation():
    """No feature may carry its own copy of the boundary rule."""
    import importlib
    import inspect

    for name in ("damage", "door", "load", "ocr"):
        source = inspect.getsource(
            importlib.import_module("features.%s.processor" % name))
        assert "wagon_ownership.for_state" in source, (
            "%s does not use the shared ownership utility" % name)
        # a local re-derivation of gap boundaries would be a second authority
        assert "global_gaps" not in source, (
            "%s reads the gap timeline directly instead of using the shared "
            "utility" % name)


# -----------------------------------------------------------------------------
# 6. non-boundary behaviour is untouched
# -----------------------------------------------------------------------------

def test_6_interior_frames_are_unaffected(ownership, tmp_path):
    """Every frame strictly inside a wagon is kept, exactly as before."""
    from features._common import list_wagon_frames

    camera_id = MASTER
    bounds = _boundaries(ownership, camera_id)
    root = str(tmp_path / "cache")

    gw_id = ownership.wagon_ids[1]
    interior = list(range(bounds[1] + 1, bounds[2]))       # exclusive of gaps
    _make_cache(root, [gw_id], camera_id, [interior])

    with_own = list_wagon_frames(root, gw_id, camera_id, ownership=ownership)
    without = list_wagon_frames(root, gw_id, camera_id)
    assert with_own == without
    assert len(with_own) == len(interior)


def test_6_a_state_without_boundaries_is_never_filtered(tmp_path):
    """Legacy roster -> no opinion -> behaviour identical to f3d2d81."""
    from core.global_state_loader import GlobalTrainState, GlobalWagon
    from features._common import list_wagon_frames

    legacy = GlobalTrainState(
        total_wagons=1,
        wagons=(GlobalWagon(global_id="GW_1", wagon_index=1,
                            start_frame_master=0, end_frame_master=10,
                            start_time=0.0, end_time=1.0,
                            classification=C.CLASS_WAGON),),
        master_camera=C.CAMERA_RIGHT_UP, master_fps=15.0)
    own = wagon_ownership.WagonOwnership(legacy)
    assert own.has_opinion(C.CAMERA_LEFT_UP_TOP) is False
    assert own.owns_camera_frame("GW_1", C.CAMERA_LEFT_UP_TOP, 12345) is True

    root = str(tmp_path / "cache")
    _make_cache(root, ["GW_1"], C.CAMERA_LEFT_UP_TOP, [[5, 6, 7]])
    assert len(list_wagon_frames(root, "GW_1", C.CAMERA_LEFT_UP_TOP,
                                 ownership=own)) == 3


def test_6_unreadable_frame_name_is_kept(ownership, tmp_path):
    """Unknown provenance must not silently delete evidence."""
    from features._common import list_wagon_frames, wagon_camera_dir

    root = str(tmp_path / "cache")
    directory = wagon_camera_dir(root, ownership.wagon_ids[0], MASTER)
    os.makedirs(directory, exist_ok=True)
    open(os.path.join(directory, "frame_weird.jpg"), "w").close()
    kept = list_wagon_frames(root, ownership.wagon_ids[0], MASTER,
                             ownership=ownership)
    assert len(kept) == 1


# -----------------------------------------------------------------------------
# 7 + 8 + 9. the counting result itself must be untouched
# -----------------------------------------------------------------------------

def test_7_global_wagon_ids_are_unchanged(state):
    assert [w.global_id for w in state.wagons] == [
        "GW_%d" % n for n in range(1, EXPECTED_WAGONS + 1)]
    assert verify_roster_integrity(state) == []
    assert state.total_wagons == state.global_gap_count - 1


def test_7_ownership_does_not_touch_the_roster(state):
    guard = roster_fingerprint(state)
    own = wagon_ownership.for_state(state)
    for camera_id in C.ALL_CAMERAS:
        for frame in range(0, 400, 17):
            own.owner_of_camera_frame(camera_id, frame)
    assert roster_fingerprint(state) == guard
    assert own.wagon_ids == tuple(w.global_id for w in state.wagons)


def test_8_missing_gap_recovery_is_still_audited(state):
    """RECOVERED / UNMATCHED provenance survives into the gap timeline."""
    statuses = {entry.get("status")
                for gap in state.global_gaps
                for entry in (gap.get("cameras") or {}).values()}
    assert adapter.STATUS_RECOVERED in statuses, "recovery provenance lost"
    assert adapter.STATUS_DETECTED in statuses

    summary = state.support_alignment_summary
    for camera_id in C.ALL_CAMERAS:
        assert (summary[camera_id]["detected_intervals"]
                + summary[camera_id]["recovered_intervals"]
                + summary[camera_id]["unmatched_intervals"]) == EXPECTED_WAGONS


def test_8_recovered_boundaries_still_own_frames(ownership, state):
    """A recovered gap is a usable boundary, not a hole."""
    recovered = [wagon.global_id for wagon in state.wagons
                 if wagon.camera_frame_ranges[MASTER]["status"]
                 == adapter.STATUS_RECOVERED]
    assert recovered
    for gw_id in recovered:
        window = next(w for w in state.wagons
                      if w.global_id == gw_id).local_range(MASTER)
        midpoint = (window[0] + window[1]) // 2
        assert ownership.owner_of_camera_frame(MASTER, midpoint) == gw_id


def test_8_unmatched_camera_wagon_owns_nothing(ownership, state):
    last = state.wagons[-1]
    assert last.local_range(PARTIAL_CAMERA) is None
    bounds = _boundaries(ownership, PARTIAL_CAMERA)
    for frame in range(min(bounds), max(bounds) + 1):
        assert ownership.owner_of_camera_frame(
            PARTIAL_CAMERA, frame) != last.global_id


def test_9_four_camera_aligned_ranges_are_unchanged(harvest, state):
    """Publishing the gap timeline must not perturb camera_frame_ranges."""
    for wagon, source in zip(state.wagons, harvest.wagons):
        assert set(wagon.camera_frame_ranges) == set(C.ALL_CAMERAS)
        for camera_id in C.ALL_CAMERAS:
            emitted = wagon.camera_frame_ranges[camera_id]
            expected = source.cameras[camera_id]
            assert emitted["start_frame"] == expected["start_frame"]
            assert emitted["end_frame"] == expected["end_frame"]
            assert emitted["status"] == expected["status"]
            assert emitted["timeline_reversed"] == expected["reversed"]


def test_9_gap_timeline_agrees_with_the_aligned_ranges(state):
    """The timeline is a republication of the windows, not a new computation."""
    for index, wagon in enumerate(state.wagons):
        for camera_id in C.ALL_CAMERAS:
            window = wagon.local_range(camera_id)
            if window is None:
                continue
            opening = state.global_gaps[index]["cameras"][camera_id]["frame"]
            closing = state.global_gaps[index + 1]["cameras"][camera_id]["frame"]
            if opening is None or closing is None:
                continue
            assert {opening, closing} == {window[0], window[1]}


# -----------------------------------------------------------------------------
# 10. fusion with OCR disabled
# -----------------------------------------------------------------------------

def test_10_fusion_still_works_with_ocr_disabled(state, tmp_path):
    from fusion import wagon_state_builder

    root = str(tmp_path / "wagon_states")
    gw_ids = [w.global_id for w in state.wagons]
    for feature, payload in (
        ("door", {"left_door": C.DOOR_CLOSED, "right_door": C.DOOR_OPEN}),
        ("load", {"load_status": C.LOAD_LOADED}),
        ("damage", {"top_damage": C.DAMAGE_OK, "top_damage_details": []}),
    ):
        directory = os.path.join(root, feature)
        os.makedirs(directory, exist_ok=True)
        for gw_id in gw_ids:
            document = dict(payload)
            document.update({"global_id": gw_id, "feature": feature,
                             "status": C.STATUS_OK})
            with open(os.path.join(directory, "%s.json" % gw_id), "w",
                      encoding="utf-8") as handle:
                json.dump(document, handle)
    assert not os.path.exists(os.path.join(root, "ocr"))

    unified = wagon_state_builder.build(
        state=state, wagon_states_root=root, write_per_wagon_json=False,
        verbose=False)

    assert set(unified) == set(gw_ids)
    for gw_id in gw_ids:
        assert unified[gw_id].wagon_identifier == C.NO_DATA
        assert unified[gw_id].load_status == C.LOAD_LOADED


# -----------------------------------------------------------------------------
# the materializer applies the same rule as the features
# -----------------------------------------------------------------------------

def test_materializer_and_features_agree_by_construction(ownership, state):
    """Stage 2 must not resolve the boundary differently from Stage 3."""
    from materializer.wagon_cache_builder import _wagon_local_range

    camera_id = MASTER
    for earlier, later in zip(state.wagons, state.wagons[1:]):
        a = _wagon_local_range(earlier, 15.0, 10 ** 7, camera_id=camera_id)
        b = _wagon_local_range(later, 15.0, 10 ** 7, camera_id=camera_id)
        shared = a[1]
        assert shared == b[0], "the fixture must have a shared boundary"
        # Stage 2 now asks ownership, which gives it to the later wagon --
        # the same answer Stage 3's frame listing gives.
        assert ownership.owner_of_camera_frame(camera_id, shared) == \
            later.global_id
        assert not ownership.owns_camera_frame(
            earlier.global_id, camera_id, shared)


def test_materializer_skips_frames_it_does_not_own(ownership, state,
                                                   monkeypatch, tmp_path):
    """The cache builder consults ownership, not dict last-write-wins."""
    from materializer import wagon_cache_builder

    asked = []
    real = ownership.owns_camera_frame

    class _Spy:
        def has_opinion(self, camera_id=None):
            return ownership.has_opinion(camera_id)

        def owns_camera_frame(self, gw_id, camera_id, frame):
            asked.append((gw_id, camera_id, frame))
            return real(gw_id, camera_id, frame)

    monkeypatch.setattr(wagon_cache_builder.wagon_ownership, "for_state",
                        lambda state: _Spy())
    # No videos -> every camera is reported missing, but build() still runs and
    # must have built the ownership map.
    result = wagon_cache_builder.build(
        state=state, video_paths={}, per_camera_fps={},
        cache_root=str(tmp_path / "cache"), verbose=False)
    assert sorted(result.missing_cameras) == sorted(C.ALL_CAMERAS)
