"""Which wagon owns a feature event -- one rule, shared by every feature.

THE PROBLEM
-----------
A global wagon is the region BETWEEN two consecutive global gaps, so gap *k* is
simultaneously the end of wagon *k* and the start of wagon *k+1*. The Stage-1
contract reflects that faithfully: adjacent wagons' `camera_frame_ranges` share
exactly one boundary frame, on forward and reversed cameras alike.

That shared frame used to be resolved by accident. `wagon_cache_builder` builds
a `{frame -> wagon}` dict, so the last wagon written simply overwrote the
earlier one -- an undocumented last-write-wins that depended on roster
iteration order, and that no feature could see or reason about. Any consumer
reading the contract directly (rather than the cache) saw a genuine overlap and
could attribute the same boundary frame, and therefore the same detection, to
two consecutive wagons.

THE RULE
--------
The global gap timeline produced by the counting engine is the single source of
truth. Nothing here detects, recounts or re-aligns anything: it only compares a
frame against the ordered gap boundaries the engine already established.

Let `g_0 < g_1 < ... < g_N` be the global gaps in MASTER order, so wagon *k*
(1-based) lies between `g_(k-1)` and `g_k`. Ownership is:

    wagon k owns a frame whose master-order position p satisfies
        g_(k-1) <= p < g_k
    and the LAST wagon additionally owns the closing boundary p == g_N.

In words: **an event exactly on a gap belongs to the wagon AFTER the gap** --
"before a gap -> previous wagon, at or after a gap -> next wagon". Intervals are
half-open, so the union covers every frame in the wagon region and no frame is
covered twice. The only inclusive endpoint is the very last gap, which would
otherwise belong to no wagon at all.

This is deliberately the same outcome the old last-write-wins produced (the
boundary frame went to the later wagon), so the fix documents and enforces the
existing behaviour instead of shifting every boundary by one frame.

REVERSED CAMERAS
----------------
A reversed camera's frames run backwards against the master, so its boundary
frames DESCEND. The rule is applied in master order, never in raw frame order,
by mapping each frame to a monotonic master-order key (`+frame` forward,
`-frame` reversed). The camera's own aligned frame ranges are used as-is -- no
alignment is recomputed here.

COORDINATES
-----------
`owner_of_master_frame` for events already on the master clock;
`owner_of_camera_frame` for camera-local events, using that camera's existing
aligned boundaries. Both answer with exactly one wagon id, or None for a frame
outside the wagon region (the locomotive / brake-van tail, which owns no GW id).

LEGACY STATES
-------------
A state carrying no aligned ranges and no gap timeline yields
`has_opinion(camera) == False`, and every helper then declines to filter, so a
roster produced by the retained counter behaves exactly as it did before.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from core.global_state_loader import GlobalTrainState

# An event exactly on a gap belongs to the wagon after it.
BOUNDARY_GOES_TO = "next_wagon"


class WagonOwnership:
    """Resolves `(camera, frame) -> exactly one global wagon id`."""

    def __init__(self, state: GlobalTrainState) -> None:
        self._wagon_ids: Tuple[str, ...] = tuple(
            wagon.global_id for wagon in state.wagons)
        # camera_id -> (boundaries_in_master_order, sign)
        self._camera: Dict[str, Tuple[List[int], int]] = {}
        self._master: Optional[Tuple[List[int], int]] = None
        self._build(state)

    # ------------------------------------------------------------------
    # construction
    # ------------------------------------------------------------------

    def _build(self, state: GlobalTrainState) -> None:
        wagons = state.wagons
        if not wagons:
            return

        # ---- master clock -------------------------------------------------
        master: List[Tuple[str, int, int]] = []
        for wagon in wagons:
            start = int(wagon.start_frame_master)
            end = int(wagon.end_frame_master)
            if end >= start:
                master.append((wagon.global_id, start, end))
        if master:
            self._master = (1, sorted(master, key=lambda item: item[1]))

        # ---- per camera ---------------------------------------------------
        gap_frames = self._gap_frames_by_camera(state)
        for camera_id in _cameras_mentioned(state):
            sign = -1 if _camera_is_reversed(state, camera_id) else 1
            boundaries = gap_frames.get(camera_id) or []
            intervals: List[Tuple[str, int, int]] = []
            for index, wagon in enumerate(wagons):
                bounds = self._wagon_bounds(
                    wagon, camera_id, index, boundaries, sign)
                if bounds is None:
                    continue
                start_key, end_key = bounds
                intervals.append((wagon.global_id, start_key, end_key))
            if intervals:
                self._camera[camera_id] = (
                    sign, sorted(intervals, key=lambda item: item[1]))

    @staticmethod
    def _gap_frames_by_camera(state: GlobalTrainState,
                              ) -> Dict[str, List[Optional[int]]]:
        """`camera -> [frame per gap]` from the engine's own gap timeline."""
        out: Dict[str, List[Optional[int]]] = {}
        gaps = state.global_gaps or ()
        if len(gaps) != len(state.wagons) + 1:
            return out
        for gap in gaps:
            for camera_id, entry in (gap.get("cameras") or {}).items():
                frame = entry.get("frame") if isinstance(entry, dict) else None
                out.setdefault(camera_id, []).append(
                    None if frame is None else int(frame))
        return {camera_id: frames for camera_id, frames in out.items()
                if len(frames) == len(gaps)}

    def _wagon_bounds(self, wagon, camera_id: str, index: int,
                      boundaries: Sequence[Optional[int]], sign: int,
                      ) -> Optional[Tuple[int, int]]:
        """`(start_key, end_key)` for one wagon on one camera, in master order.

        The gap timeline is the authority; a wagon whose neighbouring gap could
        not be projected onto this camera falls back to its own aligned window,
        which holds the same two boundary frames. Neither path recomputes
        anything -- both read numbers the engine already produced.
        """
        opening = closing = None
        if len(boundaries) > index + 1:
            opening, closing = boundaries[index], boundaries[index + 1]

        if opening is None or closing is None:
            window = wagon.local_range(camera_id)
            if window is None:
                return None
            low, high = window
            reversed_camera = sign < 0
            opening, closing = ((high, low) if reversed_camera else (low, high))

        start_key, end_key = sign * int(opening), sign * int(closing)
        if end_key < start_key:
            return None
        return (start_key, end_key)

    # ------------------------------------------------------------------
    # queries
    # ------------------------------------------------------------------

    def has_opinion(self, camera_id: Optional[str] = None) -> bool:
        """False when this state carries no usable boundaries.

        Callers must treat False as "do not filter", so a legacy roster keeps
        its original behaviour.
        """
        if camera_id is None:
            return bool(self._camera) or self._master is not None
        return camera_id in self._camera

    @property
    def wagon_ids(self) -> Tuple[str, ...]:
        return self._wagon_ids

    def camera_boundaries(self, camera_id: str) -> Optional[List[int]]:
        """The wagon boundary frames for one camera, in master order."""
        entry = self._camera.get(camera_id)
        if entry is None:
            return None
        sign, intervals = entry
        frames = [sign * intervals[0][1]]
        frames.extend(sign * end for _gw, _start, end in intervals)
        return frames

    @staticmethod
    def _resolve(intervals: Sequence[Tuple[str, int, int]],
                 key: int) -> Optional[str]:
        """The single owner of `key` under the half-open rule.

        `[start, end)` everywhere, so a boundary shared by two wagons resolves
        to the later one and no key can match two adjacent intervals. The
        furthest-reaching interval also owns its closing boundary, which would
        otherwise belong to nobody. If windows ever overlapped by more than the
        one shared boundary, the latest-starting wagon wins -- the same
        "at or after a gap -> next wagon" direction.
        """
        matches = [(start, gw_id) for gw_id, start, end in intervals
                   if start <= key < end]
        if not matches:
            last_end = max(end for _gw, _start, end in intervals)
            if key == last_end:
                matches = [(start, gw_id) for gw_id, start, end in intervals
                           if end == last_end]
        if not matches:
            return None
        return max(matches)[1]

    def owner_of_camera_frame(self, camera_id: str,
                              frame: int) -> Optional[str]:
        """Owner of a CAMERA-LOCAL frame index (original video numbering)."""
        entry = self._camera.get(camera_id)
        if entry is None:
            return None
        sign, intervals = entry
        return self._resolve(intervals, sign * int(frame))

    def owner_of_master_frame(self, frame: int) -> Optional[str]:
        """Owner of a frame already expressed on the master clock."""
        if self._master is None:
            return None
        sign, intervals = self._master
        return self._resolve(intervals, sign * int(frame))

    def owns_camera_frame(self, gw_id: str, camera_id: str,
                          frame: int) -> bool:
        """True when `gw_id` is the sole owner of this camera frame.

        Returns True unconditionally when this state has no boundaries for the
        camera, so an unknown-provenance roster is never silently emptied.
        """
        if not self.has_opinion(camera_id):
            return True
        return self.owner_of_camera_frame(camera_id, frame) == gw_id

    # ------------------------------------------------------------------
    # convenience filters
    # ------------------------------------------------------------------

    def filter_camera_frames(self, gw_id: str, camera_id: str,
                             frames: Iterable[int]) -> List[int]:
        """Keep only the frames this wagon owns."""
        if not self.has_opinion(camera_id):
            return list(frames)
        return [frame for frame in frames
                if self.owner_of_camera_frame(camera_id, frame) == gw_id]

    def filter_events(self, gw_id: str, camera_id: str,
                      events: Sequence[Any], *,
                      frame_key: str = "frame_idx") -> List[Any]:
        """Keep only the events whose source frame this wagon owns.

        `events` may be dicts or objects; `frame_key` names the frame field.
        An event with no usable frame is kept -- dropping evidence because its
        provenance is unknown would be worse than a boundary duplicate.
        """
        if not self.has_opinion(camera_id):
            return list(events)
        kept: List[Any] = []
        for event in events:
            frame = _event_frame(event, frame_key)
            if frame is None or self.owner_of_camera_frame(
                    camera_id, frame) == gw_id:
                kept.append(event)
        return kept


# -----------------------------------------------------------------------------
# helpers
# -----------------------------------------------------------------------------

def _strictly_increasing(values: Sequence[int]) -> bool:
    return all(earlier < later
               for earlier, later in zip(values, values[1:]))


def _cameras_mentioned(state: GlobalTrainState) -> List[str]:
    seen: List[str] = []
    for gap in state.global_gaps or ():
        for camera_id in (gap.get("cameras") or {}):
            if camera_id not in seen:
                seen.append(camera_id)
    for wagon in state.wagons:
        for camera_id in (wagon.camera_frame_ranges or {}):
            if camera_id not in seen:
                seen.append(camera_id)
    return seen


def _camera_is_reversed(state: GlobalTrainState, camera_id: str) -> bool:
    """Direction from the state, never inferred from frame numbers alone."""
    summary = (state.support_alignment_summary or {}).get(camera_id)
    if isinstance(summary, dict) and "timeline_reversed" in summary:
        return bool(summary["timeline_reversed"])
    for wagon in state.wagons:
        entry = (wagon.camera_frame_ranges or {}).get(camera_id)
        if isinstance(entry, dict) and "timeline_reversed" in entry:
            return bool(entry["timeline_reversed"])
    return False


def _event_frame(event: Any, frame_key: str) -> Optional[int]:
    if isinstance(event, dict):
        value = event.get(frame_key)
    else:
        value = getattr(event, frame_key, None)
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def for_state(state: Optional[GlobalTrainState]) -> Optional[WagonOwnership]:
    """Build the ownership map for a state, or None when there is nothing to do."""
    if state is None or not state.wagons:
        return None
    ownership = WagonOwnership(state)
    return ownership if ownership.has_opinion() else None
