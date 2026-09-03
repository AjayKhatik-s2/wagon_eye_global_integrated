"""A camera-local `GlobalTrainState`, so Batch's own stages can run in Phase 1.

WHY THIS EXISTS
Batch's wagon cache, the three feature processors, the fusion builder and the
camera-report renderer all take one argument in common: `state: GlobalTrainState`.
They iterate `state.wagons` and address everything by `wagon.global_id`. That is
the only reason Phase 1 could not previously produce a complete camera report --
not the intelligence, just the shape of the input.

So this module builds that shape from one camera's own sealed evidence. Nothing
downstream is reimplemented: Phase 1 calls the SAME builders Batch calls, on a
state whose wagons are camera-local.

WHY IT IS NOT A FABRICATED ROSTER
Three properties of the existing types make this honest rather than a pretence:

  * `GlobalWagon.global_id` is a free-form `str`. Camera-local ids are legal, so
    `GW_n` is never invented -- ids here are `<CAMERA>_W<n>` and
    `evidence.assert_no_canonical_ids` still refuses anything shaped like `GW_`.

  * `GlobalWagon.camera_frame_ranges` + `local_range(camera_id)` carry an
    explicit per-camera window, and `materializer._wagon_local_range` PREFERS it
    over any master-time projection. A camera-local window is therefore the
    first-class case, not a workaround.

  * `GlobalTrainState.__post_init__` coerces the roster to a tuple explicitly so
    that "the immutability guarantee holds for hand-built states too".
    Hand-building one is a supported use, not a hack.

WHAT A CAMERA-LOCAL STATE DOES NOT CLAIM
`master_camera` is set to the camera itself because a single-camera state has no
other candidate, and every consumer requires the field to name a camera present
in the state. It is NOT a master SELECTION: the real master is whichever camera
has the most confirmed unique gaps, chosen in Global Assembly from all sealed
evidence. A local state never leaves Phase 1 and is never written to
`global_state/`.

Local wagon counts may legitimately differ between cameras -- a camera that
missed a gap sees one fewer wagon. That is an observation, not an error, and
nothing here reconciles it. Reconciliation is Global Assembly's job.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from core.global_state_loader import GlobalTrainState, GlobalWagon

from sequential import evidence as ev

#: Camera-local wagon id, e.g. `LEFT_UP_W1`. Deliberately distinct from the
#: canonical `GW_n` and from this module's sibling `<CAMERA>_SEG_<n>` segment
#: ids, so a local wagon can never be mistaken for either.
LOCAL_WAGON_ID_FORMAT = "%s_W%d"

#: What a camera-local state is FOR, recorded on the state's notes so a stray
#: one is recognisable in a dump.
LOCAL_STATE_NOTE = (
    "CAMERA-LOCAL state: wagons are this camera's own observations between its "
    "own confirmed gaps. Not canonical, no global roster, no master selection."
)


def local_wagon_id(camera_id: str, index: int) -> str:
    return LOCAL_WAGON_ID_FORMAT % (camera_id, index)


def _classification_for(timeline: List[Dict[str, Any]],
                        start_frame: int, end_frame: int) -> Tuple[str, float]:
    """Majority class over the frames this local wagon spans.

    Read from the classification timeline the trimming stage already produced
    and Phase 1 already persists -- the classifier is NOT run again. When the
    timeline carries nothing for the window, the class is UNKNOWN with zero
    confidence rather than a guess: an unclassified wagon is a real outcome and
    Batch's renderer already presents it.
    """
    if not timeline:
        return ("UNKNOWN", 0.0)
    counts: Dict[str, int] = {}
    conf_sum: Dict[str, float] = {}
    for entry in timeline:
        if not isinstance(entry, dict):
            continue
        frame = entry.get("frame_idx", entry.get("frame"))
        if frame is None:
            continue
        try:
            frame = int(frame)
        except (TypeError, ValueError):
            continue
        if not (start_frame <= frame <= end_frame):
            continue
        label = str(entry.get("label") or entry.get("class_name")
                    or entry.get("classification") or "").strip()
        if not label:
            continue
        counts[label] = counts.get(label, 0) + 1
        try:
            conf_sum[label] = conf_sum.get(label, 0.0) + float(
                entry.get("confidence") or 0.0)
        except (TypeError, ValueError):
            conf_sum.setdefault(label, 0.0)
    if not counts:
        return ("UNKNOWN", 0.0)
    label = max(counts, key=lambda k: (counts[k], conf_sum.get(k, 0.0)))
    n = counts[label]
    return (label.upper(), (conf_sum.get(label, 0.0) / n) if n else 0.0)


def build_local_state(camera_evidence: ev.CameraEvidence
                      ) -> Optional[GlobalTrainState]:
    """One camera's sealed evidence -> a camera-local `GlobalTrainState`.

    `None` when the camera confirmed fewer than two gaps: N wagons are delimited
    by N+1 gaps, so one gap bounds no wagon. Returning None rather than an empty
    state makes the caller decide explicitly, instead of silently running the
    cache and three feature models over nothing.

    Windows come from `evidence.segments`, which are already the frames between
    consecutive confirmed gaps in this camera's ORIGINAL video indices -- the
    same coordinate space `camera_frame_ranges` is defined in.
    """
    segments = list(camera_evidence.segments or [])
    if not segments:
        return None

    camera_id = camera_evidence.camera_id
    timing = camera_evidence.timing
    fps = float(getattr(timing, "fps", 0.0) or 0.0)
    timeline = camera_evidence.classification_timeline or []

    wagons: List[GlobalWagon] = []
    for index, segment in enumerate(segments, start=1):
        try:
            start_frame = int(segment.get("start_frame"))
            end_frame = int(segment.get("end_frame"))
        except (TypeError, ValueError):
            # A segment without a usable window bounds no frames; skipping it
            # keeps the local roster contiguous rather than inserting a wagon
            # whose cache would be empty.
            continue
        if end_frame < start_frame:
            start_frame, end_frame = end_frame, start_frame

        label, confidence = _classification_for(timeline, start_frame, end_frame)
        wagons.append(GlobalWagon(
            global_id=local_wagon_id(camera_id, len(wagons) + 1),
            wagon_index=len(wagons) + 1,
            # Master-time fields are this camera's OWN frame space. There is no
            # master here; the names come from the shared dataclass.
            start_frame_master=start_frame,
            end_frame_master=end_frame,
            start_time=(start_frame / fps) if fps > 0 else 0.0,
            end_time=(end_frame / fps) if fps > 0 else 0.0,
            classification=label,
            classification_confidence=confidence,
            supporting_cameras=(camera_id,),
            # THE field that makes Batch's cache builder address this camera's
            # own frames: `_wagon_local_range` prefers it over any projection.
            camera_frame_ranges={camera_id: {
                "start_frame": start_frame,
                "end_frame": end_frame,
                "status": camera_evidence.status,
            }},
            leading_gap=({"local_gap_id": segment.get("opening_gap")}
                         if segment.get("opening_gap") else None),
            trailing_gap=({"local_gap_id": segment.get("closing_gap")}
                          if segment.get("closing_gap") else None),
        ))

    if not wagons:
        return None

    return GlobalTrainState(
        total_wagons=len(wagons),
        wagons=tuple(wagons),
        # Not a selection -- see the module docstring. Consumers require this to
        # name a camera present in the state, and only one is.
        master_camera=camera_id,
        master_fps=fps,
        master_total_frames=int(getattr(timing, "total_frames", 0) or 0),
        per_camera_local_counts={camera_id: len(wagons)},
        per_camera_gap_counts={camera_id: len(camera_evidence.gaps or [])},
        per_camera_status={camera_id: camera_evidence.status},
        notes=[LOCAL_STATE_NOTE],
    )


def per_camera_tracking_document(camera_evidence: ev.CameraEvidence
                                 ) -> Dict[str, Dict[str, Any]]:
    """`{camera_id: {fps, total_frames, width, height}}` for ONE camera.

    `reporting._evidence_lookup.load_per_camera_meta` reads exactly these four
    keys and returns `{}` for a missing file, so the renderer already degrades
    gracefully -- but every value is in this camera's own persisted evidence, so
    Phase 1 can supply them properly instead of relying on that fallback. No
    global data is involved.
    """
    timing = camera_evidence.timing
    video = (camera_evidence.provenance or {}).get("video") or {}
    info = (camera_evidence.engine_result or {}).get("video_info") or {}

    def _first(*values):
        for value in values:
            if value:
                return value
        return 0

    return {camera_evidence.camera_id: {
        "fps": float(getattr(timing, "fps", 0.0) or 0.0),
        "total_frames": int(getattr(timing, "total_frames", 0) or 0),
        "width": int(_first(info.get("width"), video.get("width"))),
        "height": int(_first(info.get("height"), video.get("height"))),
    }}
