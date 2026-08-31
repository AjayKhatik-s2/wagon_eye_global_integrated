"""Translate the new engine's global timeline into the OLD Stage-1 contract.

Pure transformation: this module imports nothing from the engine and holds no
engine objects, so it is testable without weights, videos or the engine itself.

    GlobalCountingResult  (plain data, harvested by runner.py)
            |
            v
    global_train_state.json     exactly the schema core.global_state_loader
    per_camera_tracking.json    already parses -- plus additive fields

Three conventions this module is responsible for:

1. **Master clock.** The old contract's `start_frame_master` / `start_time`
   live on the master camera's ORIGINAL frame timeline. The runner has already
   shifted every engine frame index out of trimmed space, so the numbers here
   are original-video frames throughout.

2. **Per-camera windows are explicit.** The old contract derived a camera's
   window as `(start_time - delta) * local_fps`, which assumes the four videos
   differ by at most a constant offset. The new engine aligns cameras with a
   scale AND a direction, and supports a fully reversed timeline, which no
   single offset can express. Every wagon therefore carries an additive
   `camera_frame_ranges` entry holding the aligned window per camera. The
   materializer prefers it and falls back to the old formula when absent, so a
   state written by the old counter still materializes exactly as before.

3. **Non-wagon objects.** The engine trims to the confirmed wagon region before
   counting, so the locomotive and brake van are outside the wagon timeline BY
   DESIGN and receive no GW id -- which is also what the old master-fixed
   counter did. Their counts are reported through `wagon_window`, the same
   channel `GlobalTrainState.engine_count` / `brake_van_count` already read, so
   the reporting layer needs no change at all.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Tuple

from core import constants as C

# Wagon id format. Identical to the old pipeline's -- there is no second
# numbering anywhere in the system.
GLOBAL_ID_FORMAT = "GW_%d"

STATE_SCHEMA = "wagon_eye.global_train_state.v1"
FUSION_MODE = "global_wagon_app_dynamic_master"
ENGINE_NAME = "global_wagon_app"

# Interval statuses the engine emits per camera.
STATUS_DETECTED = "DETECTED"
STATUS_RECOVERED = "RECOVERED"
STATUS_UNMATCHED = "UNMATCHED"

# `camera_time_offsets()` only trusts these two; anything else contributes 0.0.
OFFSET_REFERENCE = "REFERENCE"
OFFSET_RESOLVED = "RESOLVED"
# A reversed camera runs backwards against the master, so no single additive
# delta describes it. Saying so explicitly keeps the old offset consumers on
# their safe 0.0 path while the explicit frame ranges carry the real mapping.
OFFSET_REVERSED = "REVERSED_NOT_APPLICABLE"
OFFSET_UNRESOLVED = "UNRESOLVED"


def _seconds(frame: Optional[int], fps: float) -> Optional[float]:
    if frame is None or fps <= 0:
        return None
    return round(float(frame) / fps, 6)


def _camera_offset_entry(camera, wagons, master_fps: float,
                         is_master: bool) -> Dict[str, Any]:
    """Estimate this camera's clock delta, or say honestly that it has none.

    `t_global = t_local + delta`. The estimate is the mean difference between
    each wagon's master-clock centre and its centre on this camera, over the
    wagons both cameras actually resolved.
    """
    if is_master:
        return {"status": OFFSET_REFERENCE, "delta": 0.0, "samples": 0,
                "note": "master camera defines the global clock"}
    if camera.is_reversed:
        return {"status": OFFSET_REVERSED, "delta": 0.0, "samples": 0,
                "note": "timeline runs opposite to the master; use "
                        "camera_frame_ranges"}
    if camera.fps <= 0 or master_fps <= 0:
        return {"status": OFFSET_UNRESOLVED, "delta": 0.0, "samples": 0,
                "note": "fps unknown"}

    deltas: List[float] = []
    for wagon in wagons:
        interval = wagon.cameras.get(camera.camera_id) or {}
        if interval.get("status") == STATUS_UNMATCHED:
            continue
        start_frame, end_frame = interval.get("start_frame"), interval.get("end_frame")
        if start_frame is None or end_frame is None:
            continue
        local_centre = 0.5 * (start_frame + end_frame) / camera.fps
        master_centre = 0.5 * (wagon.global_start_frame_master
                               + wagon.global_end_frame_master) / master_fps
        deltas.append(master_centre - local_centre)

    if not deltas:
        return {"status": OFFSET_UNRESOLVED, "delta": 0.0, "samples": 0,
                "note": "no wagon resolved on this camera"}

    mean = sum(deltas) / len(deltas)
    spread = max(deltas) - min(deltas)
    return {
        "status": OFFSET_RESOLVED,
        "delta": round(mean, 6),
        "samples": len(deltas),
        "spread_seconds": round(spread, 6),
        "note": "mean master-vs-local wagon-centre difference",
    }


def build_global_train_state_document(result) -> Dict[str, Any]:
    """The old Stage-1 JSON, built from the harvested engine result."""
    master = result.cameras[result.master_camera]
    master_fps = master.fps

    # Attach the master-clock window to each wagon first: the offset estimator
    # and the wagon records both need it.
    for wagon in result.wagons:
        interval = wagon.cameras.get(result.master_camera) or {}
        start_frame = interval.get("start_frame")
        end_frame = interval.get("end_frame")
        if start_frame is None or end_frame is None:
            # The master camera detected every global gap by construction, so
            # this should not happen; fall back to the normalized position
            # projected onto the master's own trimmed span rather than crash.
            span = max(1, master.trimmed_total_frames - 1)
            scale = result.normalized_scale or 1000.0
            start_frame = master.crop_start_frame + int(
                round(wagon.global_start_position / scale * span))
            end_frame = master.crop_start_frame + int(
                round(wagon.global_end_position / scale * span))
        wagon.global_start_frame_master = int(min(start_frame, end_frame))
        wagon.global_end_frame_master = int(max(start_frame, end_frame))

    wagon_documents: List[Dict[str, Any]] = []
    for position, wagon in enumerate(result.wagons, start=1):
        supporting = [camera_id for camera_id in C.ALL_CAMERAS
                      if (wagon.cameras.get(camera_id) or {}).get("status")
                      != STATUS_UNMATCHED]

        camera_frame_ranges: Dict[str, Dict[str, Any]] = {}
        for camera_id in C.ALL_CAMERAS:
            interval = wagon.cameras.get(camera_id) or {}
            camera = result.cameras[camera_id]
            camera_frame_ranges[camera_id] = {
                "start_frame": interval.get("start_frame"),
                "end_frame": interval.get("end_frame"),
                "start_time": _seconds(interval.get("start_frame"), camera.fps),
                "end_time": _seconds(interval.get("end_frame"), camera.fps),
                "status": interval.get("status", STATUS_UNMATCHED),
                "timeline_reversed": bool(interval.get("reversed", False)),
                "fps": camera.fps,
                "source": "global_wagon_app_alignment",
            }

        wagon_documents.append({
            "global_id": GLOBAL_ID_FORMAT % position,
            "wagon_index": position,
            "start_frame_master": wagon.global_start_frame_master,
            "end_frame_master": wagon.global_end_frame_master,
            "start_time": _seconds(wagon.global_start_frame_master, master_fps) or 0.0,
            "end_time": _seconds(wagon.global_end_frame_master, master_fps) or 0.0,
            "classification": wagon.classification,
            "classification_confidence": wagon.classification_confidence,
            "supporting_cameras": supporting,
            "split_from_global_id": None,
            # Which global gaps bound this wagon, on the 0-1000 timeline.
            "leading_gap": {
                "source": "global_gap",
                "global_gap_index": wagon.wagon_number - 1,
                "normalized_position": wagon.global_start_position,
            },
            "trailing_gap": {
                "source": "global_gap",
                "global_gap_index": wagon.wagon_number,
                "normalized_position": wagon.global_end_position,
            },
            # ---- additive: the aligned per-camera windows ----
            "camera_frame_ranges": camera_frame_ranges,
            "normalized_start_position": wagon.global_start_position,
            "normalized_end_position": wagon.global_end_position,
        })

    camera_offsets = {
        camera_id: _camera_offset_entry(
            result.cameras[camera_id], result.wagons, master_fps,
            is_master=(camera_id == result.master_camera))
        for camera_id in C.ALL_CAMERAS
    }

    support_alignment_summary = {
        camera_id: {
            "alignment_status": result.cameras[camera_id].alignment_status,
            "timeline_reversed": result.cameras[camera_id].is_reversed,
            "scale": result.cameras[camera_id].scale,
            "offset": result.cameras[camera_id].offset,
            "matched_gaps": result.cameras[camera_id].matched_gaps,
            "unique_gaps": result.cameras[camera_id].unique_gap_count,
            "detected_intervals": sum(
                1 for wagon in result.wagons
                if (wagon.cameras.get(camera_id) or {}).get("status")
                == STATUS_DETECTED),
            "recovered_intervals": sum(
                1 for wagon in result.wagons
                if (wagon.cameras.get(camera_id) or {}).get("status")
                == STATUS_RECOVERED),
            "unmatched_intervals": sum(
                1 for wagon in result.wagons
                if (wagon.cameras.get(camera_id) or {}).get("status")
                == STATUS_UNMATCHED),
        }
        for camera_id in C.ALL_CAMERAS
    }

    # The relationship the new engine guarantees, recorded so Stage 1 can log
    # it and any consumer can re-check it.
    expected_wagons = max(0, result.global_gap_count - 1)
    violations: List[str] = []
    if result.global_wagon_count != expected_wagons:
        violations.append(
            "global_wagon_count=%d but global_gap_count-1=%d"
            % (result.global_wagon_count, expected_wagons))
    if len(wagon_documents) != result.global_wagon_count:
        violations.append(
            "emitted %d wagon records for global_wagon_count=%d"
            % (len(wagon_documents), result.global_wagon_count))

    wagon_window = {
        "source": ENGINE_NAME,
        "master_camera": result.master_camera,
        "wagon_region_start_frame": master.crop_start_frame,
        "wagon_region_end_frame": master.crop_end_frame,
        "master_total_frames": master.total_frames,
        # Read back by GlobalTrainState.engine_count / brake_van_count.
        "leading_non_wagon_classes": dict(result.leading_non_wagon),
        "trailing_non_wagon_classes": dict(result.trailing_non_wagon),
        "interior_non_wagon_classes": {},
        "note": ("the engine trims to the confirmed wagon region before "
                 "counting, so locomotives and brake vans are outside the "
                 "wagon timeline by design and receive no GW id"),
    }

    return {
        "schema": STATE_SCHEMA,
        "global_counting_engine": ENGINE_NAME,
        "total_wagons": len(wagon_documents),
        "wagons": wagon_documents,
        "master_camera": result.master_camera,
        "master_fps": master_fps,
        "master_total_frames": master.total_frames,
        "per_camera_local_counts": {
            camera_id: max(0, result.cameras[camera_id].unique_gap_count - 1)
            for camera_id in C.ALL_CAMERAS
        },
        "per_camera_gap_counts": {
            camera_id: result.cameras[camera_id].unique_gap_count
            for camera_id in C.ALL_CAMERAS
        },
        "per_camera_status": {
            camera_id: result.cameras[camera_id].trim_status
            for camera_id in C.ALL_CAMERAS
        },
        "corrections_applied": [],
        "fallback_used": False,
        "fallback_reason": "",
        "notes": [
            "counted by %s (validated, frozen)" % ENGINE_NAME,
            "GLOBAL_WAGON_COUNT = GLOBAL_GAP_COUNT - 1",
            "master camera selected dynamically by max confirmed unique gaps",
        ],
        "fusion_mode": FUSION_MODE,
        "master_wagon_count": result.global_wagon_count,
        "global_gap_count": result.global_gap_count,
        "wagon_window": wagon_window,
        "camera_offsets": camera_offsets,
        "support_alignment_summary": support_alignment_summary,
        "invariant_checks": {
            "checks_run": 2,
            "invariant_holds": not violations,
            "violations": violations,
            "global_gap_count": result.global_gap_count,
            "global_wagon_count": result.global_wagon_count,
            "rule": "global_wagon_count == global_gap_count - 1",
        },
        # Additive provenance, ignored by the parser.
        "normalized_timeline_scale": result.normalized_scale,
        "engine_output_dir": result.engine_output_dir,
        "engine_csv_artifacts": dict(result.csv_paths),
        "engine_elapsed_seconds": round(result.elapsed_seconds, 3),
    }


def build_per_camera_tracking_document(result) -> Dict[str, Any]:
    """`{camera_id: {...}}`; `load_per_camera_fps` reads only `fps`."""
    document: Dict[str, Any] = {}
    for camera_id in C.ALL_CAMERAS:
        camera = result.cameras[camera_id]
        document[camera_id] = {
            "fps": camera.fps,
            "total_frames": camera.total_frames,
            "video_path": camera.video_path,
            "unique_gap_count": camera.unique_gap_count,
            "trim_status": camera.trim_status,
            "wagon_region_start_frame": camera.crop_start_frame,
            "wagon_region_end_frame": camera.crop_end_frame,
            "trimmed_total_frames": camera.trimmed_total_frames,
            "alignment_status": camera.alignment_status,
            "timeline_reversed": camera.is_reversed,
            "alignment_scale": camera.scale,
            "alignment_offset": camera.offset,
            "matched_gaps": camera.matched_gaps,
            "is_master": camera_id == result.master_camera,
        }
    return document


def write_documents(result, output_dir: str) -> Tuple[str, str]:
    """Write both Stage-1 files and return their paths."""
    os.makedirs(output_dir, exist_ok=True)
    state_path = os.path.join(output_dir, "global_train_state.json")
    tracking_path = os.path.join(output_dir, "per_camera_tracking.json")

    with open(state_path, "w", encoding="utf-8") as handle:
        json.dump(build_global_train_state_document(result), handle, indent=2)
    with open(tracking_path, "w", encoding="utf-8") as handle:
        json.dump(build_per_camera_tracking_document(result), handle, indent=2)
    return state_path, tracking_path
