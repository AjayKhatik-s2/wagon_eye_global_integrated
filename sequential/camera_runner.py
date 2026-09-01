"""One camera, processed by the ENGINE'S OWN per-camera pipeline, then sealed.

    engine camera_pipeline.process_camera(camera)
        classification (engine)             -> per-frame timeline
        wagon-region trimming (engine)      -> the TRIMMED, RE-ENCODED clip
        gap detection ON THAT CLIP (engine) -> GapTracker -> unique gaps
        normalized gap timeline (engine)
    -> persist the result -> camera report -> seal -> release

WHY THIS IS THE ENGINE'S FUNCTION AND NOT OURS
Batch reaches its Stage-1 numbers by running exactly this function on exactly
this video. Anything else -- detecting on raw frames instead of the re-encoded
clip, reproducing the trimming, mirroring the classification batching --
changes the pixels or the frame set, and can therefore change the gap count,
the master camera and the wagon count. Batch is the golden reference, so
Sequential runs the same function and inherits its result.

The video is consequently decoded twice per camera, exactly as Batch decodes
it: once to classify, once for gap detection on the trimmed clip. That was
deliberately accepted -- exact parity outranks decode count.

WHAT IS PERSISTED, AND WHY IT IS LOSSLESS
The engine's whole global half reads only four fields per camera:

    normalized_timeline    -> ga.build_normalized_timelines
    trimmed_total_frames   -> ga.build_normalized_timelines,
                              ga.camera_frame_for_position
    trimmed_info["fps"]    -> ga.build_normalized_timelines
    status                 -> ga.build_normalized_timelines

and `gc_runner._harvest` additionally reads video_info, final_start_frame,
final_end_frame, unique_gap_count and the classification timeline. All of that
is plain data, so it is persisted verbatim and no engine object outlives the
session. No GapTracker is stored because nothing downstream reads one -- its
output IS the normalized timeline.

WHAT THIS STAGE DELIBERATELY DOES NOT DO
It runs NO feature inference. Batch infers Door/Damage/Load over each wagon's
stable interior of JPEG-90 cached frames, which cannot exist before the
canonical roster does. Running them here would use a different frame set and
different pixels, so they run in Global Assembly against the materialized
cache, through Batch's own processors.
"""

from __future__ import annotations

import gc
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from core import constants as C
from global_counting import runner as gc_runner

from sequential import evidence as ev

# This repository's model SLOT names -> the keys the ENGINE's camera mapping
# uses. `load_all_models` validates the WHOLE mapping, so the complete set is
# always supplied.
ENGINE_CLASSIFICATION_KEYS = {
    "classification_side": "side",
    "classification_top": "top",
}
ENGINE_GAP_KEYS = {
    "gap_right": "right",
    "gap_left": "left",
    "gap_top": "top",
}

# Recorded so the seal states the run's intent and resume invalidates on a
# stride change. The strides are APPLIED by Batch's processors in Global
# Assembly, which is the only place the wagon windows exist.
DEFAULT_STRIDES = {"door": 3, "damage": 3, "load": 2, "ocr": 1}

FEATURE_MODEL_FILES = {
    "door":   C.MODEL_DOOR_STATE,
    "damage": C.MODEL_DAMAGE,
    "load":   C.MODEL_LOADED,
    "ocr":    C.MODEL_WAGON_ID_COUNTING,
}


class CameraRunError(RuntimeError):
    pass


@dataclass
class CameraRunResult:
    camera_id: str
    status: str
    reused: bool = False
    reason: str = ""
    evidence_path: Optional[str] = None
    seal_path: Optional[str] = None
    report_paths: Dict[str, Optional[str]] = field(default_factory=dict)
    seconds: float = 0.0
    unique_gap_count: int = 0

    @property
    def sealed(self) -> bool:
        return self.status == ev.STATUS_SEALED


def engine_model_registries(counting_models: Dict[str, str]):
    """The COMPLETE registries `load_all_models` requires, engine-keyed.

    It raises unless every key in CAMERA_CLASSIFICATION_MODEL and
    CAMERA_GAP_MODEL is present, whichever camera is about to run, and calls
    `.stat()` on the values -- so all five weights are passed, as Path, using
    the same helper Batch uses.
    """
    classification = {engine_key: counting_models[slot]
                      for slot, engine_key in ENGINE_CLASSIFICATION_KEYS.items()}
    gap = {engine_key: counting_models[slot]
           for slot, engine_key in ENGINE_GAP_KEYS.items()}
    return gc_runner._as_paths(classification), gc_runner._as_paths(gap)


def _model_fingerprints(camera_id: str, *, recon_models_dir: str,
                        feat_models_dir: str,
                        features: Sequence[str]) -> Dict[str, Any]:
    """Fingerprints of every weight this camera's result can depend on.

    All five counting weights: the engine loads all five and `build_class_maps`
    reads every loaded model. Feature weights too, because the seal is reused
    for a run that will later apply those features in Global Assembly.
    """
    out: Dict[str, Any] = {}
    for slot, path in sorted(gc_runner.resolve_models(recon_models_dir).items()):
        out[slot] = ev.file_fingerprint(path)
    for name in features:
        out["feature_%s" % name] = ev.file_fingerprint(
            os.path.join(feat_models_dir, FEATURE_MODEL_FILES[name]))
    return out


# -----------------------------------------------------------------------------
# The engine's per-camera result -> a serialisable record
# -----------------------------------------------------------------------------

def _records(frame) -> List[Dict[str, Any]]:
    if frame is None:
        return []
    if hasattr(frame, "to_dict"):
        return frame.to_dict("records")
    return [dict(row) for row in frame]


def engine_record(result: Dict[str, Any]) -> Dict[str, Any]:
    """The lossless subset of the engine's per-camera result.

    Exactly the fields the engine's global half and `gc_runner._harvest` read,
    as plain JSON.
    """
    video_info = dict(result.get("video_info") or {})
    trimmed_info = dict(result.get("trimmed_info") or {})
    return {
        "status": result.get("status", "UNKNOWN"),
        "final_start_frame": int(result.get("final_start_frame", 0) or 0),
        "final_end_frame": int(result.get("final_end_frame", 0) or 0),
        "trimmed_total_frames": int(result.get("trimmed_total_frames", 0) or 0),
        "unique_gap_count": int(result.get("unique_gap_count", 0) or 0),
        "n_frames": int(result.get("n_frames", 0) or 0),
        "video_info": {
            "fps": float(video_info.get("fps", 0.0) or 0.0),
            "total_frames": int(video_info.get("total_frames", 0) or 0),
            "width": int(video_info.get("width", 0) or 0),
            "height": int(video_info.get("height", 0) or 0),
        },
        "trimmed_info": {
            "fps": float(trimmed_info.get("fps", 0.0) or 0.0),
            "total_frames": int(trimmed_info.get("total_frames", 0) or 0),
        },
        "trimmed_video_path": str(result.get("trimmed_video_path") or ""),
        "normalized_timeline": _records(result.get("normalized_timeline")),
        "classification_timeline": _records(result.get("timeline_df")),
    }


def _gaps_from_record(record: Dict[str, Any]) -> List[ev.GapObservation]:
    """Local gaps, with frames shifted into ORIGINAL video numbering.

    The engine numbers gap frames inside the TRIMMED clip; the camera report
    and every consumer outside the engine speak original frames.
    """
    crop = int(record.get("final_start_frame", 0) or 0)
    out: List[ev.GapObservation] = []
    for row in record.get("normalized_timeline") or []:
        confirmation = int(row.get("confirmation_frame", 0) or 0)
        out.append(ev.GapObservation(
            local_gap_id=str(row.get("local_gap_id", "")),
            confirmation_frame=crop + confirmation,
            first_frame=crop + int(row.get("first_seen_frame", confirmation)
                                   or confirmation),
            last_frame=crop + int(row.get("last_seen_frame", confirmation)
                                  or confirmation),
            normalized_position=float(
                row.get("normalized_confirmation_time", 0.0) or 0.0),
            max_confidence=float(row.get("max_confidence", 0.0) or 0.0),
            normalized_duration=float(
                row.get("normalized_duration", 0.0) or 0.0)))
    return out


def _segments(camera_id: str, gaps: Sequence[ev.GapObservation],
              ) -> List[Dict[str, Any]]:
    """Camera-local segments between consecutive local gaps. NOT canonical."""
    out: List[Dict[str, Any]] = []
    for index, (earlier, later) in enumerate(zip(gaps, gaps[1:]), start=1):
        out.append({
            "segment_id": ev.SEGMENT_ID_FORMAT % (camera_id, index),
            "segment_index": index,
            "start_frame": earlier.confirmation_frame,
            "end_frame": later.confirmation_frame,
            "start_normalized": earlier.normalized_position,
            "end_normalized": later.normalized_position,
            "opening_gap": earlier.local_gap_id,
            "closing_gap": later.local_gap_id,
            "canonical": False,
        })
    return out


# -----------------------------------------------------------------------------
# Public entry
# -----------------------------------------------------------------------------

def process_camera(
    *,
    camera_id: str,
    video_path: str,
    workspace: str,
    repo_root: str,
    recon_models_dir: str,
    feat_models_dir: str,
    features: Sequence[str],
    engine_dir: Optional[str] = None,
    door_stride: int = 3,
    damage_stride: int = 3,
    load_stride: int = 2,
    force: bool = False,
    verbose: bool = True,
    batch_key: str = "",
) -> CameraRunResult:
    """Run the ENGINE's per-camera pipeline for ONE camera, then seal it."""
    started = time.time()
    strides = {"door": int(door_stride), "damage": int(damage_stride),
               "load": int(load_stride), "ocr": 1}

    video_fingerprint = ev.file_fingerprint(video_path)
    model_fingerprints = _model_fingerprints(
        camera_id, recon_models_dir=recon_models_dir,
        feat_models_dir=feat_models_dir, features=features)
    config_digest = ev.config_fingerprint(
        features=features, door_stride=door_stride,
        damage_stride=damage_stride, load_stride=load_stride)
    feature_config = {"features": list(features),
                      "strides": {name: strides[name] for name in features},
                      "applied_in": "global_assembly"}

    if verbose:
        print("[SEQ] Camera %s START" % camera_id)

    decision = ev.evaluate_resume(
        workspace, camera_id, video_fingerprint=video_fingerprint,
        model_fingerprints=model_fingerprints, config_digest=config_digest)
    if decision.reuse and not force:
        if verbose:
            print("[SEQ/%s] RESUME: reusing sealed evidence (%s)"
                  % (camera_id, decision.reason))
            print("[SEQ] Camera %s SEALED (resumed)" % camera_id)
        seal = decision.seal or {}
        return CameraRunResult(
            camera_id=camera_id, status=ev.STATUS_SEALED, reused=True,
            reason=decision.reason,
            evidence_path=ev.evidence_path(workspace, camera_id),
            seal_path=ev.seal_path(workspace, camera_id),
            report_paths=dict(seal.get("reports") or {}),
            seconds=time.time() - started,
            unique_gap_count=int(seal.get("unique_gap_count", 0) or 0))
    if verbose:
        print("[SEQ/%s] %s" % (
            camera_id, "REPROCESS: --force-cameras" if force
            else "REPROCESS: %s" % decision.reason))

    resolved_engine = gc_runner.locate_engine(repo_root, engine_dir)
    counting_models = gc_runner.resolve_models(recon_models_dir)
    engine_output_dir = os.path.join(workspace, "global_state",
                                     "global_counting")
    os.makedirs(engine_output_dir, exist_ok=True)

    camera_key = gc_runner.CAMERA_ID_TO_KEY[camera_id]
    record: Dict[str, Any] = {}
    try:
        with gc_runner.engine_session(resolved_engine):
            import camera_pipeline
            import config
            import io_paths
            import models

            # The ONLY configuration Batch changes, so Sequential changes the
            # same two and nothing else.
            config.apply_overrides(GENERATE_TRIM_DEBUG_VIDEO=False,
                                   GENERATE_GAP_ANNOTATED_VIDEO=False)

            models.load_all_models(*engine_model_registries(counting_models))
            models.build_class_maps()

            # State population, not a substitute for resolve_inputs: only this
            # camera's video is needed, and requiring all four here would
            # defeat camera independence.
            io_paths.prepare_output_dirs(engine_output_dir)
            io_paths.VIDEO_PATHS[camera_key] = Path(video_path)

            # THE engine's own per-camera pipeline -- identical to what Batch
            # runs for this camera, on the same video, with the same weights.
            result = camera_pipeline.process_camera(camera_key, force=True)
            record = engine_record(result)

            camera_pipeline.CAMERA_RESULTS.clear()
            models.CLASSIFICATION_MODELS.clear()
            models.GAP_MODELS.clear()
            models.CLASSIFICATION_CLASS_MAPS.clear()
            models.GAP_CLASS_MAPS.clear()
    finally:
        _release()

    status = (ev.STATUS_NO_REGION if record.get("status") == "NO_REGION"
              else ev.STATUS_SEALED)
    video_info = record["video_info"]
    timing = ev.CameraTiming(
        fps=video_info["fps"], total_frames=video_info["total_frames"],
        decoded_frames=record["n_frames"],
        wagon_region_start_frame=record["final_start_frame"],
        wagon_region_end_frame=record["final_end_frame"],
        wagon_region_frames=record["trimmed_total_frames"],
        duration_seconds=round(record["n_frames"] / (video_info["fps"] or 1.0),
                               3))
    gaps = _gaps_from_record(record)

    camera_evidence = ev.CameraEvidence(
        camera_id=camera_id, status=status, timing=timing, gaps=gaps,
        observations=[],                 # features run after Global Assembly
        classification_timeline=record["classification_timeline"],
        segments=_segments(camera_id, gaps),
        provenance={
            "engine_dir": resolved_engine,
            "video": video_fingerprint,
            "models": model_fingerprints,
            "config_fingerprint": config_digest,
            "batch_key": batch_key,
            "frame_width": video_info["width"],
            "frame_height": video_info["height"],
            "produced_by": "engine camera_pipeline.process_camera",
        },
        feature_config=feature_config,
        diagnostics={"engine_status": record["status"],
                     "trimmed_video_path": record["trimmed_video_path"]},
        engine_result=record,
    )

    evidence_file = ev.write_evidence(workspace, camera_evidence)

    from sequential import camera_report
    report_paths = camera_report.build(
        workspace=workspace, evidence=camera_evidence, batch_key=batch_key,
        verbose=verbose)

    seal_file = ev.write_seal(
        workspace, camera_id=camera_id, status=status, timing=timing,
        video_fingerprint=video_fingerprint,
        model_fingerprints=model_fingerprints, config_digest=config_digest,
        feature_config=feature_config,
        processing_seconds=time.time() - started, report_paths=report_paths,
        unique_gap_count=len(gaps), observation_count=0,
        notes=([] if status == ev.STATUS_SEALED
               else ["no confirmed wagon region on this camera"]))

    if verbose:
        print("[SEQ/%s] unique gaps=%d  region=[%d..%d]  trimmed frames=%d"
              % (camera_id, len(gaps), record["final_start_frame"],
                 record["final_end_frame"], record["trimmed_total_frames"]))
        print("[SEQ] Camera %s %s" % (camera_id, status))

    return CameraRunResult(
        camera_id=camera_id, status=status, reused=False,
        reason=decision.reason, evidence_path=evidence_file,
        seal_path=seal_file, report_paths=report_paths,
        seconds=time.time() - started, unique_gap_count=len(gaps))


def _release() -> None:
    """Drop this camera's models and caches before the next camera starts."""
    try:
        from features._common import clear_yolo_cache
        clear_yolo_cache()
    except Exception:
        pass
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
