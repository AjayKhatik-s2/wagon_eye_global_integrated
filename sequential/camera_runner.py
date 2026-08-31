"""One camera, ONE decode, camera-local evidence only.

    open the video ONCE
      every frame  -> gap detector          (engine, per-frame, stateless)
      every frame  -> classification batch  (engine model + engine class map)
      stride 3     -> door detector         (side cameras)
      stride 3     -> damage detector       (top cameras)
      stride 2     -> load classifier       (top cameras)
    close the video
      trimming boundaries  <- engine's own pure helpers on the is_wagon array
      unique gaps          <- replay the persisted detections through a fresh
                              engine GapTracker (NO second decode, NO re-inference)
      persist evidence -> camera report -> seal -> release

Why the replay works, and why it is not a second inference pass
--------------------------------------------------------------
The engine detects gaps on the TRIMMED clip, whose frame *k* is the original
frame `final_start + k` with identical pixels. `detect_gaps_in_frame` is
per-frame and stateless, so its output for a given frame does not depend on when
it was called. Only `GapTracker` is temporal. So this module runs the detector
once per original frame during the single decode, keeps the surviving
detections, and afterwards feeds the `[final_start .. final_end]` slice -- in
order, re-indexed to 0..N, with `timestamp = idx / fps` -- into a fresh
`GapTracker`. The tracker therefore sees exactly the sequence it sees in the
validated engine, and produces the same confirmed unique gaps.

The cost is that the gap detector also runs outside the wagon region, which the
engine avoids. That is the price of a single decode; it changes no result.

What this module deliberately does NOT do
-----------------------------------------
* It never assigns a canonical wagon id. Segments are `<CAMERA>_SEG_n` and are
  explicitly camera-local. Global meaning is created once, later, by
  `sequential.global_assembly`.
* It never aggregates a feature into a per-wagon verdict, because it does not
  know which wagon a frame belongs to. It stores RAW detections and lets the
  existing aggregators run in Global Assembly.
* It never modifies the frozen engine. Every engine call is a read.
"""

from __future__ import annotations

import gc
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from core import constants as C
from global_counting import runner as gc_runner

from sequential import evidence as ev

# Which cameras each feature actually inspects -- exactly the Batch mapping:
# door reads the two side cameras, damage and load read the two top cameras,
# and OCR reads RIGHT_UP only.
FEATURE_CAMERAS: Dict[str, Tuple[str, ...]] = {
    "door":   C.SIDE_CAMERAS,
    "damage": C.TOP_CAMERAS,
    "load":   C.TOP_CAMERAS,
    "ocr":    (C.CAMERA_RIGHT_UP,),
}

# Sampling strides. Production defaults, unchanged from Batch.
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
    decode_passes: int = 0
    unique_gap_count: int = 0
    observation_count: int = 0

    @property
    def sealed(self) -> bool:
        return self.status == ev.STATUS_SEALED


def features_for_camera(camera_id: str,
                        selected: Sequence[str]) -> Tuple[str, ...]:
    """The selected features that actually inspect this camera."""
    return tuple(name for name in selected
                 if camera_id in FEATURE_CAMERAS.get(name, ()))


def _model_fingerprints(camera_id: str, *, recon_models_dir: str,
                        feat_models_dir: str,
                        camera_features: Sequence[str]) -> Dict[str, Any]:
    """Fingerprints of every weight this camera's processing depends on."""
    out: Dict[str, Any] = {}
    counting = gc_runner.resolve_models(recon_models_dir)
    classification_slot = ("classification_top" if camera_id in C.TOP_CAMERAS
                           else "classification_side")
    gap_slot = {
        C.CAMERA_RIGHT_UP:     "gap_right",
        C.CAMERA_LEFT_UP:      "gap_left",
        C.CAMERA_RIGHT_UP_TOP: "gap_top",
        C.CAMERA_LEFT_UP_TOP:  "gap_top",
    }[camera_id]
    out[classification_slot] = ev.file_fingerprint(counting[classification_slot])
    out[gap_slot] = ev.file_fingerprint(counting[gap_slot])
    for name in camera_features:
        out["feature_%s" % name] = ev.file_fingerprint(
            os.path.join(feat_models_dir, FEATURE_MODEL_FILES[name]))
    return out


# -----------------------------------------------------------------------------
# The single decode
# -----------------------------------------------------------------------------

@dataclass
class _DecodeOutput:
    """Everything the one decode pass produced."""
    classification: List[Dict[str, Any]] = field(default_factory=list)
    # original frame index -> the engine's surviving gap detections
    gap_detections: Dict[int, List[Any]] = field(default_factory=dict)
    observations: List[ev.FeatureObservation] = field(default_factory=list)
    decoded_frames: int = 0
    fps: float = 0.0
    total_frames: int = 0
    width: int = 0
    height: int = 0
    raw_detection_count: int = 0


def _decode_once(
    *, camera_id: str, video_path: str, engine, feature_models: Dict[str, Any],
    strides: Dict[str, int], verbose: bool,
) -> _DecodeOutput:
    """Open the video ONCE and feed every consumer from the same frame."""
    import cv2

    from features._common import _predict_kwargs

    classification = engine["classification"]
    gap_detection = engine["gap_detection"]
    config = engine["config"]
    camera_map = engine["camera_map"]
    models = engine["models"]

    classification_key = camera_map.CAMERA_CLASSIFICATION_MODEL[
        gc_runner.CAMERA_ID_TO_KEY[camera_id]]
    gap_key = camera_map.CAMERA_GAP_MODEL[gc_runner.CAMERA_ID_TO_KEY[camera_id]]

    class_info = models.CLASSIFICATION_MODELS[classification_key]
    class_map = models.CLASSIFICATION_CLASS_MAPS[classification_key]
    gap_model = models.GAP_MODELS[gap_key]["model"]
    gap_class_names = models.GAP_CLASS_MAPS[gap_key]["raw"]
    gap_allowed_ids = models.GAP_CLASS_MAPS[gap_key]["gap_ids"]

    info = classification.inspect_video(video_path)
    out = _DecodeOutput(fps=float(info["fps"]),
                        total_frames=int(info["total_frames"] or 0),
                        width=int(info["width"]), height=int(info["height"]))
    fps = out.fps or 1.0

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise CameraRunError("cv2.VideoCapture could not open %s" % video_path)

    batch_frames: List[Any] = []
    batch_ids: List[int] = []
    frame_limit = int(getattr(config, "MAX_FRAMES_TO_PROCESS", 0) or 0)

    def _flush_classification() -> None:
        """Classify one batch.

        Mirrors the engine's own `_classify_batch` closure inside
        `classification.classify_video_frames`, which is not importable on its
        own. It uses the engine's model_info, the engine's class map and the
        engine's confidence threshold, so the per-frame verdict is the engine's
        -- only the frame SOURCE differs (our decode loop rather than a second
        VideoCapture). `tests/test_sequential_architecture.py` pins the record
        shape, and the opt-in parity test compares both paths on a real video.
        """
        if not batch_frames:
            return
        # `half` is deprecated in Ultralytics (-> `quantize`) and warns for
        # every call that mentions it, including half=False, which is the
        # library default. `_predict_kwargs` omits it unless fp16 is genuinely
        # requested, so precision is unchanged and no warning is emitted from
        # our code. See tests/test_multi_door_reporting.py for the provenance.
        predict_kwargs = dict(_predict_kwargs(bool(class_info["half"])),
                              imgsz=class_info["imgsz"],
                              device=engine["DEVICE_YOLO"])
        try:
            results = class_info["model"].predict(batch_frames, **predict_kwargs)
        except Exception:
            results = []
            for one in batch_frames:
                results.extend(
                    class_info["model"].predict(one, **predict_kwargs))
        threshold = float(config.CLASSIFICATION_CONFIDENCE_THRESHOLD)
        for frame_id, result in zip(batch_ids, results):
            probs = getattr(result, "probs", None)
            if probs is None:
                raise CameraRunError(
                    "the classification model returned no 'probs'; a YOLO "
                    "task='classify' checkpoint is required (got task=%r)"
                    % class_info.get("task"))
            class_id = int(probs.top1)
            confidence = float(probs.top1conf)
            is_wagon_class = bool(class_map["is_wagon"].get(class_id, False))
            out.classification.append({
                "frame_id": int(frame_id),
                "timestamp_seconds": round(float(frame_id) / fps, 4),
                "predicted_class": class_map["raw"].get(class_id, str(class_id)),
                "normalized_class": class_map["normalized"].get(class_id, ""),
                "confidence": round(confidence, 4),
                "is_wagon_class": is_wagon_class,
                "is_wagon": bool(is_wagon_class and confidence >= threshold),
            })
        batch_frames.clear()
        batch_ids.clear()

    frame_index = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok or frame is None:
                break

            timestamp = frame_index / fps

            # ---- classification: batched, every frame -------------------
            batch_frames.append(frame)
            batch_ids.append(frame_index)
            if len(batch_frames) >= int(config.BATCH_SIZE):
                _flush_classification()

            # ---- GAP: EVERY decoded frame ------------------------------
            raw, valid = gap_detection.detect_gaps_in_frame(
                gap_model, frame, gap_class_names, gap_allowed_ids)
            out.raw_detection_count += len(raw)
            if valid:
                out.gap_detections[frame_index] = list(valid)

            # ---- features: the SAME frame, at their own strides ---------
            for name, model in feature_models.items():
                stride = max(1, int(strides.get(name, 1)))
                if frame_index % stride:
                    continue
                out.observations.extend(_observe(
                    name, model, frame, frame_index, timestamp,
                    out.width, out.height))

            frame_index += 1
            if frame_limit and frame_index >= frame_limit:
                if verbose:
                    print("[SEQ/%s] MAX_FRAMES_TO_PROCESS=%d reached"
                          % (camera_id, frame_limit))
                break
    finally:
        _flush_classification()
        capture.release()               # the ONE capture, always closed

    out.decoded_frames = frame_index
    return out


def _observe(feature: str, model, frame, frame_index: int, timestamp: float,
             width: int, height: int) -> List[ev.FeatureObservation]:
    """Raw detections for one feature on one frame -- no aggregation.

    The RAW model class name is stored, not a canonical verdict, so the
    existing feature-specific mapping and aggregation run exactly once, later,
    in Global Assembly.
    """
    from core.frame_quality import detection_quality, snapshot_score
    from features._common import run_classification, run_detection

    if model is None:
        return []

    if feature == "load":
        # Load is a classifier: one observation per sampled frame.
        raw_class, confidence = run_classification(model, frame)
        if not raw_class:
            return []
        return [ev.FeatureObservation(
            feature="load", frame_idx=frame_index, timestamp=timestamp,
            state="", confidence=float(confidence), raw_class=str(raw_class))]

    detections = run_detection(model, frame, confidence=0.0)
    out: List[ev.FeatureObservation] = []
    for detection in detections:
        bbox = [float(v) for v in detection["bbox"]]
        quality = detection_quality(frame, bbox)
        out.append(ev.FeatureObservation(
            feature=feature, frame_idx=frame_index, timestamp=timestamp,
            state="", confidence=float(detection["confidence"]),
            bbox=bbox, raw_class=str(detection["class_name"]),
            score=float(snapshot_score(bbox, float(detection["confidence"]),
                                       quality, width, height)),
            extra={"quality": round(float(quality), 4)}))
    return out


# -----------------------------------------------------------------------------
# Post-decode: trimming + gap replay, both without touching the video
# -----------------------------------------------------------------------------

def _wagon_region(engine, classification: List[Dict[str, Any]],
                  ) -> Tuple[Optional[int], Optional[int], Dict[str, Any]]:
    """The confirmed wagon region, using the engine's OWN pure helpers.

    `find_reliable_wagon_start` / `find_reliable_wagon_end` operate on the
    is_wagon array, so the validated rolling-window confirmation, non-wagon
    tolerance and padding are reused rather than reimplemented.
    """
    import numpy as np

    trimming = engine["trimming"]
    config = engine["config"]

    if not classification:
        return None, None, {"reason": "no classified frames"}

    is_wagon = np.array([bool(r["is_wagon"]) for r in classification], dtype=bool)
    total = int(is_wagon.size)
    cumsum = np.concatenate(([0], np.cumsum(is_wagon.astype(np.int64))))

    start_info = trimming.find_reliable_wagon_start(
        is_wagon, cumsum, config.START_CONFIRMATION_WINDOW,
        config.START_MIN_WAGON_FRAMES)
    if start_info is None:
        return None, None, {"reason": "no confirmed wagon region",
                            "wagon_frames": int(is_wagon.sum()),
                            "total_frames": total}

    detected_start = int(start_info["detected_start_frame"])
    start_confirm = int(start_info["confirm_frame"])
    end_info = trimming.find_reliable_wagon_end(
        is_wagon, start_confirm, detected_start,
        config.END_CONFIRMATION_WINDOW, config.END_MIN_NON_WAGON_FRAMES,
        config.NON_WAGON_TOLERANCE_FRAMES)
    detected_end = int(end_info["detected_end_frame"])

    final_start = int(max(0, detected_start - config.START_PADDING_FRAMES))
    final_end = int(min(total - 1, detected_end + config.END_PADDING_FRAMES))
    if final_end < final_start:
        final_start, final_end = final_end, final_start

    diagnostics = {
        "detected_start_frame": detected_start,
        "detected_end_frame": detected_end,
        "start_confirm_frame": start_confirm,
        "end_confirm_frame": end_info.get("confirm_frame"),
        "end_confirmed": bool(end_info.get("end_confirmed")),
        "longest_tolerated_gap": int(end_info.get("longest_tolerated_gap", 0)),
        "wagon_frames": int(is_wagon.sum()),
        "total_frames": total,
    }
    return final_start, final_end, diagnostics


def _replay_gaps(engine, camera_id: str, decoded: _DecodeOutput,
                 final_start: int, final_end: int,
                 ) -> Tuple[List[ev.GapObservation], Dict[str, Any]]:
    """Feed the persisted detections through a fresh engine GapTracker.

    No video is opened and no model is called: the detections already exist.
    The tracker sees the `[final_start..final_end]` slice re-indexed to 0..N
    with `timestamp = idx / fps`, exactly as it does on the trimmed clip.
    """
    gap_tracking = engine["gap_tracking"]
    camera_pipeline = engine["camera_pipeline"]
    config = engine["config"]

    fps = decoded.fps or 1.0
    limit = int(getattr(config, "GAP_MAX_FRAMES_TO_PROCESS", 0) or 0)
    span = list(range(final_start, final_end + 1))
    if limit:
        span = span[:limit]

    tracker = gap_tracking.GapTracker()
    for trimmed_index, original_index in enumerate(span):
        tracker.update(decoded.gap_detections.get(original_index, []),
                       trimmed_index, trimmed_index / fps)
    tracker.finalize()

    trimmed_total = len(span)
    frame = camera_pipeline.build_normalized_gap_timeline(
        gc_runner.CAMERA_ID_TO_KEY[camera_id], tracker, trimmed_total, fps)

    gaps: List[ev.GapObservation] = []
    records = frame.to_dict("records") if hasattr(frame, "to_dict") else []
    for record in records:
        confirmation = int(record.get("confirmation_frame", 0) or 0)
        first = int(record.get("first_frame", confirmation) or confirmation)
        last = int(record.get("last_frame", confirmation) or confirmation)
        gaps.append(ev.GapObservation(
            local_gap_id=str(record.get("local_gap_id", "")),
            # back into ORIGINAL video numbering
            confirmation_frame=final_start + confirmation,
            first_frame=final_start + first,
            last_frame=final_start + last,
            normalized_position=float(
                record.get("normalized_confirmation_time", 0.0) or 0.0),
            max_confidence=float(record.get("max_confidence", 0.0) or 0.0),
            average_confidence=float(record.get("average_confidence", 0.0) or 0.0),
            frame_count=int(record.get("frame_count", 0) or 0),
        ))

    diagnostics = {
        "trimmed_total_frames": trimmed_total,
        "confirmed_unique_gap_count": int(tracker.confirmed_unique_gap_count),
        "frames_with_detections": sum(
            1 for index in span if decoded.gap_detections.get(index)),
        "raw_detections_all_frames": decoded.raw_detection_count,
    }
    return gaps, diagnostics


def _segments(camera_id: str, gaps: Sequence[ev.GapObservation],
              ) -> List[Dict[str, Any]]:
    """Camera-local segments between consecutive local gaps.

    Labelled `<CAMERA>_SEG_n`, never `GW_n`: this camera cannot know the
    canonical roster, and a single-camera report must not pretend otherwise.
    """
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
    """Process ONE camera end to end, then seal and release it."""
    started = time.time()
    camera_features = features_for_camera(camera_id, features)
    strides = {"door": int(door_stride), "damage": int(damage_stride),
               "load": int(load_stride), "ocr": 1}

    video_fingerprint = ev.file_fingerprint(video_path)
    model_fingerprints = _model_fingerprints(
        camera_id, recon_models_dir=recon_models_dir,
        feat_models_dir=feat_models_dir, camera_features=camera_features)
    config_digest = ev.config_fingerprint(
        features=camera_features, door_stride=door_stride,
        damage_stride=damage_stride, load_stride=load_stride)
    feature_config = {"features": list(camera_features),
                      "strides": {name: strides[name]
                                  for name in camera_features}}

    if verbose:
        print("[SEQ] Camera %s START" % camera_id)
        print("[SEQ/%s] features: %s" % (
            camera_id, ", ".join(camera_features) or "(none for this camera)"))

    # ---- resume ------------------------------------------------------
    decision = ev.evaluate_resume(
        workspace, camera_id, video_fingerprint=video_fingerprint,
        model_fingerprints=model_fingerprints, config_digest=config_digest)
    if decision.reuse and not force:
        if verbose:
            print("[SEQ/%s] RESUME: reusing sealed evidence (%s) -- no "
                  "inference re-run" % (camera_id, decision.reason))
            print("[SEQ] Camera %s SEALED (resumed)" % camera_id)
        seal = decision.seal or {}
        return CameraRunResult(
            camera_id=camera_id, status=ev.STATUS_SEALED, reused=True,
            reason=decision.reason,
            evidence_path=ev.evidence_path(workspace, camera_id),
            seal_path=ev.seal_path(workspace, camera_id),
            report_paths=dict(seal.get("reports") or {}),
            seconds=time.time() - started,
            decode_passes=0,
            unique_gap_count=int(seal.get("unique_gap_count", 0) or 0),
            observation_count=int(seal.get("observation_count", 0) or 0))
    if verbose and not decision.reuse:
        print("[SEQ/%s] REPROCESS: %s" % (camera_id, decision.reason))
    elif verbose and force:
        print("[SEQ/%s] REPROCESS: --force-cameras" % camera_id)

    resolved_engine = gc_runner.locate_engine(repo_root, engine_dir)
    counting_models = gc_runner.resolve_models(recon_models_dir)

    feature_models: Dict[str, Any] = {}
    decoded: Optional[_DecodeOutput] = None
    capture_released = True
    try:
        with gc_runner.engine_session(resolved_engine):
            import camera_map, camera_pipeline, classification, config
            import gap_detection, gap_tracking, models, trimming

            config.apply_overrides(GENERATE_TRIM_DEBUG_VIDEO=False,
                                   GENERATE_GAP_ANNOTATED_VIDEO=False)

            # Load ONLY this camera's two counting weights, so Sequential does
            # not hold all five resident at once.
            camera_key = gc_runner.CAMERA_ID_TO_KEY[camera_id]
            classification_key = camera_map.CAMERA_CLASSIFICATION_MODEL[camera_key]
            gap_key = camera_map.CAMERA_GAP_MODEL[camera_key]
            models.load_all_models(
                {classification_key: counting_models[
                    "classification_top" if camera_id in C.TOP_CAMERAS
                    else "classification_side"]},
                {gap_key: counting_models[{
                    C.CAMERA_RIGHT_UP: "gap_right",
                    C.CAMERA_LEFT_UP: "gap_left",
                    C.CAMERA_RIGHT_UP_TOP: "gap_top",
                    C.CAMERA_LEFT_UP_TOP: "gap_top"}[camera_id]]})
            models.build_class_maps()

            from features._common import load_yolo
            for name in camera_features:
                feature_models[name] = load_yolo(
                    os.path.join(feat_models_dir, FEATURE_MODEL_FILES[name]))

            engine = {
                "camera_map": camera_map, "camera_pipeline": camera_pipeline,
                "classification": classification, "config": config,
                "gap_detection": gap_detection, "gap_tracking": gap_tracking,
                "models": models, "trimming": trimming,
                "DEVICE_YOLO": __import__("runtime").DEVICE_YOLO,
            }

            capture_released = False
            decoded = _decode_once(
                camera_id=camera_id, video_path=video_path, engine=engine,
                feature_models=feature_models, strides=strides, verbose=verbose)
            capture_released = True

            final_start, final_end, trim_diagnostics = _wagon_region(
                engine, decoded.classification)

            if final_start is None:
                gaps: List[ev.GapObservation] = []
                gap_diagnostics = {"reason": trim_diagnostics.get("reason")}
                status = ev.STATUS_NO_REGION
                final_start = final_end = 0
            else:
                gaps, gap_diagnostics = _replay_gaps(
                    engine, camera_id, decoded, final_start, final_end)
                status = ev.STATUS_SEALED

            # Engine objects must not outlive the session.
            models.CLASSIFICATION_MODELS.clear()
            models.GAP_MODELS.clear()
            models.CLASSIFICATION_CLASS_MAPS.clear()
            models.GAP_CLASS_MAPS.clear()
    finally:
        _release(feature_models)

    timing = ev.CameraTiming(
        fps=decoded.fps, total_frames=decoded.total_frames,
        decoded_frames=decoded.decoded_frames,
        wagon_region_start_frame=final_start,
        wagon_region_end_frame=final_end,
        wagon_region_frames=max(0, final_end - final_start + 1),
        duration_seconds=round(decoded.decoded_frames / (decoded.fps or 1.0), 3))

    camera_evidence = ev.CameraEvidence(
        camera_id=camera_id, status=status, timing=timing, gaps=gaps,
        observations=decoded.observations,
        classification_timeline=decoded.classification,
        segments=_segments(camera_id, gaps),
        provenance={
            "engine_dir": resolved_engine,
            "video": video_fingerprint,
            "models": model_fingerprints,
            "config_fingerprint": config_digest,
            "batch_key": batch_key,
            "decode_passes": 1,
            "frame_width": decoded.width,
            "frame_height": decoded.height,
        },
        feature_config=feature_config,
        diagnostics={"trimming": trim_diagnostics, "gap": gap_diagnostics},
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
        unique_gap_count=len(gaps),
        observation_count=len(decoded.observations),
        notes=([] if status == ev.STATUS_SEALED
               else ["no confirmed wagon region on this camera"]))

    if verbose:
        print("[SEQ/%s] gaps=%d observations=%d frames=%d region=[%d..%d]"
              % (camera_id, len(gaps), len(decoded.observations),
                 decoded.decoded_frames, final_start, final_end))
        print("[SEQ] Camera %s %s" % (camera_id, status))

    return CameraRunResult(
        camera_id=camera_id, status=status, reused=False,
        reason=decision.reason, evidence_path=evidence_file,
        seal_path=seal_file, report_paths=report_paths,
        seconds=time.time() - started, decode_passes=1,
        unique_gap_count=len(gaps),
        observation_count=len(decoded.observations))


def _release(feature_models: Dict[str, Any]) -> None:
    """Drop this camera's models and inference caches before the next camera.

    Sequential only reduces peak usage if a camera's resources actually go away
    when it is done, so this runs in a `finally`.
    """
    feature_models.clear()
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
