"""One camera, ONE decode, camera-local evidence only.

    open the video ONCE
      every frame  -> gap detector          (engine, per-frame, stateless)
      every frame  -> classification_adapter (engine model, class map, threshold)
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

from sequential import classification_adapter, evidence as ev

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


# How this repository's model SLOT names map onto the keys the ENGINE's camera
# mapping uses. The engine keys classification models by "side"/"top" and gap
# models by "right"/"left"/"top" -- never by camera id -- and
# `camera_map.CAMERA_CLASSIFICATION_MODEL` / `CAMERA_GAP_MODEL` are the
# authority for which camera needs which. Sequential reads that authority at
# runtime rather than deriving a key from the camera id.
ENGINE_CLASSIFICATION_KEYS = {
    "classification_side": "side",
    "classification_top": "top",
}
ENGINE_GAP_KEYS = {
    "gap_right": "right",
    "gap_left": "left",
    "gap_top": "top",
}


class CameraRunError(RuntimeError):
    pass


def engine_model_registries(counting_models: Dict[str, str]):
    """The COMPLETE registries `load_all_models` requires, engine-keyed.

    `load_all_models` is all-or-nothing: it raises unless EVERY key named in
    `CAMERA_CLASSIFICATION_MODEL` and `CAMERA_GAP_MODEL` is present, no matter
    which camera is about to run. So Sequential passes the same five weights
    Batch passes, under the engine's own keys, converted to `pathlib.Path`
    because the engine calls `.stat()` on the values it is handed.

    Loading fewer would need a change to the frozen engine; the models are
    released again after each camera, which is where Sequential's resource
    saving actually comes from.
    """
    classification = {engine_key: counting_models[slot]
                      for slot, engine_key in ENGINE_CLASSIFICATION_KEYS.items()}
    gap = {engine_key: counting_models[slot]
           for slot, engine_key in ENGINE_GAP_KEYS.items()}
    return gc_runner._as_paths(classification), gc_runner._as_paths(gap)


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
    """Fingerprints of every weight this camera's processing depends on.

    `gc_runner.resolve_models` is the SAME resolver Batch/Stage-1 uses, applied
    to the SAME `--recon-models-dir` value. Sequential adds no search path of
    its own, so the two modes can never disagree about which weight file a slot
    means.

    Only the weights that affect THIS camera's observations are fingerprinted --
    its own classification and gap model, plus the feature models it runs. All
    five are LOADED (the engine's loader demands the complete set) but the other
    three are never used for this camera's inference, so including them would
    invalidate a perfectly good seal whenever an unrelated weight changed.
    """
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

    frame_limit = int(getattr(config, "MAX_FRAMES_TO_PROCESS", 0) or 0)

    # Classification goes through the ONE isolated adapter, which takes its
    # model_info, class map, threshold, batch size and device from the engine.
    # See sequential/classification_adapter.py for exactly what is mirrored and
    # for the contract test that detects engine drift.
    classifier = classification_adapter.from_engine(
        engine=engine, classification_key=classification_key, fps=fps,
        predict_kwargs_factory=_predict_kwargs)

    frame_index = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok or frame is None:
                break

            timestamp = frame_index / fps

            # ---- classification: every frame, batched by the adapter ----
            classifier.add(frame, frame_index)

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
        capture.release()               # the ONE capture, always closed

    out.classification = classifier.finish()
    out.decoded_frames = frame_index
    return out


def door_confidence_floor() -> float:
    """Batch's Door gate, read from Batch's own config object.

    `features/door/processor.py` computes
    `min_conf = float(tracker_config.closed_confidence_threshold)` and keeps
    only detections at or above it. Reading the same attribute here means there
    is ONE threshold, not a copy that can drift.
    """
    from features.door.processor import TrackerConfig
    return float(TrackerConfig().closed_confidence_threshold)


def damage_confidence_floor() -> float:
    """Batch's Damage gate: the default `confidence` of its `run()`."""
    return float(C.CONF_DAMAGE)


def _yolo_arrays(model, frame):
    """Raw YOLO output as the (boxes, confs, cls_ids, names) Batch filters on.

    Batch calls the model and then gates the ARRAYS, so Sequential does the
    same instead of going through a dict-returning helper -- that is what makes
    the surviving detection set identical.
    """
    import numpy as np

    try:
        result = model(frame, verbose=False)[0]
    except Exception:
        return None
    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return None
    return (boxes.xyxy.cpu().numpy(), boxes.conf.cpu().numpy(),
            boxes.cls.cpu().numpy().astype(int),
            getattr(model, "names", {}) or {})


def _observe(feature: str, model, frame, frame_index: int, timestamp: float,
             width: int, height: int) -> List[ev.FeatureObservation]:
    """Detections for one feature on one frame, gated EXACTLY as Batch gates.

    The gate is a detector-level validity filter (confidence floor, skip
    classes, area ratio, edge-zone suppression) -- a property of the frame, not
    of any wagon -- so it belongs here, where the frame is, and it must be the
    same filter Batch applies or the two modes could not agree. Door and Damage
    therefore call Batch's OWN threshold and Batch's OWN
    `_filter_detections_for_top`; nothing is reimplemented.

    What is NOT decided here is which wagon an observation belongs to, and what
    the wagon's verdict is. That is Global Assembly's job.
    """
    from core.frame_quality import detection_quality, snapshot_score

    if model is None:
        return []

    if feature == "load":
        # Load has no detection gate in Batch: every sampled frame votes.
        from features._common import run_classification
        raw_class, confidence = run_classification(model, frame)
        if not raw_class:
            return []
        return [ev.FeatureObservation(
            feature="load", frame_idx=frame_index, timestamp=timestamp,
            state="", confidence=float(confidence), raw_class=str(raw_class))]

    arrays = _yolo_arrays(model, frame)
    if arrays is None:
        return []
    boxes, confs, cls_ids, names = arrays

    if feature == "door":
        # features/door/processor.py: keep = confs >= min_conf
        floor = door_confidence_floor()
        keep = confs >= floor
        boxes, confs, cls_ids = boxes[keep], confs[keep], cls_ids[keep]
    elif feature == "damage":
        # features/damage/processor.py: the SAME pure filter, same floor.
        from features.damage.processor import _filter_detections_for_top
        boxes, confs, cls_ids = _filter_detections_for_top(
            boxes, confs, cls_ids, names, width, height,
            damage_confidence_floor())

    out: List[ev.FeatureObservation] = []
    for bbox, confidence, class_id in zip(boxes, confs, cls_ids):
        bbox_list = [float(bbox[0]), float(bbox[1]),
                     float(bbox[2]), float(bbox[3])]
        raw_class = str(names.get(int(class_id), "unknown")).lower()
        quality = detection_quality(frame, bbox_list)
        out.append(ev.FeatureObservation(
            feature=feature, frame_idx=frame_index, timestamp=timestamp,
            state="", confidence=float(confidence), bbox=bbox_list,
            raw_class=raw_class,
            score=float(snapshot_score(bbox_list, float(confidence), quality,
                                       width, height)),
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
            normalized_duration=float(
                record.get("normalized_duration", 0.0) or 0.0),
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
        damage_stride=damage_stride, load_stride=load_stride,
        extra={"gates": {
            "door": door_confidence_floor() if "door" in camera_features else None,
            "damage": (damage_confidence_floor()
                       if "damage" in camera_features else None)}})
    # The gates are part of the contract: they decide which detections become
    # evidence, so they are recorded and they take part in the config digest.
    gates = {}
    if "door" in camera_features:
        gates["door_confidence_floor"] = door_confidence_floor()
    if "damage" in camera_features:
        gates["damage_confidence_floor"] = damage_confidence_floor()
    feature_config = {"features": list(camera_features),
                      "strides": {name: strides[name]
                                  for name in camera_features},
                      "gates": gates}

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

            # This camera's own keys come from the ENGINE's mapping, not from
            # the camera id -- that mapping is the authority for which camera
            # needs side/top classification and which gap model it uses.
            camera_key = gc_runner.CAMERA_ID_TO_KEY[camera_id]
            classification_key = camera_map.CAMERA_CLASSIFICATION_MODEL[camera_key]
            gap_key = camera_map.CAMERA_GAP_MODEL[camera_key]

            # ...but load_all_models validates the WHOLE mapping, so it gets the
            # complete five-weight registries, exactly as Batch passes them.
            classification_paths, gap_paths = engine_model_registries(
                counting_models)
            models.load_all_models(classification_paths, gap_paths)
            models.build_class_maps()

            if verbose:
                print("[SEQ/%s] engine keys: classification=%r gap=%r"
                      % (camera_id, classification_key, gap_key))

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
