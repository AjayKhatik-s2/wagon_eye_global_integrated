"""The ONE place engine classification logic is mirrored, and why.

WHY THIS EXISTS
---------------
The engine's only classification entry point is

    classification.classify_video_frames(model_info, class_map, video_path,
                                         video_info, desc="Classifying")

It takes a **video path** and owns its own `cv2.VideoCapture`. Sequential's
defining invariant is ONE decode per camera feeding GAP, Door, Damage and Load
from the same frames, so it cannot hand the engine a path without decoding the
video a second time. The two requirements are mutually exclusive with the
engine's current API, and `global_count_ec2` is frozen, so the per-frame
verdict is reproduced here instead.

WHAT IS MIRRORED -- exactly this, and nothing else
--------------------------------------------------
The record built per frame, and the WAGON rule:

    cls_id  = int(probs.top1)
    conf    = float(probs.top1conf)
    is_wagon = class_map["is_wagon"][cls_id] and conf >= threshold

    {"frame_id", "timestamp_seconds", "predicted_class", "normalized_class",
     "confidence", "is_wagon_class", "is_wagon"}

WHAT IS NOT MIRRORED -- it is REUSED
------------------------------------
* the model, its imgsz and its half flag           -> engine `model_info`
* the raw / normalized / is_wagon class maps        -> engine `class_map`
* CLASSIFICATION_CONFIDENCE_THRESHOLD              -> engine `config`
* BATCH_SIZE                                       -> engine `config`
* DEVICE_YOLO                                       -> engine `runtime`
* the batch-then-per-frame fallback on exception    -> same structure

So every threshold, every class name and every device decision is the engine's.
Only the frame SOURCE differs.

DRIFT PROTECTION
----------------
`tests/test_classification_adapter_contract.py`:

1. runs the REAL `classify_video_frames` and this adapter over the SAME frames
   and asserts the records are identical, and
2. asserts the engine's source still contains the record keys and the threshold
   comparison this adapter mirrors,

so a change inside the frozen engine fails a test here instead of silently
diverging.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

# The record keys the engine emits per frame. Pinned so a drift test can
# compare against the engine's own source.
RECORD_KEYS = ("frame_id", "timestamp_seconds", "predicted_class",
               "normalized_class", "confidence", "is_wagon_class", "is_wagon")


class ClassificationAdapter:
    """Incremental, frame-fed equivalent of `classify_video_frames`.

    Feed frames as they are decoded; `finish()` returns the timeline in frame
    order, identical to what the engine would have produced for those frames.
    """

    def __init__(self, *, model_info: Dict[str, Any], class_map: Dict[str, Any],
                 threshold: float, fps: float, device: Any, batch_size: int,
                 predict_kwargs_factory=None) -> None:
        self._model_info = model_info
        self._class_map = class_map
        self._threshold = float(threshold)
        self._fps = float(fps) or 1.0
        self._device = device
        self._batch_size = max(1, int(batch_size))
        self._records: List[Dict[str, Any]] = []
        self._frames: List[Any] = []
        self._ids: List[int] = []
        # Injected so the deprecated `half` argument is omitted unless fp16 is
        # genuinely requested (see features/_common._predict_kwargs).
        self._predict_kwargs_factory = predict_kwargs_factory

    # ------------------------------------------------------------------
    def add(self, frame: Any, frame_id: int) -> None:
        self._frames.append(frame)
        self._ids.append(int(frame_id))
        if len(self._frames) >= self._batch_size:
            self.flush()

    def flush(self) -> None:
        if not self._frames:
            return
        model = self._model_info["model"]
        kwargs = self._base_kwargs()
        try:
            results = model.predict(self._frames, **kwargs)
        except Exception:
            # Same fallback the engine uses: one problematic frame must not
            # abort the whole camera.
            results = []
            for one in self._frames:
                results.extend(model.predict(one, **kwargs))

        for frame_id, result in zip(self._ids, results):
            self._records.append(self._record(frame_id, result))
        self._frames = []
        self._ids = []

    def finish(self) -> List[Dict[str, Any]]:
        self.flush()
        return self._records

    # ------------------------------------------------------------------
    def _base_kwargs(self) -> Dict[str, Any]:
        half = bool(self._model_info.get("half"))
        if self._predict_kwargs_factory is not None:
            kwargs = dict(self._predict_kwargs_factory(half))
        else:
            kwargs = {"verbose": False}
            if half:
                kwargs["half"] = True
        kwargs["imgsz"] = self._model_info["imgsz"]
        kwargs["device"] = self._device
        return kwargs

    def _record(self, frame_id: int, result: Any) -> Dict[str, Any]:
        probs = getattr(result, "probs", None)
        if probs is None:
            raise RuntimeError(
                "the classification model returned no 'probs'; an Ultralytics "
                "YOLO task='classify' checkpoint is required (got task=%r)"
                % self._model_info.get("task"))
        class_id = int(probs.top1)
        confidence = float(probs.top1conf)
        is_wagon_class = bool(self._class_map["is_wagon"].get(class_id, False))
        return {
            "frame_id": int(frame_id),
            "timestamp_seconds": round(float(frame_id) / self._fps, 4),
            "predicted_class": self._class_map["raw"].get(class_id,
                                                          str(class_id)),
            "normalized_class": self._class_map["normalized"].get(class_id, ""),
            "confidence": round(confidence, 4),
            "is_wagon_class": is_wagon_class,
            "is_wagon": bool(is_wagon_class and confidence >= self._threshold),
        }


def from_engine(*, engine: Dict[str, Any], classification_key: str,
                fps: float, predict_kwargs_factory=None,
                ) -> ClassificationAdapter:
    """Build the adapter from live engine modules, taking every value from them."""
    models = engine["models"]
    config = engine["config"]
    return ClassificationAdapter(
        model_info=models.CLASSIFICATION_MODELS[classification_key],
        class_map=models.CLASSIFICATION_CLASS_MAPS[classification_key],
        threshold=float(config.CLASSIFICATION_CONFIDENCE_THRESHOLD),
        fps=fps,
        device=engine["DEVICE_YOLO"],
        batch_size=int(config.BATCH_SIZE),
        predict_kwargs_factory=predict_kwargs_factory,
    )
