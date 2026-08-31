"""The mirrored classification logic, held to the engine by contract.

`sequential/classification_adapter.py` reproduces the per-frame verdict of the
engine's `classify_video_frames` because that function owns its own
`VideoCapture` and Sequential must decode each video exactly once. Mirroring is
a liability unless it is pinned, so this module pins it two ways:

1. **Equivalence** -- the REAL engine function and the adapter are run over the
   SAME frames and their records must be identical, field for field. The
   engine's `cv2` is swapped for a fake capture inside the test, so no video
   and no weights are needed; the classification model is a stub, so both sides
   see the same predictions.
2. **Drift** -- the engine's source must still contain the record keys and the
   threshold comparison the adapter mirrors. If the frozen engine ever changes
   its classification contract, this fails here rather than silently diverging
   in production.

    python -m pytest tests/test_classification_adapter_contract.py -q
"""

from __future__ import annotations

import inspect
import os
import sys

import numpy as np
import pytest

_TEST_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_TEST_DIR)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from global_counting import runner as gc_runner
from sequential import classification_adapter


def _engine_dir():
    try:
        return gc_runner.locate_engine(_REPO_ROOT)
    except gc_runner.GlobalCountingError:
        return None


ENGINE_DIR = _engine_dir()
requires_engine = pytest.mark.skipif(
    ENGINE_DIR is None,
    reason="global_wagon_app engine not installed (set GLOBAL_WAGON_APP_DIR)")

FRAME_COUNT = 37          # not a multiple of BATCH_SIZE, so the tail is tested
FPS = 15.0
WAGON_FROM, WAGON_TO = 8, 25
LOW_CONFIDENCE_FRAMES = {12, 13}      # wagon class but BELOW the threshold


# -----------------------------------------------------------------------------
# stubs shared by both sides
# -----------------------------------------------------------------------------

def _frames():
    out = []
    for index in range(FRAME_COUNT):
        frame = np.zeros((32, 32, 3), dtype=np.uint8)
        frame[0, 0, 0] = index
        out.append(frame)
    return out


class _Probs:
    def __init__(self, top1, conf):
        self.top1 = top1
        self.top1conf = conf


class _Result:
    def __init__(self, top1, conf):
        self.probs = _Probs(top1, conf)


class _StubClassifier:
    """Deterministic per-frame prediction, driven by the painted frame index."""

    def predict(self, frames, **kwargs):
        batch = frames if isinstance(frames, list) else [frames]
        out = []
        for frame in batch:
            index = int(frame[0, 0, 0])
            if WAGON_FROM <= index <= WAGON_TO:
                confidence = 0.30 if index in LOW_CONFIDENCE_FRAMES else 0.93
                out.append(_Result(1, confidence))
            else:
                out.append(_Result(0, 0.88))
        return out


MODEL_INFO = {"model": _StubClassifier(), "imgsz": 224, "half": False,
              "task": "classify", "path": "stub.pt"}
CLASS_MAP = {
    "raw": {0: "empty_track", 1: "wagon"},
    "normalized": {0: "empty_track", 1: "wagon"},
    "is_wagon": {0: False, 1: True},
    "wagon_ids": [1],
}


class _FakeCapture:
    """Yields the same frames the adapter is fed."""

    def __init__(self, _path):
        self._frames = _frames()
        self._index = 0

    def isOpened(self):
        return True

    def read(self):
        if self._index >= len(self._frames):
            return False, None
        frame = self._frames[self._index]
        self._index += 1
        return True, frame

    def release(self):
        pass


# -----------------------------------------------------------------------------
# 1. equivalence against the REAL engine function
# -----------------------------------------------------------------------------

@requires_engine
def test_adapter_matches_the_engine_record_for_record(monkeypatch):
    with gc_runner.engine_session(ENGINE_DIR):
        import classification
        import config

        # Give the engine our frames without touching its source: swap the cv2
        # it looks up, and neutralise the progress bar.
        monkeypatch.setattr(classification, "cv2",
                            type("cv2", (), {"VideoCapture": _FakeCapture}))
        monkeypatch.setattr(classification, "tqdm",
                            lambda *a, **kw: type(
                                "bar", (), {"update": lambda *_: None,
                                            "close": lambda *_: None,
                                            "set_postfix": lambda *_, **__: None,
                                            "total": None})())

        video_info = {"fps": FPS, "total_frames": FRAME_COUNT,
                      "width": 32, "height": 32}
        returned = classification.classify_video_frames(
            MODEL_INFO, CLASS_MAP, "stub.mp4", video_info, desc="test")
        # The engine returns (timeline_df, elapsed_seconds).
        timeline = returned[0] if isinstance(returned, tuple) else returned
        engine_records = (timeline.to_dict("records")
                          if hasattr(timeline, "to_dict") else list(timeline))

        threshold = float(config.CLASSIFICATION_CONFIDENCE_THRESHOLD)
        batch_size = int(config.BATCH_SIZE)

    adapter = classification_adapter.ClassificationAdapter(
        model_info=MODEL_INFO, class_map=CLASS_MAP, threshold=threshold,
        fps=FPS, device="cpu", batch_size=batch_size)
    for index, frame in enumerate(_frames()):
        adapter.add(frame, index)
    adapter_records = adapter.finish()

    assert len(adapter_records) == len(engine_records) == FRAME_COUNT
    for key in classification_adapter.RECORD_KEYS:
        engine_column = [record[key] for record in engine_records]
        adapter_column = [record[key] for record in adapter_records]
        assert adapter_column == engine_column, (
            "field %r diverged from the engine" % key)


@requires_engine
def test_the_threshold_rule_is_the_engine_s(monkeypatch):
    """The low-confidence wagon frames must be is_wagon=False on BOTH sides."""
    with gc_runner.engine_session(ENGINE_DIR):
        import config
        threshold = float(config.CLASSIFICATION_CONFIDENCE_THRESHOLD)
        batch_size = int(config.BATCH_SIZE)

    adapter = classification_adapter.ClassificationAdapter(
        model_info=MODEL_INFO, class_map=CLASS_MAP, threshold=threshold,
        fps=FPS, device="cpu", batch_size=batch_size)
    for index, frame in enumerate(_frames()):
        adapter.add(frame, index)
    records = {record["frame_id"]: record for record in adapter.finish()}

    for index in LOW_CONFIDENCE_FRAMES:
        assert records[index]["is_wagon_class"] is True
        assert records[index]["is_wagon"] is False, (
            "a wagon-class frame below the threshold must not count as WAGON")
    assert records[WAGON_FROM]["is_wagon"] is True
    assert records[0]["is_wagon"] is False


# -----------------------------------------------------------------------------
# 2. drift detection against the engine's source
# -----------------------------------------------------------------------------

@requires_engine
def test_engine_still_emits_the_fields_the_adapter_mirrors():
    with gc_runner.engine_session(ENGINE_DIR):
        import classification
        source = inspect.getsource(classification.classify_video_frames)

    for key in classification_adapter.RECORD_KEYS:
        assert '"%s"' % key in source, (
            "the engine no longer emits %r; the adapter must be updated" % key)
    assert "probs.top1" in source
    assert "CLASSIFICATION_CONFIDENCE_THRESHOLD" in source
    assert "is_wagon_class and conf >= CLASSIFICATION_CONFIDENCE_THRESHOLD" in source, (
        "the engine's WAGON rule changed shape; re-check the adapter")


@requires_engine
def test_engine_classification_still_owns_its_own_capture():
    """The reason the adapter exists. If this ever stops being true, delete it."""
    with gc_runner.engine_session(ENGINE_DIR):
        import classification
        source = inspect.getsource(classification.classify_video_frames)
        parameters = list(inspect.signature(
            classification.classify_video_frames).parameters)

    assert "cv2.VideoCapture" in source, (
        "the engine no longer decodes internally -- the mirrored adapter can "
        "probably be replaced by a direct call now")
    assert "video_path" in parameters
    assert not any(name in parameters for name in ("frames", "frame_iter")), (
        "the engine gained a frame-based entry point; route Sequential through "
        "it and delete sequential/classification_adapter.py")


# -----------------------------------------------------------------------------
# 3. the adapter's own behaviour (engine not required)
# -----------------------------------------------------------------------------

def test_adapter_batches_and_flushes_the_tail():
    seen_batches = []

    class _Counting(_StubClassifier):
        def predict(self, frames, **kwargs):
            batch = frames if isinstance(frames, list) else [frames]
            seen_batches.append(len(batch))
            return super().predict(frames, **kwargs)

    adapter = classification_adapter.ClassificationAdapter(
        model_info=dict(MODEL_INFO, model=_Counting()), class_map=CLASS_MAP,
        threshold=0.5, fps=FPS, device="cpu", batch_size=8)
    for index, frame in enumerate(_frames()):
        adapter.add(frame, index)
    records = adapter.finish()

    assert len(records) == FRAME_COUNT
    assert sum(seen_batches) == FRAME_COUNT
    assert seen_batches[:4] == [8, 8, 8, 8]
    assert seen_batches[-1] == FRAME_COUNT % 8, "the tail batch must be flushed"
    assert [r["frame_id"] for r in records] == list(range(FRAME_COUNT))


def test_adapter_falls_back_to_per_frame_on_batch_failure():
    """One bad frame must not lose the whole camera, as in the engine."""
    class _FlakyBatch(_StubClassifier):
        def predict(self, frames, **kwargs):
            if isinstance(frames, list) and len(frames) > 1:
                raise RuntimeError("batch inference blew up")
            return super().predict(frames, **kwargs)

    adapter = classification_adapter.ClassificationAdapter(
        model_info=dict(MODEL_INFO, model=_FlakyBatch()), class_map=CLASS_MAP,
        threshold=0.5, fps=FPS, device="cpu", batch_size=8)
    for index, frame in enumerate(_frames()):
        adapter.add(frame, index)
    records = adapter.finish()
    assert len(records) == FRAME_COUNT


def test_adapter_rejects_a_detection_model():
    class _NoProbs:
        def predict(self, frames, **kwargs):
            batch = frames if isinstance(frames, list) else [frames]
            return [type("r", (), {"probs": None})() for _ in batch]

    adapter = classification_adapter.ClassificationAdapter(
        model_info={"model": _NoProbs(), "imgsz": 224, "half": False,
                    "task": "detect"},
        class_map=CLASS_MAP, threshold=0.5, fps=FPS, device="cpu",
        batch_size=4)
    adapter.add(_frames()[0], 0)
    with pytest.raises(RuntimeError, match="classify"):
        adapter.finish()


def test_adapter_omits_the_deprecated_half_argument():
    """fp32 must be requested by OMITTING `half`, as everywhere else."""
    from features._common import _predict_kwargs

    seen = {}

    class _Recording(_StubClassifier):
        def predict(self, frames, **kwargs):
            seen.update(kwargs)
            return super().predict(frames, **kwargs)

    adapter = classification_adapter.ClassificationAdapter(
        model_info=dict(MODEL_INFO, model=_Recording()), class_map=CLASS_MAP,
        threshold=0.5, fps=FPS, device="cpu", batch_size=2,
        predict_kwargs_factory=_predict_kwargs)
    adapter.add(_frames()[0], 0)
    adapter.add(_frames()[1], 1)
    adapter.finish()

    assert "half" not in seen, "fp32 must be obtained by omitting `half`"
    assert seen["verbose"] is False
    assert seen["imgsz"] == 224

    # ...and a genuine fp16 request is still passed through.
    seen.clear()
    fp16 = classification_adapter.ClassificationAdapter(
        model_info=dict(MODEL_INFO, model=_Recording(), half=True),
        class_map=CLASS_MAP, threshold=0.5, fps=FPS, device="cpu",
        batch_size=1, predict_kwargs_factory=_predict_kwargs)
    fp16.add(_frames()[0], 0)
    fp16.finish()
    assert seen.get("half") is True


def test_camera_runner_holds_no_classification_logic():
    """The mirror must live in ONE place only."""
    import re

    with open(os.path.join(_REPO_ROOT, "sequential", "camera_runner.py"),
              encoding="utf-8") as handle:
        source = re.sub(r'"""(?:.|\n)*?"""', "", handle.read())
    for banned in ("probs.top1", "CLASSIFICATION_CONFIDENCE_THRESHOLD",
                   "is_wagon_class", "top1conf"):
        assert banned not in source, (
            "classification logic leaked back into camera_runner: %r" % banned)
    assert "classification_adapter" in source
