"""The dual-mode architecture, proved rather than asserted in a comment.

Batch and Sequential are both first-class. Batch must keep behaving exactly as
it does today; Sequential must be genuinely camera-independent, with ONE decode
per camera, evidence persisted before sealing, resources released afterwards,
and exactly one canonical global roster created later by Global Assembly.

Everything here runs without weights, videos or the engine: the engine is
replaced by a stub module set loaded through the REAL `engine_session`, and
`cv2.VideoCapture` is replaced by a fake that yields synthetic frames. That
lets the tests count decodes, detector calls and stride hits exactly.

    python -m pytest tests/test_sequential_architecture.py -q
"""

from __future__ import annotations

import json
import os
import re
import sys
import types

import numpy as np
import pytest

_TEST_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_TEST_DIR)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from core import constants as C
from global_counting import runner as gc_runner
from orchestrator import master_runner as mr
from sequential import camera_runner, evidence as ev, global_assembly
from sequential import runner as sequential_runner


# =============================================================================
# Stub engine + stub video
# =============================================================================

FRAME_COUNT = 60
FPS = 15.0
WIDTH, HEIGHT = 320, 240

# Frames 10..49 classify as WAGON, so the confirmed region is inside the clip.
WAGON_FROM, WAGON_TO = 10, 49

STUB_ENGINE = {
"global_wagon_pipeline.py": "def run(args):\n    return 0\n",

"camera_map.py": '''
CAMERAS = ["right_up", "left_up", "right_up_top", "left_up_top"]
CAMERA_LABELS = {c: c.upper() for c in CAMERAS}
CAMERA_CLASSIFICATION_MODEL = {"right_up": "side", "left_up": "side",
                               "right_up_top": "top", "left_up_top": "top"}
CAMERA_GAP_MODEL = {"right_up": "right", "left_up": "left",
                    "right_up_top": "top", "left_up_top": "top"}
CLASSIFICATION_MODEL_FILENAMES = {"side": ["side_classification.pt"],
                                  "top": ["top_classification.pt"]}
GAP_MODEL_FILENAMES = {"right": ["right_up_wagon_gap.pt"],
                       "left": ["left_up_wagon_gap.pt"],
                       "top": ["top_gap.pt"]}
''',

"config.py": '''
BATCH_SIZE = 8
CLASSIFICATION_CONFIDENCE_THRESHOLD = 0.5
START_CONFIRMATION_WINDOW = 5
START_MIN_WAGON_FRAMES = 4
END_CONFIRMATION_WINDOW = 5
END_MIN_NON_WAGON_FRAMES = 4
NON_WAGON_TOLERANCE_FRAMES = 3
START_PADDING_FRAMES = 2
END_PADDING_FRAMES = 2
MAX_FRAMES_TO_PROCESS = 0
GAP_MAX_FRAMES_TO_PROCESS = 0
NORMALIZED_TIMELINE_SCALE = 1000.0
GLOBAL_ALIGNMENT_TOLERANCE = 20.0
GENERATE_TRIM_DEBUG_VIDEO = True
GENERATE_GAP_ANNOTATED_VIDEO = True
_OVERRIDABLE = ("GENERATE_TRIM_DEBUG_VIDEO", "GENERATE_GAP_ANNOTATED_VIDEO")

def apply_overrides(**overrides):
    for name, value in overrides.items():
        if value is None:
            continue
        if name not in _OVERRIDABLE:
            raise KeyError(name)
        globals()[name] = value
    return dict(overrides)
''',

"runtime.py": 'DEVICE = "cpu"\nDEVICE_YOLO = "cpu"\nDEVICE_LABEL = "CPU"\nUSE_HALF = False\n',

"classification.py": '''
import os, json
def inspect_video(video_path):
    return {"fps": %(fps)s, "total_frames": %(frames)d,
            "width": %(width)d, "height": %(height)d, "fourcc": "h264"}
''' % {"fps": FPS, "frames": FRAME_COUNT, "width": WIDTH, "height": HEIGHT},

# Records every call so the tests can prove GAP saw EVERY decoded frame.
"gap_detection.py": '''
import json, os
CALLS = []

def detect_gaps_in_frame(model, frame, class_names, allowed_class_ids):
    CALLS.append(int(frame[0, 0, 0]))          # frame index is painted in
    path = os.environ.get("STUB_GAP_CALLS")
    if path:
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(CALLS, handle)
    index = int(frame[0, 0, 0])
    # A gap on three frames inside the wagon region -> three unique gaps.
    if index in (14, 28, 44):
        return ([{"i": index}], [{"i": index}])
    return ([], [])
''',

"gap_tracking.py": '''
class GapTracker(object):
    """Confirms one unique gap per distinct detection frame."""
    def __init__(self):
        self.seen = []
        self.finalized = False
        self.confirmed_unique_gap_count = 0

    def update(self, detections, frame_idx, timestamp):
        if detections:
            self.seen.append(int(frame_idx))
            self.confirmed_unique_gap_count = len(self.seen)
        return {"newly_confirmed": [], "matched": [], "active_tracks": 0}

    def finalize(self):
        self.finalized = True
''',

"camera_pipeline.py": '''
import pandas as pd
CAMERA_RESULTS = {}

def build_normalized_gap_timeline(camera, tracker, trimmed_total_frames, fps):
    denominator = float(max(1, int(trimmed_total_frames) - 1))
    rows = []
    for order, frame_idx in enumerate(tracker.seen, start=1):
        rows.append({
            "local_gap_id": "%s_G%d" % (camera, order),
            "confirmation_frame": int(frame_idx),
            "first_frame": int(frame_idx),
            "last_frame": int(frame_idx),
            "normalized_confirmation_time": (frame_idx / denominator) * 1000.0,
            "max_confidence": 0.9,
            "average_confidence": 0.85,
            "frame_count": 1,
        })
    return pd.DataFrame(rows)
''',

"trimming.py": '''
def find_reliable_wagon_start(is_wagon, cumsum, window, min_wagon_frames):
    n = len(is_wagon)
    for end in range(window - 1, n):
        start = end - window + 1
        if int(cumsum[end + 1] - cumsum[start]) >= min_wagon_frames:
            first = next(i for i in range(start, end + 1) if is_wagon[i])
            return {"detected_start_frame": first, "confirm_frame": end,
                    "window_start_frame": start,
                    "wagon_in_window": int(cumsum[end + 1] - cumsum[start])}
    return None

def find_reliable_wagon_end(is_wagon, start_confirm_frame, detected_start_frame,
                            end_window, min_non_wagon_frames, tolerance):
    n = len(is_wagon)
    last_wagon = int(detected_start_frame)
    for i in range(int(start_confirm_frame), n):
        if is_wagon[i]:
            last_wagon = i
            continue
        window = is_wagon[max(0, i - end_window + 1):i + 1]
        if (len(window) - sum(bool(v) for v in window)) >= min_non_wagon_frames:
            return {"detected_end_frame": last_wagon, "confirm_frame": i,
                    "end_candidate_start": i, "end_confirmed": True,
                    "longest_tolerated_gap": 0}
    return {"detected_end_frame": last_wagon, "confirm_frame": n - 1,
            "end_candidate_start": None, "end_confirmed": False,
            "longest_tolerated_gap": 0}
''',

# Deliberately named like ours, to keep exercising engine_session isolation.
"models.py": '''
CLASSIFICATION_MODELS, GAP_MODELS = {}, {}
CLASSIFICATION_CLASS_MAPS, GAP_CLASS_MAPS = {}, {}

class _Probs(object):
    def __init__(self, top1, conf):
        self.top1 = top1
        self.top1conf = conf

class _Result(object):
    def __init__(self, top1, conf):
        self.probs = _Probs(top1, conf)

class _ClassModel(object):
    """Class 1 == wagon; frames %d..%d are wagon frames."""
    def predict(self, frames, **kwargs):
        batch = frames if isinstance(frames, list) else [frames]
        out = []
        for frame in batch:
            index = int(frame[0, 0, 0])
            wagon = %d <= index <= %d
            out.append(_Result(1 if wagon else 0, 0.95))
        return out

def load_all_models(classification_paths, gap_paths):
    CLASSIFICATION_MODELS.clear(); GAP_MODELS.clear()
    for key, path in classification_paths.items():
        CLASSIFICATION_MODELS[key] = {"model": _ClassModel(), "imgsz": 224,
                                      "half": False, "task": "classify",
                                      "path": path}
    for key, path in gap_paths.items():
        GAP_MODELS[key] = {"model": object(), "path": path, "task": "detect",
                           "names": {0: "gap"}}
    return CLASSIFICATION_MODELS, GAP_MODELS

def build_class_maps():
    CLASSIFICATION_CLASS_MAPS.clear(); GAP_CLASS_MAPS.clear()
    for key in CLASSIFICATION_MODELS:
        CLASSIFICATION_CLASS_MAPS[key] = {
            "raw": {0: "empty_track", 1: "wagon"},
            "normalized": {0: "empty_track", 1: "wagon"},
            "is_wagon": {0: False, 1: True},
            "wagon_ids": [1]}
    for key in GAP_MODELS:
        GAP_CLASS_MAPS[key] = {"raw": {0: "gap"}, "gap_ids": [0]}
''' % (WAGON_FROM, WAGON_TO, WAGON_FROM, WAGON_TO),

"reporting.py": '''
def write_normalized_gap_timelines(path): open(str(path), "w").close()
def write_camera_alignment_summary(path): open(str(path), "w").close()
def write_global_gap_timeline(path): open(str(path), "w").close()
''',

"io_paths.py": "VIDEO_PATHS = {}\nOUTPUT_DIR = None\n",
"global_alignment.py": '''
def monotonic_gap_match(master_positions, camera_positions, tolerance,
                        master_durations=None, camera_durations=None,
                        duration_weight=None):
    matches, used = [], set()
    for i, m in enumerate(master_positions):
        best, best_error = None, None
        for j, c in enumerate(camera_positions):
            if j in used:
                continue
            error = abs(float(m) - float(c))
            if error <= tolerance and (best_error is None or error < best_error):
                best, best_error = j, error
        if best is not None:
            used.add(best)
            matches.append({"master_index": i, "camera_index": best,
                            "error": best_error, "score": 1.0})
    return matches

def robust_linear_fit(x, y):
    return (1.0, 0.0, len(x), "stub")
''',
"utils.py": "def banner(text):\n    pass\n",
"gap_annotation.py": "", "snapshot_extraction.py": "", "pdf_report.py": "",
"visualization.py": "", "wagon_mapping.py": "GLOBAL_WAGONS = []\nGLOBAL_WAGON_COUNT = 0\n",
}


class FakeCapture:
    """Counts how many times a video is opened, and paints the frame index in."""
    open_count = 0

    def __init__(self, path):
        FakeCapture.open_count += 1
        self._index = 0
        self.released = False

    def isOpened(self):
        return True

    def read(self):
        if self._index >= FRAME_COUNT:
            return False, None
        frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
        frame[0, 0, 0] = self._index            # the frame's own index
        self._index += 1
        return True, frame

    def release(self):
        self.released = True

    def get(self, _prop):
        return 0


@pytest.fixture
def stub_engine(tmp_path, monkeypatch):
    engine_dir = tmp_path / "global_wagon_app"
    engine_dir.mkdir()
    for name, source in STUB_ENGINE.items():
        (engine_dir / name).write_text(source, encoding="utf-8")
    monkeypatch.setenv("STUB_GAP_CALLS", str(tmp_path / "gap_calls.json"))
    return engine_dir


@pytest.fixture
def inputs(tmp_path):
    """Real files so fingerprints work; contents are irrelevant to the stubs."""
    videos = tmp_path / "videos"
    models = tmp_path / "models"
    videos.mkdir()
    models.mkdir()
    paths = {}
    for camera in C.ALL_CAMERAS:
        path = videos / ("%s.mp4" % camera)
        path.write_bytes(b"video")
        paths[camera] = str(path)
    for filenames in gc_runner.MODEL_SLOTS.values():
        (models / filenames[0]).write_bytes(b"w")
    for name in ("door_state.pt", "damage.pt", "loaded.pt",
                 "wagon_id_counting.pt"):
        (models / name).write_bytes(b"w")
    return paths, str(models)


@pytest.fixture
def wired(monkeypatch, stub_engine):
    """Fake cv2 capture + fake feature detectors that record their calls."""
    import cv2
    FakeCapture.open_count = 0
    monkeypatch.setattr(cv2, "VideoCapture", FakeCapture)

    calls = {"door": [], "damage": [], "load": []}

    def _fake_load_yolo(path):
        return types.SimpleNamespace(name=os.path.basename(path or ""))

    def _fake_run_detection(model, frame, confidence=0.4, half=False):
        index = int(frame[0, 0, 0])
        name = getattr(model, "name", "")
        feature = "door" if "door" in name else "damage"
        calls[feature].append(index)
        state = "open_door" if feature == "door" and index % 2 else (
            "closed_door" if feature == "door" else "dent")
        return [{"class_id": 0, "class_name": state, "confidence": 0.9,
                 "bbox": [10.0, 10.0, 60.0, 120.0]}]

    def _fake_run_classification(model, frame):
        calls["load"].append(int(frame[0, 0, 0]))
        return ("loaded", 0.9)

    from features import _common
    monkeypatch.setattr(_common, "load_yolo", _fake_load_yolo)
    monkeypatch.setattr(_common, "run_detection", _fake_run_detection)
    monkeypatch.setattr(_common, "run_classification", _fake_run_classification)
    return calls


def _process(camera_id, *, workspace, stub_engine, inputs, features=("door", "damage", "load"),
             **kwargs):
    video_paths, models_dir = inputs
    return camera_runner.process_camera(
        camera_id=camera_id, video_path=video_paths[camera_id],
        workspace=str(workspace), repo_root=_REPO_ROOT,
        recon_models_dir=models_dir, feat_models_dir=models_dir,
        features=features, engine_dir=str(stub_engine), verbose=False,
        batch_key="archtest", **kwargs)


# =============================================================================
# 1. Mode routing
# =============================================================================

def test_batch_is_the_default_mode():
    """An existing production command must keep its behaviour."""
    assert mr.DEFAULT_MODE == mr.MODE_BATCH
    args = mr._build_parser().parse_args(["--local-only"])
    assert args.mode is None
    assert mr.resolve_mode(args.mode) == mr.MODE_BATCH


def test_mode_choices_and_env_override(monkeypatch):
    assert mr.MODES == ("batch", "sequential")
    assert mr.resolve_mode("sequential") == mr.MODE_SEQUENTIAL
    monkeypatch.setenv(mr.MODE_ENV_VAR, "sequential")
    assert mr.resolve_mode() == mr.MODE_SEQUENTIAL
    assert mr.resolve_mode("batch") == mr.MODE_BATCH      # argument still wins
    with pytest.raises(ValueError):
        mr.resolve_mode("nope")


def test_mode_batch_routes_to_the_batch_architecture(monkeypatch, tmp_path):
    """`--mode batch` must execute Batch, never the Sequential path."""
    called = []
    monkeypatch.setattr(mr, "process_batch",
                        lambda **kw: called.append("batch") or _outcome())
    monkeypatch.setattr(mr, "_run_sequential_local",
                        lambda **kw: called.append("sequential") or 0)
    _local_inputs(tmp_path)

    mr.run_local(local_inputs=str(tmp_path), batch_key="k",
                 workspace=str(tmp_path / "ws"), recon_models_dir=str(tmp_path),
                 feat_models_dir=str(tmp_path), mode="batch")
    assert called == ["batch"]


def test_mode_sequential_routes_to_the_sequential_architecture(monkeypatch,
                                                               tmp_path):
    called = []
    monkeypatch.setattr(mr, "process_batch",
                        lambda **kw: called.append("batch") or _outcome())
    monkeypatch.setattr(mr, "_run_sequential_local",
                        lambda **kw: called.append("sequential") or 0)
    _local_inputs(tmp_path)

    mr.run_local(local_inputs=str(tmp_path), batch_key="k",
                 workspace=str(tmp_path / "ws"), recon_models_dir=str(tmp_path),
                 feat_models_dir=str(tmp_path), mode="sequential")
    assert called == ["sequential"]


def _local_inputs(tmp_path):
    for camera in C.ALL_CAMERAS:
        (tmp_path / ("%s.mp4" % camera)).write_bytes(b"v")


def _outcome():
    class _O:
        final_status = C.BATCH_OK if hasattr(C, "BATCH_OK") else "ok"
        report_pdf_path = None
        report_json_path = None
        camera_pdf_paths: dict = {}
        processed_video_paths: dict = {}
        error = None
    return _O()


def test_sequential_accepts_partial_cameras_batch_still_requires_four(
        monkeypatch, tmp_path, capsys):
    """The core business goal: one camera is enough for Sequential."""
    (tmp_path / "RIGHT_UP.mp4").write_bytes(b"v")        # only one camera

    seen = {}
    monkeypatch.setattr(mr, "_run_sequential_local",
                        lambda **kw: seen.update(kw) or 0)
    code = mr.run_local(local_inputs=str(tmp_path), batch_key="k",
                        workspace=str(tmp_path / "ws"),
                        recon_models_dir=str(tmp_path),
                        feat_models_dir=str(tmp_path), mode="sequential")
    assert code == 0
    assert list(seen["video_paths"]) == [C.CAMERA_RIGHT_UP]

    # Batch keeps its strict requirement, unchanged.
    code = mr.run_local(local_inputs=str(tmp_path), batch_key="k",
                        workspace=str(tmp_path / "ws"),
                        recon_models_dir=str(tmp_path),
                        feat_models_dir=str(tmp_path), mode="batch")
    assert code == 2
    assert "missing videos" in capsys.readouterr().err


# =============================================================================
# 2. One decode per camera; GAP sees every frame; strides honoured
# =============================================================================

def test_exactly_one_decode_per_camera(stub_engine, inputs, tmp_path, wired):
    result = _process(C.CAMERA_RIGHT_UP, workspace=tmp_path / "ws",
                      stub_engine=stub_engine, inputs=inputs)
    assert result.sealed
    assert FakeCapture.open_count == 1, (
        "the camera opened its video %d times; Sequential allows exactly one"
        % FakeCapture.open_count)
    assert result.decode_passes == 1


def test_gap_receives_every_decoded_frame(stub_engine, inputs, tmp_path, wired):
    _process(C.CAMERA_RIGHT_UP, workspace=tmp_path / "ws",
             stub_engine=stub_engine, inputs=inputs)
    with open(os.environ["STUB_GAP_CALLS"], encoding="utf-8") as handle:
        gap_calls = json.load(handle)
    assert gap_calls == list(range(FRAME_COUNT)), (
        "GAP is continuous: it must see every decoded frame, in order")


def test_feature_strides_are_three_three_two(stub_engine, inputs, tmp_path,
                                             wired):
    # RIGHT_UP is a side camera -> door only.
    _process(C.CAMERA_RIGHT_UP, workspace=tmp_path / "ws",
             stub_engine=stub_engine, inputs=inputs)
    assert wired["door"] == list(range(0, FRAME_COUNT, 3))
    assert wired["damage"] == [] and wired["load"] == []

    # A top camera -> damage (3) and load (2), no door.
    wired["door"].clear()
    _process(C.CAMERA_RIGHT_UP_TOP, workspace=tmp_path / "ws2",
             stub_engine=stub_engine, inputs=inputs)
    assert wired["damage"] == list(range(0, FRAME_COUNT, 3))
    assert wired["load"] == list(range(0, FRAME_COUNT, 2))
    assert wired["door"] == []


def test_feature_camera_mapping_matches_batch():
    assert camera_runner.FEATURE_CAMERAS["door"] == C.SIDE_CAMERAS
    assert camera_runner.FEATURE_CAMERAS["damage"] == C.TOP_CAMERAS
    assert camera_runner.FEATURE_CAMERAS["load"] == C.TOP_CAMERAS
    assert camera_runner.DEFAULT_STRIDES["door"] == 3
    assert camera_runner.DEFAULT_STRIDES["damage"] == 3
    assert camera_runner.DEFAULT_STRIDES["load"] == 2


def test_ocr_is_not_selected_for_a_camera_run(stub_engine, inputs, tmp_path,
                                              wired):
    """`--features door,load,damage` must leave OCR out of the camera run."""
    workspace = tmp_path / "ws"
    _process(C.CAMERA_RIGHT_UP, workspace=workspace, stub_engine=stub_engine,
             inputs=inputs, features=("door", "load", "damage"))

    assert camera_runner.features_for_camera(
        C.CAMERA_RIGHT_UP, ("door", "load", "damage")) == ("door",)
    camera_evidence = ev.load_evidence(str(workspace), C.CAMERA_RIGHT_UP)
    assert camera_evidence.observations_for("ocr") == []
    assert "ocr" not in camera_evidence.feature_config["features"]
    seal = ev.load_seal(str(workspace), C.CAMERA_RIGHT_UP)
    assert "ocr" not in seal["feature_config"]["features"]
    # ...and no OCR weight fingerprint was even taken.
    assert not any(key.endswith("_ocr")
                   for key in seal["model_fingerprints"])


def test_sequential_import_pulls_in_neither_ocr_nor_easyocr():
    """Process-isolated: an in-process sys.modules check would be polluted by
    any earlier test that imported the OCR processor."""
    import subprocess

    code = (
        "import sys; sys.path.insert(0, %r);"
        "import sequential.runner, sequential.camera_runner,"
        " sequential.global_assembly;"
        "print('OCR=' + str('features.ocr.processor' in sys.modules));"
        "print('EASYOCR=' + str('easyocr' in sys.modules))" % _REPO_ROOT
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True,
                         text=True, cwd=_REPO_ROOT)
    assert out.returncode == 0, out.stderr
    assert "OCR=False" in out.stdout, out.stdout
    assert "EASYOCR=False" in out.stdout, out.stdout


# =============================================================================
# 3. Camera-local persistence, sealing, resume, release
# =============================================================================

def test_camera_persists_evidence_and_reports_then_seals(stub_engine, inputs,
                                                         tmp_path, wired):
    workspace = tmp_path / "ws"
    result = _process(C.CAMERA_RIGHT_UP, workspace=workspace,
                      stub_engine=stub_engine, inputs=inputs)

    assert os.path.isfile(result.evidence_path)
    assert os.path.isfile(result.seal_path)
    assert os.path.isfile(result.report_paths["json_path"])

    # The seal is written LAST, so it may never be newer-than-nothing: the
    # evidence and the report must both already exist when it appears.
    seal_mtime = os.path.getmtime(result.seal_path)
    assert os.path.getmtime(result.evidence_path) <= seal_mtime
    assert os.path.getmtime(result.report_paths["json_path"]) <= seal_mtime

    seal = ev.load_seal(str(workspace), C.CAMERA_RIGHT_UP)
    for key in ("camera_id", "status", "frame_count", "fps",
                "video_fingerprint", "model_fingerprints",
                "config_fingerprint", "schema_version", "processing_seconds",
                "feature_config", "evidence_path", "reports"):
        assert key in seal, "seal is missing %r" % key
    assert seal["status"] == ev.STATUS_SEALED
    assert seal["frame_count"] == FRAME_COUNT


def test_camera_evidence_has_no_canonical_wagon_ids(stub_engine, inputs,
                                                    tmp_path, wired):
    workspace = tmp_path / "ws"
    result = _process(C.CAMERA_RIGHT_UP, workspace=workspace,
                      stub_engine=stub_engine, inputs=inputs)
    for path in (result.evidence_path, result.seal_path,
                 result.report_paths["json_path"]):
        with open(path, encoding="utf-8") as handle:
            text = handle.read()
        assert not re.search(r"\bGW_\d+\b", text), (
            "%s contains a canonical global wagon id; those belong to Global "
            "Assembly only" % path)

    camera_evidence = ev.load_evidence(str(workspace), C.CAMERA_RIGHT_UP)
    assert camera_evidence.segments
    for segment in camera_evidence.segments:
        assert segment["segment_id"].startswith(C.CAMERA_RIGHT_UP + "_SEG_")
        assert segment["canonical"] is False


def test_evidence_contains_what_assembly_needs(stub_engine, inputs, tmp_path,
                                               wired):
    workspace = tmp_path / "ws"
    _process(C.CAMERA_RIGHT_UP, workspace=workspace, stub_engine=stub_engine,
             inputs=inputs)
    camera_evidence = ev.load_evidence(str(workspace), C.CAMERA_RIGHT_UP)

    assert camera_evidence.camera_id == C.CAMERA_RIGHT_UP
    assert camera_evidence.timing.fps == FPS
    assert camera_evidence.timing.decoded_frames == FRAME_COUNT
    assert camera_evidence.timing.wagon_region_frames > 0
    assert camera_evidence.gaps and all(
        g.confirmation_frame >= 0 for g in camera_evidence.gaps)
    assert camera_evidence.observations
    assert camera_evidence.classification_timeline
    assert camera_evidence.provenance["video"]["fingerprint"]
    assert camera_evidence.provenance["models"]
    assert camera_evidence.provenance["decode_passes"] == 1
    assert camera_evidence.schema_version == ev.SCHEMA_VERSION
    observation = camera_evidence.observations[0]
    for field in ("feature", "frame_idx", "timestamp", "confidence",
                  "raw_class"):
        assert hasattr(observation, field)


def test_matching_seal_resumes_without_rerunning_inference(
        stub_engine, inputs, tmp_path, wired):
    workspace = tmp_path / "ws"
    first = _process(C.CAMERA_RIGHT_UP, workspace=workspace,
                     stub_engine=stub_engine, inputs=inputs)
    assert first.decode_passes == 1

    FakeCapture.open_count = 0
    wired["door"].clear()
    second = _process(C.CAMERA_RIGHT_UP, workspace=workspace,
                      stub_engine=stub_engine, inputs=inputs)

    assert second.reused is True
    assert second.decode_passes == 0
    assert FakeCapture.open_count == 0, "resume must not open the video"
    assert wired["door"] == [], "resume must not run inference"


def test_stale_evidence_is_reprocessed_and_the_reason_is_reported(
        stub_engine, inputs, tmp_path, wired):
    workspace = tmp_path / "ws"
    video_paths, models_dir = inputs
    _process(C.CAMERA_RIGHT_UP, workspace=workspace, stub_engine=stub_engine,
             inputs=inputs)

    # Change the input video -> the fingerprint no longer matches.
    with open(video_paths[C.CAMERA_RIGHT_UP], "wb") as handle:
        handle.write(b"different video content")
    os.utime(video_paths[C.CAMERA_RIGHT_UP], (1, 1))

    FakeCapture.open_count = 0
    again = _process(C.CAMERA_RIGHT_UP, workspace=workspace,
                     stub_engine=stub_engine, inputs=inputs)
    assert again.reused is False
    assert "video changed" in again.reason
    assert FakeCapture.open_count == 1


def test_force_ignores_a_matching_seal(stub_engine, inputs, tmp_path, wired):
    workspace = tmp_path / "ws"
    _process(C.CAMERA_RIGHT_UP, workspace=workspace, stub_engine=stub_engine,
             inputs=inputs)
    FakeCapture.open_count = 0
    forced = _process(C.CAMERA_RIGHT_UP, workspace=workspace,
                      stub_engine=stub_engine, inputs=inputs, force=True)
    assert forced.reused is False
    assert FakeCapture.open_count == 1


def test_resources_are_released_after_sealing(stub_engine, inputs, tmp_path,
                                              wired):
    from features import _common

    _common._MODEL_CACHE["stale"] = object()
    _process(C.CAMERA_RIGHT_UP, workspace=tmp_path / "ws",
             stub_engine=stub_engine, inputs=inputs)
    assert _common._MODEL_CACHE == {}, (
        "a camera's models must be released before the next camera starts")


def test_the_one_capture_is_always_released(stub_engine, inputs, tmp_path,
                                            wired):
    captured = []
    real_init = FakeCapture.__init__

    def _tracking_init(self, path):
        real_init(self, path)
        captured.append(self)

    FakeCapture.__init__ = _tracking_init
    try:
        _process(C.CAMERA_RIGHT_UP, workspace=tmp_path / "ws",
                 stub_engine=stub_engine, inputs=inputs)
    finally:
        FakeCapture.__init__ = real_init
    assert captured and all(capture.released for capture in captured)


def test_engine_session_leaves_nothing_behind(stub_engine, inputs, tmp_path,
                                              wired):
    _process(C.CAMERA_RIGHT_UP, workspace=tmp_path / "ws",
             stub_engine=stub_engine, inputs=inputs)
    for name in gc_runner.ENGINE_MODULES:
        module = sys.modules.get(name)
        origin = getattr(module, "__file__", None) if module else None
        if origin:
            assert not os.path.abspath(origin).startswith(str(stub_engine))
    import reporting
    assert os.path.abspath(reporting.__file__).startswith(_REPO_ROOT)


# =============================================================================
# 4. No canonical roster during camera processing
# =============================================================================

def test_camera_processing_creates_no_canonical_roster(stub_engine, inputs,
                                                       tmp_path, wired):
    workspace = tmp_path / "ws"
    _process(C.CAMERA_RIGHT_UP, workspace=workspace, stub_engine=stub_engine,
             inputs=inputs)
    # No global contract may exist yet.
    assert not os.path.exists(
        os.path.join(str(workspace), "global_state", "global_train_state.json"))
    assert not os.path.exists(
        os.path.join(str(workspace), ev.COMBINED_DIRNAME,
                     "combined_train_report.json"))


def test_a_single_camera_report_is_valid_on_its_own(stub_engine, inputs,
                                                    tmp_path, wired):
    """RIGHT_UP alone must produce its JSON and PDF, no waiting."""
    workspace = tmp_path / "ws"
    result = _process(C.CAMERA_RIGHT_UP, workspace=workspace,
                      stub_engine=stub_engine, inputs=inputs)

    with open(result.report_paths["json_path"], encoding="utf-8") as handle:
        document = json.load(handle)
    assert document["report_type"] == "single_camera"
    assert document["canonical"] is False
    assert "NOT canonical" in document["disclaimer"]
    assert document["camera_id"] == C.CAMERA_RIGHT_UP
    assert document["local_gaps"]["count"] >= 1
    assert "canonical gap authority" in document["gap_authority"]

    try:
        import reportlab                                        # noqa: F401
    except ImportError:
        pytest.skip("reportlab not installed")
    assert result.report_paths["pdf_path"]
    assert os.path.getsize(result.report_paths["pdf_path"]) > 800


# =============================================================================
# 5. Global Assembly
# =============================================================================

def _seal_all(stub_engine, inputs, workspace, wired, cameras=C.ALL_CAMERAS):
    results = []
    for camera_id in cameras:
        results.append(_process(camera_id, workspace=workspace,
                                stub_engine=stub_engine, inputs=inputs))
    return results


def test_assembly_is_not_ready_without_the_authority(tmp_path):
    ready, sealed, missing, reason = global_assembly.readiness(str(tmp_path))
    assert ready is False
    assert sealed == []
    assert C.MASTER_CAMERA in missing
    assert C.MASTER_CAMERA in reason


def test_assembly_refuses_to_fabricate_a_train(tmp_path):
    result = global_assembly.assemble(
        workspace=str(tmp_path), repo_root=_REPO_ROOT, batch_key="k",
        verbose=False)
    assert result.ready is False
    assert result.report_json_path is None
    assert not os.path.exists(os.path.join(str(tmp_path), ev.COMBINED_DIRNAME,
                                           "combined_train_report.json"))


def test_assembly_builds_exactly_one_canonical_roster(stub_engine, inputs,
                                                      tmp_path, wired):
    workspace = tmp_path / "ws"
    _seal_all(stub_engine, inputs, workspace, wired)

    result = global_assembly.assemble(
        workspace=str(workspace), repo_root=_REPO_ROOT, batch_key="archtest",
        engine_dir=str(stub_engine), verbose=False)

    assert result.ready, result.reason
    # three gap frames -> three canonical gaps -> two canonical wagons
    assert result.global_gap_count == 3
    assert result.global_wagon_count == result.global_gap_count - 1

    from core.global_state_loader import (load_global_train_state,
                                          verify_roster_integrity)
    state = load_global_train_state(result.state_json_path)
    assert [w.global_id for w in state.wagons] == ["GW_1", "GW_2"]
    assert verify_roster_integrity(state) == []
    assert state.master_camera == C.MASTER_CAMERA
    assert state.uses_camera_frame_ranges
    assert len(state.global_gaps) == state.total_wagons + 1


def test_assembly_produces_one_combined_report(stub_engine, inputs, tmp_path,
                                               wired):
    workspace = tmp_path / "ws"
    _seal_all(stub_engine, inputs, workspace, wired)
    result = global_assembly.assemble(
        workspace=str(workspace), repo_root=_REPO_ROOT, batch_key="archtest",
        engine_dir=str(stub_engine), verbose=False)

    assert result.report_json_path and os.path.isfile(result.report_json_path)
    assert ev.COMBINED_DIRNAME in result.report_json_path
    try:
        import reportlab                                        # noqa: F401
    except ImportError:
        return
    assert result.report_pdf_path and os.path.isfile(result.report_pdf_path)


def test_right_up_is_the_canonical_gap_authority():
    assert C.MASTER_CAMERA == C.CAMERA_RIGHT_UP
    assert global_assembly.required_cameras() == (C.CAMERA_RIGHT_UP,)


def test_support_camera_extra_gap_is_diagnostic_not_a_wagon(stub_engine, inputs,
                                                            tmp_path, wired):
    """An extra gap seen only by a support camera must not create a wagon."""
    workspace = tmp_path / "ws"
    _seal_all(stub_engine, inputs, workspace, wired)

    # Give LEFT_UP one extra local gap, far from any canonical position.
    document = json.loads(open(ev.evidence_path(str(workspace),
                                                C.CAMERA_LEFT_UP),
                               encoding="utf-8").read())
    document["gaps"].append({
        "local_gap_id": "LEFT_UP_EXTRA", "confirmation_frame": 40,
        "first_frame": 40, "last_frame": 40, "normalized_position": 700.0,
        "max_confidence": 0.9, "average_confidence": 0.9, "frame_count": 1})
    with open(ev.evidence_path(str(workspace), C.CAMERA_LEFT_UP), "w",
              encoding="utf-8") as handle:
        json.dump(document, handle)

    result = global_assembly.assemble(
        workspace=str(workspace), repo_root=_REPO_ROOT, batch_key="archtest",
        engine_dir=str(stub_engine), verbose=False)

    assert result.global_gap_count == 3, "a support camera changed the count"
    assert result.global_wagon_count == 2
    extra = (result.diagnostics["alignments"][C.CAMERA_LEFT_UP]
             ["extra_camera_gaps"])
    assert extra and "DIAGNOSTIC" in extra[0]["note"]


def test_support_camera_missing_gap_keeps_the_canonical_gap(stub_engine, inputs,
                                                            tmp_path, wired):
    """A canonical gap a support camera missed is projected, never dropped."""
    workspace = tmp_path / "ws"
    _seal_all(stub_engine, inputs, workspace, wired)

    path = ev.evidence_path(str(workspace), C.CAMERA_LEFT_UP)
    document = json.loads(open(path, encoding="utf-8").read())
    document["gaps"] = document["gaps"][:1]          # LEFT_UP now misses two
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(document, handle)

    result = global_assembly.assemble(
        workspace=str(workspace), repo_root=_REPO_ROOT, batch_key="archtest",
        engine_dir=str(stub_engine), verbose=False)

    assert result.global_gap_count == 3
    from core.global_state_loader import load_global_train_state
    state = load_global_train_state(result.state_json_path)
    # LEFT_UP still receives a window for every canonical wagon: the gap it
    # missed was projected in rather than dropped.
    from global_counting import adapter
    for wagon in state.wagons:
        entry = wagon.camera_frame_ranges[C.CAMERA_LEFT_UP]
        assert entry["start_frame"] is not None
        assert entry["status"] in (adapter.STATUS_DETECTED,
                                  adapter.STATUS_RECOVERED)
    recovered = [w.global_id for w in state.wagons
                 if w.camera_frame_ranges[C.CAMERA_LEFT_UP]["status"]
                 == adapter.STATUS_RECOVERED]
    assert recovered, "a missed canonical gap must be projected as RECOVERED"


def test_assembly_assigns_observations_through_ownership(stub_engine, inputs,
                                                         tmp_path, wired):
    workspace = tmp_path / "ws"
    _seal_all(stub_engine, inputs, workspace, wired)
    result = global_assembly.assemble(
        workspace=str(workspace), repo_root=_REPO_ROOT, batch_key="archtest",
        engine_dir=str(stub_engine), verbose=False)
    assert result.ready

    from core import wagon_ownership
    from core.global_state_loader import load_global_train_state
    state = load_global_train_state(result.state_json_path)
    ownership = wagon_ownership.for_state(state)
    assert ownership is not None
    assert wagon_ownership.BOUNDARY_GOES_TO == "next_wagon"

    evidences = {camera_id: ev.load_evidence(str(workspace), camera_id)
                 for camera_id in C.ALL_CAMERAS}
    assigned = global_assembly.assign_observations(state, evidences)

    # No observation may be claimed by two wagons.
    seen = {}
    for gw_id, features in assigned.items():
        for feature, per_camera in features.items():
            for camera_id, observations in per_camera.items():
                for observation in observations:
                    key = (camera_id, feature, observation.frame_idx,
                           observation.raw_class, observation.confidence)
                    seen.setdefault(key, []).append(gw_id)
    doubles = {k: v for k, v in seen.items() if len(set(v)) > 1}
    assert doubles == {}, "observation owned by two wagons: %s" % list(doubles)[:3]


def test_assembly_preserves_multi_door(stub_engine, inputs, tmp_path, wired):
    """The b6f67b5 multi-door behaviour must survive the new assembly path."""
    workspace = tmp_path / "ws"
    _seal_all(stub_engine, inputs, workspace, wired)
    global_assembly.assemble(
        workspace=str(workspace), repo_root=_REPO_ROOT, batch_key="archtest",
        engine_dir=str(stub_engine), verbose=False)

    door_dir = os.path.join(str(workspace), "wagon_states", "door")
    payloads = [json.load(open(os.path.join(door_dir, name), encoding="utf-8"))
                for name in sorted(os.listdir(door_dir))]
    assert payloads
    for payload in payloads:
        assert "doors" in payload
        assert "door_status" in payload
        for door in payload["doors"]:
            assert "door_index" in door and "state" in door
        # the stub alternates open/closed, so a wagon sees both states
    states = {d["state"] for p in payloads for d in p["doors"]}
    assert states, "no door state survived assembly"


# =============================================================================
# 6. Static architecture audit
# =============================================================================

def _source(relative):
    with open(os.path.join(_REPO_ROOT, relative), encoding="utf-8") as handle:
        return handle.read()


def _code_only(text):
    """Strip comments and docstring blocks well enough to audit real calls."""
    without_docstrings = re.sub(r'"""(?:.|\n)*?"""', "", text)
    return "\n".join(line for line in without_docstrings.splitlines()
                     if not line.strip().startswith("#"))


AUDIT_FORBIDDEN_IN_ASSEMBLY = (
    "VideoCapture", "load_yolo", "GapTracker", ".predict(",
    "detect_gaps_in_frame", "classify_video_frames",
)


def test_global_assembly_performs_no_inference_and_no_decode():
    """The audit that keeps `INFERENCE ONCE -> PERSIST -> INTERPRET` true."""
    code = _code_only(_source("sequential/global_assembly.py"))
    for needle in AUDIT_FORBIDDEN_IN_ASSEMBLY:
        assert needle not in code, (
            "Global Assembly must not %r: it consumes persisted evidence only"
            % needle)


def test_camera_local_code_builds_no_canonical_roster():
    for relative in ("sequential/camera_runner.py", "sequential/camera_report.py"):
        code = _code_only(_source(relative))
        assert not re.search(r'"GW_|\bGW_\d', code), (
            "%s references a canonical wagon id" % relative)
        for needle in ("build_global_train_state_document", "write_documents",
                       "verify_roster_integrity"):
            assert needle not in code, (
                "%s builds the canonical contract; that is Global Assembly's "
                "job" % relative)


def test_camera_runner_opens_exactly_one_capture_in_source():
    code = _code_only(_source("sequential/camera_runner.py"))
    assert code.count("cv2.VideoCapture(") == 1, (
        "one decode lifecycle per camera means exactly one VideoCapture site")
    assert code.count("capture.release()") >= 1


def test_only_global_assembly_writes_the_combined_report():
    assembly = _code_only(_source("sequential/global_assembly.py"))
    assert "combined_train_report" in assembly
    for relative in ("sequential/camera_runner.py",
                     "sequential/camera_report.py",
                     "sequential/runner.py"):
        assert "combined_train_report" not in _code_only(_source(relative)), (
            "%s must not write the combined report" % relative)


def test_master_runner_does_not_duplicate_detector_logic():
    code = _code_only(_source("orchestrator/master_runner.py"))
    for needle in ("VideoCapture", "detect_gaps_in_frame", "load_yolo",
                   ".predict("):
        assert needle not in code, (
            "master_runner is a selection layer; %r belongs in an "
            "architecture module" % needle)


def test_global_count_ec2_is_not_vendored():
    assert not os.path.exists(os.path.join(_REPO_ROOT,
                                           "global_wagon_pipeline.py"))
    for root, dirs, files in os.walk(os.path.join(_REPO_ROOT, "sequential")):
        dirs[:] = [d for d in dirs if d != "__pycache__"]
        assert "global_wagon_pipeline.py" not in files


# =============================================================================
# 7. Sequential runner orchestration
# =============================================================================

def test_camera_order_is_deterministic_and_configuration_driven():
    paths = {camera: "x" for camera in reversed(C.ALL_CAMERAS)}
    assert sequential_runner.camera_order(paths) == list(C.ALL_CAMERAS)
    assert sequential_runner.camera_order({C.CAMERA_LEFT_UP_TOP: "x"}) == \
        [C.CAMERA_LEFT_UP_TOP]


def test_runner_processes_cameras_in_order_then_assembles(monkeypatch, tmp_path):
    """Sequential is camera -> seal -> camera -> ... -> ONE assembly."""
    events = []

    def _fake_camera(**kwargs):
        events.append(("camera", kwargs["camera_id"]))
        return camera_runner.CameraRunResult(
            camera_id=kwargs["camera_id"], status=ev.STATUS_SEALED)

    def _fake_assemble(**kwargs):
        events.append(("assembly", None))
        return global_assembly.AssemblyResult(ready=True, reason="ok")

    monkeypatch.setattr(camera_runner, "process_camera", _fake_camera)
    monkeypatch.setattr(global_assembly, "assemble", _fake_assemble)

    sequential_runner.run_sequential(
        video_paths={c: "v" for c in C.ALL_CAMERAS},
        workspace=str(tmp_path), repo_root=_REPO_ROOT,
        recon_models_dir=str(tmp_path), feat_models_dir=str(tmp_path),
        features=("door", "load", "damage"), batch_key="k", verbose=False)

    assert events == [("camera", c) for c in C.ALL_CAMERAS] + [("assembly", None)]
    assert len([e for e in events if e[0] == "assembly"]) == 1, (
        "exactly one Global Assembly per run")


def test_a_failing_camera_does_not_stop_the_others(monkeypatch, tmp_path):
    def _fake_camera(**kwargs):
        if kwargs["camera_id"] == C.CAMERA_LEFT_UP:
            raise RuntimeError("boom")
        return camera_runner.CameraRunResult(
            camera_id=kwargs["camera_id"], status=ev.STATUS_SEALED)

    monkeypatch.setattr(camera_runner, "process_camera", _fake_camera)
    monkeypatch.setattr(global_assembly, "assemble",
                        lambda **kw: global_assembly.AssemblyResult(True))

    outcome = sequential_runner.run_sequential(
        video_paths={c: "v" for c in C.ALL_CAMERAS},
        workspace=str(tmp_path), repo_root=_REPO_ROOT,
        recon_models_dir=str(tmp_path), feat_models_dir=str(tmp_path),
        features=("door",), batch_key="k", verbose=False)

    assert outcome.failed_cameras == [C.CAMERA_LEFT_UP]
    assert len(outcome.sealed_cameras) == 3


def test_skip_assembly_stops_after_camera_reports(monkeypatch, tmp_path):
    monkeypatch.setattr(camera_runner, "process_camera",
                        lambda **kw: camera_runner.CameraRunResult(
                            camera_id=kw["camera_id"], status=ev.STATUS_SEALED))
    called = []
    monkeypatch.setattr(global_assembly, "assemble",
                        lambda **kw: called.append(1))

    outcome = sequential_runner.run_sequential(
        video_paths={C.CAMERA_RIGHT_UP: "v"}, workspace=str(tmp_path),
        repo_root=_REPO_ROOT, recon_models_dir=str(tmp_path),
        feat_models_dir=str(tmp_path), features=("door",), batch_key="k",
        skip_assembly=True, verbose=False)
    assert called == []
    assert outcome.assembly is None


def test_single_camera_run_reports_then_declines_assembly(stub_engine, inputs,
                                                          tmp_path, wired):
    """End to end with ONE camera: local report yes, combined report no."""
    video_paths, models_dir = inputs
    workspace = tmp_path / "ws"

    outcome = sequential_runner.run_sequential(
        video_paths={C.CAMERA_LEFT_UP: video_paths[C.CAMERA_LEFT_UP]},
        workspace=str(workspace), repo_root=_REPO_ROOT,
        recon_models_dir=models_dir, feat_models_dir=models_dir,
        features=("door", "load", "damage"), batch_key="solo",
        engine_dir=str(stub_engine), verbose=False)

    assert outcome.sealed_cameras == [C.CAMERA_LEFT_UP]
    paths = outcome.cameras[0].report_paths
    assert os.path.isfile(paths["json_path"])
    # RIGHT_UP is the canonical authority and is absent -> no combined train.
    assert outcome.assembly is not None
    assert outcome.assembly.ready is False
    assert C.MASTER_CAMERA in outcome.assembly.reason
