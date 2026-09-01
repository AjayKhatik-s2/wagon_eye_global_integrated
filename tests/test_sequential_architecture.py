"""The dual-mode architecture, under the EXACT-PARITY contract.

Batch is the golden reference. Sequential may change execution architecture,
timing, decode count, intermediate storage and WHEN inference happens -- but for
the same four videos, models, configuration and feature selection it must
produce the same canonical output. Three parts of the contract these tests
enforce changed deliberately, and the obsolete assertions are named here so the
change is auditable rather than silent:

  OBSOLETE (A) "exactly one decode per camera" / "the one capture is always
      released" / "GAP receives every decoded frame".
      The camera stage no longer decodes anything. It calls the ENGINE's own
      `camera_pipeline.process_camera`, which classifies, trims (writing the
      re-encoded clip) and detects gaps ON THAT CLIP -- which is the only way
      the gap input pixels can equal Batch's. Replaced by a STRONGER assertion:
      `camera_runner` contains no capture, no detector and no tracker at all.

  OBSOLETE (A) "camera feature strides are 3/3/2".
      Door/Damage/Load inference moved to Global Assembly, because Batch infers
      them over each wagon's stable interior of JPEG-90 cached frames -- a frame
      set that cannot exist before the roster does. The strides are still
      asserted, now where they are actually applied (see
      test_assembly_runs_batchs_processors_with_batchs_strides), and the
      behaviour of the sampler is covered by test_sampled_inference_modes.py.

  OBSOLETE (A) "Global Assembly performs no inference and no decode".
      That was the old `INFERENCE ONCE -> PERSIST -> INTERPRET` rule. Assembly
      now runs Batch's own materializer and processors, which is the point.
      Replaced by an assertion that it calls BATCH's implementations rather
      than mirroring them.

Camera-stage tests drive a stub that implements the ENGINE's process_camera
contract. Assembly tests drive the REAL engine, which -- as the fixtures in
tests/_parity_fixtures.py prove -- can execute the entire global phase from
persisted records with no video and no models.

    python -m pytest tests/test_sequential_architecture.py -q
"""

from __future__ import annotations

import json
import os
import re
import sys

import numpy as np
import pytest

_TEST_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_TEST_DIR)
for _path in (_REPO_ROOT, _TEST_DIR):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import _parity_fixtures as F

from core import constants as C
from global_counting import runner as gc_runner
from orchestrator import master_runner as mr
from sequential import camera_runner, evidence as ev, global_assembly
from sequential import runner as sequential_runner


FPS = F.FPS
TOTAL_FRAMES = F.TOTAL_FRAMES
REGION_START = F.REGION_START
REGION_END = F.REGION_END
TRIMMED_FRAMES = F.TRIMMED_FRAMES
GAP_POSITIONS = list(F.CANONICAL_POSITIONS)

WIDTH, HEIGHT = 640, 480

REAL_ENGINE = F.real_engine_dir()
needs_engine = pytest.mark.skipif(
    REAL_ENGINE is None,
    reason="the frozen global_wagon_app checkout is not on this machine")


# =============================================================================
# A stub implementing the ENGINE's per-camera contract
# =============================================================================

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
GENERATE_TRIM_DEBUG_VIDEO = True
GENERATE_GAP_ANNOTATED_VIDEO = True
NORMALIZED_TIMELINE_SCALE = 1000.0
_OVERRIDABLE = ("GENERATE_TRIM_DEBUG_VIDEO", "GENERATE_GAP_ANNOTATED_VIDEO")

def apply_overrides(**overrides):
    applied = {}
    for name, value in overrides.items():
        if value is None:
            continue
        if name not in _OVERRIDABLE:
            raise KeyError(name)
        globals()[name] = value
        applied[name] = value
    return applied
''',

"runtime.py": ('DEVICE = "cpu"\nDEVICE_YOLO = "cpu"\nDEVICE_LABEL = "CPU"\n'
               'USE_HALF = False\n'),

"io_paths.py": '''
from pathlib import Path

VIDEO_PATHS = {}
OUTPUT_DIR = None
_NAMES = ("normalized_gap_timelines", "camera_alignment_summary",
          "global_gap_timeline", "unmatched_extra_detections",
          "global_wagon_timeline", "global_wagon_snapshots")

def prepare_output_dirs(output_dir):
    global OUTPUT_DIR
    OUTPUT_DIR = Path(output_dir)
    root = OUTPUT_DIR / "global"
    root.mkdir(parents=True, exist_ok=True)
    return {name: root / (name + ".csv") for name in _NAMES}
''',

# Deliberately named like ours, to keep exercising engine_session isolation.
"models.py": '''
CLASSIFICATION_MODELS, GAP_MODELS = {}, {}
CLASSIFICATION_CLASS_MAPS, GAP_CLASS_MAPS = {}, {}
LOADED_CALLS = []

def load_all_models(classification_paths, gap_paths):
    """Mirrors the engine, INCLUDING its all-or-nothing validation."""
    from camera_map import CAMERA_CLASSIFICATION_MODEL, CAMERA_GAP_MODEL

    LOADED_CALLS.append((sorted(classification_paths), sorted(gap_paths)))
    CLASSIFICATION_MODELS.clear(); GAP_MODELS.clear()
    for key, path in classification_paths.items():
        path.stat()
        CLASSIFICATION_MODELS[key] = {"path": path, "task": "classify"}
    absent = [k for k in sorted(set(CAMERA_CLASSIFICATION_MODEL.values()))
              if k not in CLASSIFICATION_MODELS]
    if absent:
        raise RuntimeError("Classification model(s) required by the camera "
                           "mapping are not loaded: %s" % absent)
    for key, path in gap_paths.items():
        path.stat()
        GAP_MODELS[key] = {"path": path, "task": "detect"}
    absent = [k for k in sorted(set(CAMERA_GAP_MODEL.values()))
              if k not in GAP_MODELS]
    if absent:
        raise RuntimeError("Gap model(s) required by the camera mapping are "
                           "not loaded: %s" % absent)
    return CLASSIFICATION_MODELS, GAP_MODELS

def build_class_maps():
    CLASSIFICATION_CLASS_MAPS.clear(); GAP_CLASS_MAPS.clear()
    for key in CLASSIFICATION_MODELS:
        CLASSIFICATION_CLASS_MAPS[key] = {"raw": {0: "wagon"}}
    for key in GAP_MODELS:
        GAP_CLASS_MAPS[key] = {"raw": {0: "gap"}, "gap_ids": [0]}
''',

# The engine's per-camera pipeline: classification + trimming + gap detection
# ON THE TRIMMED CLIP. The stub returns a result of the SAME SHAPE, and records
# that it was called, which is what the camera-stage tests assert.
"camera_pipeline.py": '''
import json
import os
import pandas as pd
from io_paths import VIDEO_PATHS

CAMERA_RESULTS = {}
PROCESSED = []

FPS = %(fps)r
TOTAL_FRAMES = %(total)d
REGION_START, REGION_END = %(start)d, %(end)d
TRIMMED = REGION_END - REGION_START + 1
POSITIONS = %(positions)r

def process_camera(camera, force=False):
    """Same contract as the engine: decodes, trims, detects, records."""
    if camera not in VIDEO_PATHS:
        raise KeyError("no video registered for %(pct)sr" %(pct)s camera)
    PROCESSED.append(camera)
    marker = os.environ.get("STUB_PROCESS_CALLS")
    if marker:
        existing = []
        if os.path.isfile(marker):
            existing = json.load(open(marker, encoding="utf-8"))
        existing.append(camera)
        json.dump(existing, open(marker, "w", encoding="utf-8"))

    rows = []
    for index, position in enumerate(POSITIONS, start=1):
        frame = int(round(position / 1000.0 * (TRIMMED - 1)))
        rows.append({"camera": camera,
                     "local_gap_id": "%(pct)ss_G%(pct)sd" %(pct)s (camera, index),
                     "confirmation_frame": frame,
                     "first_seen_frame": frame, "last_seen_frame": frame,
                     "normalized_confirmation_time": float(position),
                     "normalized_first_time": float(position),
                     "normalized_last_time": float(position),
                     "normalized_duration": 8.0, "max_confidence": 0.9,
                     "average_confidence": 0.85, "frame_count": 3})
    result = {
        "camera": camera, "status": "VALID",
        "video_info": {"fps": FPS, "total_frames": TOTAL_FRAMES,
                       "width": %(w)d, "height": %(h)d},
        "trimmed_info": {"fps": FPS, "total_frames": TRIMMED},
        "final_start_frame": REGION_START, "final_end_frame": REGION_END,
        "trimmed_total_frames": TRIMMED, "unique_gap_count": len(POSITIONS),
        "n_frames": TOTAL_FRAMES,
        "trimmed_video_path": str(VIDEO_PATHS[camera]) + ".trimmed.mp4",
        "normalized_timeline": pd.DataFrame(rows),
        "timeline_df": pd.DataFrame(
            [{"frame_id": i, "normalized_class": "wagon", "is_wagon": True}
             for i in range(TOTAL_FRAMES)]),
    }
    CAMERA_RESULTS[camera] = result
    return result
''' % {"fps": FPS, "total": TOTAL_FRAMES, "start": REGION_START,
       "end": REGION_END, "positions": GAP_POSITIONS, "w": WIDTH, "h": HEIGHT,
       "pct": "%"},

"reporting.py": ("def write_normalized_gap_timelines(p):\n"
                 "    open(str(p), 'w').close()\n"
                 "def write_camera_alignment_summary(p):\n"
                 "    open(str(p), 'w').close()\n"
                 "def write_global_gap_timeline(p):\n"
                 "    open(str(p), 'w').close()\n"),
"global_alignment.py": "GAP_TIMELINES = {}\nCAMERA_ALIGNMENTS = {}\n",
"wagon_mapping.py": "GLOBAL_WAGONS = []\nGLOBAL_WAGON_COUNT = 0\n",
"utils.py": "def banner(text):\n    pass\n",
"classification.py": "", "gap_detection.py": "", "gap_tracking.py": "",
"trimming.py": "", "snapshot_extraction.py": "", "pdf_report.py": "",
"visualization.py": "", "gap_annotation.py": "",
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
        if self._index >= TOTAL_FRAMES:
            return False, None
        frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
        frame[0, 0, 0] = self._index % 256
        self._index += 1
        return True, frame

    def release(self):
        self.released = True

    def get(self, prop):
        import cv2
        if prop == cv2.CAP_PROP_FRAME_COUNT:
            return float(TOTAL_FRAMES)
        if prop == cv2.CAP_PROP_FPS:
            return FPS
        if prop == cv2.CAP_PROP_FRAME_WIDTH:
            return float(WIDTH)
        if prop == cv2.CAP_PROP_FRAME_HEIGHT:
            return float(HEIGHT)
        return 0.0


@pytest.fixture
def stub_engine(tmp_path, monkeypatch):
    engine_dir = tmp_path / "global_wagon_app"
    engine_dir.mkdir()
    for name, source in STUB_ENGINE.items():
        (engine_dir / name).write_text(source, encoding="utf-8")
    monkeypatch.setenv("STUB_PROCESS_CALLS", str(tmp_path / "process.json"))
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
def wired(monkeypatch):
    """Fake capture + fake feature detectors that record every frame shown.

    Used for ASSEMBLY tests now: the materializer is what opens the video and
    the processors are what run the detectors.
    """
    import cv2

    FakeCapture.open_count = 0
    monkeypatch.setattr(cv2, "VideoCapture", FakeCapture)

    calls = {"door": [], "damage": [], "load": []}

    # A centred, high-confidence box that survives Batch's REAL gates on merit:
    # conf 0.9 >= door 0.68 and >= damage 0.55; area ratio 0.104 inside
    # [0.005, 0.40]; centre (0.47, 0.50) clear of every edge zone.
    BBOX = [100.0, 80.0, 200.0, 160.0]

    class _Arr:
        def __init__(self, value):
            self._value = value

        def cpu(self):
            return self

        def numpy(self):
            return np.asarray(self._value)

    class _Boxes:
        def __init__(self, bbox, conf, cls_id):
            self.xyxy = _Arr([bbox])
            self.conf = _Arr([conf])
            self.cls = _Arr([cls_id])

        def __len__(self):
            return 1

    class _Result:
        def __init__(self, boxes):
            self.boxes = boxes

    class _FakeYolo:
        """Callable like a YOLO model; records every frame it was shown."""

        def __init__(self, feature, names):
            self.feature = feature
            self.names = names

        def __call__(self, frame, **kwargs):
            index = int(frame[0, 0, 0])
            calls[self.feature].append(index)
            # Door alternates open/closed so multi-door survival is observable.
            cls_id = 1 if (self.feature == "door" and index % 2) else 0
            return [_Result(_Boxes(BBOX, 0.9, cls_id))]

    def _fake_load_yolo(path):
        name = os.path.basename(path or "")
        if "door" in name:
            return _FakeYolo("door", {0: "closed_door", 1: "open_door"})
        if "damage" in name:
            return _FakeYolo("damage", {0: "dent"})
        return _FakeYolo("load", {0: "loaded"})

    def _fake_run_classification(model, frame):
        calls["load"].append(int(frame[0, 0, 0]))
        return ("loaded", 0.9)

    from features import _common
    monkeypatch.setattr(_common, "load_yolo", _fake_load_yolo)
    monkeypatch.setattr(_common, "run_classification", _fake_run_classification)
    return calls


def _process(camera_id, *, workspace, stub_engine, inputs,
             features=("door", "damage", "load"), **kwargs):
    video_paths, models_dir = inputs
    return camera_runner.process_camera(
        camera_id=camera_id, video_path=video_paths[camera_id],
        workspace=str(workspace), repo_root=_REPO_ROOT,
        recon_models_dir=models_dir, feat_models_dir=models_dir,
        features=features, engine_dir=str(stub_engine), verbose=False,
        batch_key="archtest", **kwargs)


# =============================================================================
# 1. Mode routing  (contract UNCHANGED)
# =============================================================================

def test_batch_is_the_default_mode():
    assert mr.DEFAULT_MODE == mr.MODE_BATCH
    args = mr._build_parser().parse_args(["--local-only"])
    assert args.mode is None
    assert mr.resolve_mode(args.mode) == mr.MODE_BATCH


def test_mode_choices_and_env_override(monkeypatch):
    assert mr.MODES == ("batch", "sequential")
    assert mr.resolve_mode("sequential") == mr.MODE_SEQUENTIAL
    monkeypatch.setenv(mr.MODE_ENV_VAR, "sequential")
    assert mr.resolve_mode() == mr.MODE_SEQUENTIAL
    assert mr.resolve_mode("batch") == mr.MODE_BATCH
    with pytest.raises(ValueError):
        mr.resolve_mode("nope")


def _local_inputs(directory):
    for camera in C.ALL_CAMERAS:
        (directory / ("%s.mp4" % camera)).write_bytes(b"v")


def _outcome():
    class _O:
        final_status = "ok"
        report_pdf_path = None
        report_json_path = None
        camera_pdf_paths = {}
        processed_video_paths = {}
        error = None
    return _O()


def test_mode_batch_routes_to_the_batch_architecture(monkeypatch, tmp_path):
    called = []
    monkeypatch.setattr(mr, "process_batch",
                        lambda **kw: called.append("batch") or _outcome())
    monkeypatch.setattr(mr, "_run_sequential_local",
                        lambda **kw: called.append("sequential") or 0)
    _local_inputs(tmp_path)
    mr.run_local(local_inputs=str(tmp_path), batch_key="k",
                 workspace=str(tmp_path / "ws"),
                 recon_models_dir=str(tmp_path),
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
                 workspace=str(tmp_path / "ws"),
                 recon_models_dir=str(tmp_path),
                 feat_models_dir=str(tmp_path), mode="sequential")
    assert called == ["sequential"]


def test_sequential_accepts_partial_cameras_batch_still_requires_four(
        monkeypatch, tmp_path, capsys):
    (tmp_path / "RIGHT_UP.mp4").write_bytes(b"v")
    seen = {}
    monkeypatch.setattr(mr, "_run_sequential_local",
                        lambda **kw: seen.update(kw) or 0)
    assert mr.run_local(local_inputs=str(tmp_path), batch_key="k",
                        workspace=str(tmp_path / "ws"),
                        recon_models_dir=str(tmp_path),
                        feat_models_dir=str(tmp_path), mode="sequential") == 0
    assert list(seen["video_paths"]) == [C.CAMERA_RIGHT_UP]

    assert mr.run_local(local_inputs=str(tmp_path), batch_key="k",
                        workspace=str(tmp_path / "ws"),
                        recon_models_dir=str(tmp_path),
                        feat_models_dir=str(tmp_path), mode="batch") == 2
    assert "missing videos" in capsys.readouterr().err


# =============================================================================
# 2. The camera stage delegates to the ENGINE  (contract CHANGED, deliberately)
# =============================================================================

def test_camera_stage_calls_the_engines_own_process_camera(stub_engine, inputs,
                                                           tmp_path):
    """Batch's Stage-1 numbers come from this function, so Sequential runs it.

    Replaces test_exactly_one_decode_per_camera and
    test_gap_receives_every_decoded_frame: the decode and the gap detection are
    no longer Sequential's to count, they are the engine's, and running the
    engine's function is what makes the gap input pixels equal to Batch's.
    """
    _process(C.CAMERA_RIGHT_UP, workspace=tmp_path / "ws",
             stub_engine=stub_engine, inputs=inputs)
    marker = os.environ["STUB_PROCESS_CALLS"]
    assert os.path.isfile(marker), "the engine's process_camera was never called"
    assert json.load(open(marker, encoding="utf-8")) == ["right_up"]


def test_camera_runner_performs_no_decode_and_no_inference_itself():
    """Stronger than the old 'exactly one VideoCapture site': there are none.

    Every decode and every model call now happens inside the engine's own
    per-camera pipeline. Any reappearance here would be a mirrored algorithm.
    """
    code = _code("sequential/camera_runner.py")
    for banned in ("cv2.VideoCapture", "detect_gaps_in_frame", ".predict(",
                   "load_yolo", "GapTracker", "run_detection",
                   "run_classification", "find_reliable_wagon_start"):
        assert banned not in code, (
            "the camera stage is doing its own %r again" % banned)
    assert "camera_pipeline.process_camera" in code


def test_camera_stage_runs_no_feature_inference(stub_engine, inputs, tmp_path):
    """Door/Damage/Load moved to assembly: Batch's frame set needs the roster."""
    workspace = tmp_path / "ws"
    _process(C.CAMERA_RIGHT_UP, workspace=workspace, stub_engine=stub_engine,
             inputs=inputs)
    camera_evidence = ev.load_evidence(str(workspace), C.CAMERA_RIGHT_UP)
    assert camera_evidence.observations == []
    assert camera_evidence.feature_config["applied_in"] == "global_assembly"
    assert camera_evidence.feature_config["features"] == ["door", "damage",
                                                          "load"]


def test_the_complete_engine_registry_is_loaded(stub_engine, inputs, tmp_path):
    """load_all_models validates the WHOLE camera mapping, so all five go in.

    A regression here raised, on EC2:
      RuntimeError: Classification model(s) required by the camera mapping are
      not loaded: ['side']
    """
    _process(C.CAMERA_LEFT_UP_TOP, workspace=tmp_path / "ws",
             stub_engine=stub_engine, inputs=inputs)
    # reaching here without RuntimeError proves both classification keys and
    # all three gap keys were supplied


def test_model_paths_reach_the_engine_as_path_objects(stub_engine, inputs,
                                                      tmp_path):
    """The engine calls .stat() on them (the 3dc848c EC2 crash)."""
    _video_paths, models_dir = inputs
    registries = camera_runner.engine_model_registries(
        {slot: os.path.join(models_dir, names[0])
         for slot, names in gc_runner.MODEL_SLOTS.items()})
    from pathlib import Path
    for registry in registries:
        for value in registry.values():
            assert isinstance(value, Path), value
            value.stat()


# =============================================================================
# 3. Persistence, sealing, resume, release
# =============================================================================

def test_camera_persists_evidence_and_reports_then_seals(stub_engine, inputs,
                                                         tmp_path):
    workspace = tmp_path / "ws"
    result = _process(C.CAMERA_RIGHT_UP, workspace=workspace,
                      stub_engine=stub_engine, inputs=inputs)

    assert os.path.isfile(result.evidence_path)
    assert os.path.isfile(result.seal_path)
    assert os.path.isfile(result.report_paths["json_path"])
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


def test_persisted_evidence_is_lossless_for_every_field_assembly_reads(
        stub_engine, inputs, tmp_path):
    """The whole point of the split: nothing the global half needs is dropped.

    Replaces test_evidence_contains_what_assembly_needs, which checked the old
    hand-rolled evidence shape. The field list here is not a guess: it is every
    key `restore_camera_results` rebuilds, which in turn is every key the
    engine's global half and `gc_runner._harvest` read.
    """
    workspace = tmp_path / "ws"
    _process(C.CAMERA_RIGHT_UP, workspace=workspace, stub_engine=stub_engine,
             inputs=inputs)
    record = ev.load_evidence(str(workspace),
                              C.CAMERA_RIGHT_UP).engine_result

    # read by ga.build_normalized_timelines
    assert record["normalized_timeline"]
    assert record["trimmed_total_frames"] == TRIMMED_FRAMES
    assert record["trimmed_info"]["fps"] == FPS
    assert record["status"] == "VALID"
    # additionally read by gc_runner._harvest
    assert record["video_info"]["fps"] == FPS
    assert record["video_info"]["total_frames"] == TOTAL_FRAMES
    assert record["final_start_frame"] == REGION_START
    assert record["final_end_frame"] == REGION_END
    assert record["unique_gap_count"] == len(GAP_POSITIONS)
    assert len(record["classification_timeline"]) == TOTAL_FRAMES
    # every column the engine emits on the normalized timeline survives
    row = record["normalized_timeline"][0]
    for column in ("local_gap_id", "confirmation_frame",
                   "normalized_confirmation_time", "normalized_duration",
                   "max_confidence"):
        assert column in row, "lost column %r" % column


def test_restore_is_exactly_inverse_to_persist(stub_engine, inputs, tmp_path):
    """Persist -> restore must reproduce the engine result field for field."""
    workspace = tmp_path / "ws"
    _process(C.CAMERA_RIGHT_UP, workspace=workspace, stub_engine=stub_engine,
             inputs=inputs)
    camera_evidence = ev.load_evidence(str(workspace), C.CAMERA_RIGHT_UP)
    restored = global_assembly.restore_camera_results(
        {C.CAMERA_RIGHT_UP: camera_evidence})["right_up"]
    record = camera_evidence.engine_result

    for key in ("status", "final_start_frame", "final_end_frame",
                "trimmed_total_frames", "unique_gap_count", "n_frames"):
        assert restored[key] == record[key], key
    assert restored["video_info"] == record["video_info"]
    assert restored["trimmed_info"] == record["trimmed_info"]
    assert (restored["normalized_timeline"].to_dict("records")
            == record["normalized_timeline"])
    assert len(restored["timeline_df"]) == len(record["classification_timeline"])


def test_camera_evidence_has_no_canonical_wagon_ids(stub_engine, inputs,
                                                    tmp_path):
    workspace = tmp_path / "ws"
    result = _process(C.CAMERA_RIGHT_UP, workspace=workspace,
                      stub_engine=stub_engine, inputs=inputs)
    for path in (result.evidence_path, result.seal_path,
                 result.report_paths["json_path"]):
        with open(path, encoding="utf-8") as handle:
            assert not re.search(r"\bGW_\d+\b", handle.read()), path

    camera_evidence = ev.load_evidence(str(workspace), C.CAMERA_RIGHT_UP)
    assert camera_evidence.segments
    for segment in camera_evidence.segments:
        assert segment["segment_id"].startswith(C.CAMERA_RIGHT_UP + "_SEG_")
        assert segment["canonical"] is False


def test_gap_frames_are_shifted_into_original_video_numbering(
        stub_engine, inputs, tmp_path):
    """The engine numbers gaps inside the trimmed clip; consumers want original.

    This is a real consequence of running the engine's trimming: the frame
    indices coming back are clip-relative, and everything outside the engine
    (camera report, snapshots) speaks original frames.
    """
    workspace = tmp_path / "ws"
    _process(C.CAMERA_RIGHT_UP, workspace=workspace, stub_engine=stub_engine,
             inputs=inputs)
    camera_evidence = ev.load_evidence(str(workspace), C.CAMERA_RIGHT_UP)
    assert camera_evidence.gaps
    for gap in camera_evidence.gaps:
        assert REGION_START <= gap.confirmation_frame <= REGION_END
    assert camera_evidence.gaps[0].confirmation_frame == REGION_START
    assert camera_evidence.gaps[-1].confirmation_frame == REGION_END


def test_matching_seal_resumes_without_reprocessing(stub_engine, inputs,
                                                    tmp_path):
    workspace = tmp_path / "ws"
    _process(C.CAMERA_RIGHT_UP, workspace=workspace, stub_engine=stub_engine,
             inputs=inputs)
    marker = os.environ["STUB_PROCESS_CALLS"]
    first = json.load(open(marker, encoding="utf-8"))

    second = _process(C.CAMERA_RIGHT_UP, workspace=workspace,
                      stub_engine=stub_engine, inputs=inputs)
    assert second.reused is True
    assert json.load(open(marker, encoding="utf-8")) == first, (
        "resume re-ran the engine pipeline")


def test_stale_evidence_is_reprocessed_and_the_reason_is_reported(
        stub_engine, inputs, tmp_path):
    workspace = tmp_path / "ws"
    video_paths, _models = inputs
    _process(C.CAMERA_RIGHT_UP, workspace=workspace, stub_engine=stub_engine,
             inputs=inputs)
    with open(video_paths[C.CAMERA_RIGHT_UP], "wb") as handle:
        handle.write(b"different content")
    os.utime(video_paths[C.CAMERA_RIGHT_UP], (1, 1))
    again = _process(C.CAMERA_RIGHT_UP, workspace=workspace,
                     stub_engine=stub_engine, inputs=inputs)
    assert again.reused is False
    assert "video changed" in again.reason


def test_force_ignores_a_matching_seal(stub_engine, inputs, tmp_path):
    workspace = tmp_path / "ws"
    _process(C.CAMERA_RIGHT_UP, workspace=workspace, stub_engine=stub_engine,
             inputs=inputs)
    forced = _process(C.CAMERA_RIGHT_UP, workspace=workspace,
                      stub_engine=stub_engine, inputs=inputs, force=True)
    assert forced.reused is False


def test_resources_are_released_after_sealing(stub_engine, inputs, tmp_path):
    from features import _common

    _common._MODEL_CACHE["stale"] = object()
    _process(C.CAMERA_RIGHT_UP, workspace=tmp_path / "ws",
             stub_engine=stub_engine, inputs=inputs)
    assert _common._MODEL_CACHE == {}


def test_engine_session_leaves_nothing_behind(stub_engine, inputs, tmp_path):
    _process(C.CAMERA_RIGHT_UP, workspace=tmp_path / "ws",
             stub_engine=stub_engine, inputs=inputs)
    for name in gc_runner.ENGINE_MODULES:
        module = sys.modules.get(name)
        origin = getattr(module, "__file__", None) if module else None
        if origin:
            assert not os.path.abspath(origin).startswith(str(stub_engine))
    import reporting
    assert os.path.abspath(reporting.__file__).startswith(_REPO_ROOT)


def test_camera_processing_creates_no_canonical_roster(stub_engine, inputs,
                                                       tmp_path):
    workspace = tmp_path / "ws"
    _process(C.CAMERA_RIGHT_UP, workspace=workspace, stub_engine=stub_engine,
             inputs=inputs)
    assert not os.path.exists(os.path.join(
        str(workspace), "global_state", "global_train_state.json"))
    assert not os.path.exists(os.path.join(
        str(workspace), ev.COMBINED_DIRNAME, "combined_train_report.json"))


def test_a_single_camera_report_is_valid_on_its_own(stub_engine, inputs,
                                                    tmp_path):
    result = _process(C.CAMERA_RIGHT_UP, workspace=tmp_path / "ws",
                      stub_engine=stub_engine, inputs=inputs)
    with open(result.report_paths["json_path"], encoding="utf-8") as handle:
        document = json.load(handle)
    assert document["report_type"] == "single_camera"
    assert document["canonical"] is False
    assert "NOT canonical" in document["disclaimer"]
    assert document["camera_id"] == C.CAMERA_RIGHT_UP
    assert document["local_gaps"]["count"] == len(GAP_POSITIONS)

    try:
        import reportlab                                        # noqa: F401
    except ImportError:
        pytest.skip("reportlab not installed")
    assert result.report_paths["pdf_path"]
    assert os.path.getsize(result.report_paths["pdf_path"]) > 800


# =============================================================================
# 4. Global Assembly -- wiring
# =============================================================================

def test_assembly_calls_the_engines_own_global_half():
    import inspect

    source = inspect.getsource(global_assembly.run_engine_global_half)
    for call in ("build_normalized_timelines", "select_master_camera",
                 "validate_temporal_ordering", "set_master_camera",
                 "match_all_cameras", "report_alignment_mappings",
                 "recover_missing_gaps", "collect_unmatched_extras",
                 "build_global_gap_timeline", "build_global_wagon_timeline"):
        assert call in source, "the engine's %s is not called" % call


def test_assembly_then_runs_batchs_own_stages():
    """Every stage after the engine is Batch's own implementation."""
    import inspect

    source = inspect.getsource(global_assembly.assemble)
    for call in ("gc_runner._harvest", "adapter.write_documents",
                 "wagon_cache_builder.build", "wagon_state_builder.build",
                 "camera_reports.build_all", "combined_train_report.build"):
        assert call in source, "Batch's %s is not reused" % call


def test_assembly_runs_batchs_processors_with_batchs_strides():
    """Where the 3/3/2 strides live now (they moved, they did not vanish)."""
    import inspect

    source = inspect.getsource(global_assembly._run_features)
    assert 'for name in ("load", "door", "ocr", "damage")' in source, (
        "LOAD must complete before DAMAGE: damage reads the load result")
    assert "load_feature_runner" in source, (
        "the processors must be Batch's, resolved through Batch's registry")
    # whitespace-insensitive: the real source wraps these across lines
    flat = " ".join(source.split())
    for needle in ('"door_sample_stride", 3', '"damage_sample_stride", 3',
                   '"load_sample_stride", 2'):
        assert needle in flat, "Batch's default stride changed: %s" % needle


def test_assembly_is_not_batch_comparable_without_every_camera(tmp_path):
    """The master is unknown until every camera is counted."""
    ready, sealed, missing, reason = global_assembly.readiness(str(tmp_path))
    assert ready is False
    assert sealed == []
    assert set(missing) == set(C.ALL_CAMERAS)
    assert "not Batch-comparable" in reason


def test_assembly_refuses_to_fabricate_a_train(tmp_path):
    result = global_assembly.assemble(
        workspace=str(tmp_path), repo_root=_REPO_ROOT, batch_key="k",
        verbose=False)
    assert result.ready is False
    assert result.report_json_path is None
    assert not os.path.exists(os.path.join(str(tmp_path), ev.COMBINED_DIRNAME,
                                           "combined_train_report.json"))


def test_master_camera_is_chosen_the_way_batch_chooses_it():
    """Batch has NO fixed master: the engine picks the max-unique-gaps camera.

    An earlier version of this test asserted a fixed RIGHT_UP, which was the
    divergence, not the contract. The global gap count IS the master's gap
    count, so a wrong master means a wrong wagon count.
    """
    import inspect

    source = inspect.getsource(global_assembly.run_engine_global_half)
    assert "select_master_camera()" in source
    assert "set_master_camera()" in source
    assert global_assembly.required_cameras() == tuple(C.ALL_CAMERAS)
    assert "C.MASTER_CAMERA" not in inspect.getsource(global_assembly.assemble)


def test_assembly_refuses_evidence_without_an_engine_record(tmp_path):
    """Evidence from an older Sequential build must be reprocessed, not guessed."""
    camera_evidence = ev.CameraEvidence(camera_id=C.CAMERA_RIGHT_UP)
    with pytest.raises(global_assembly.AssemblyNotReady) as excinfo:
        global_assembly.restore_camera_results(
            {C.CAMERA_RIGHT_UP: camera_evidence})
    assert "--force-cameras" in str(excinfo.value)


# =============================================================================
# 5. Global Assembly -- against the REAL engine
# =============================================================================

@needs_engine
def test_real_engine_forms_the_canonical_roster_from_persisted_records(
        tmp_path, capsys):
    """Five persisted gaps -> five global gaps -> four wagons, GW_1..GW_4."""
    workspace = str(tmp_path / "ws")
    F.seal_all(workspace)
    snapshot = F.run_global_half(
        REAL_ENGINE, global_assembly.restore_camera_results(
            {c: ev.load_evidence(workspace, c) for c in C.ALL_CAMERAS}),
        str(tmp_path / "engine_out"))
    capsys.readouterr()

    assert snapshot["global_gap_count"] == len(F.CANONICAL_POSITIONS)
    assert snapshot["global_wagon_count"] == snapshot["global_gap_count"] - 1
    assert snapshot["master_camera"] == "right_up"
    assert "most confirmed unique gaps" in snapshot["master_reason"]


@needs_engine
def test_real_engine_master_follows_the_gap_count_not_the_camera_name(
        tmp_path, capsys):
    """Give a non-first camera the most gaps and it must become master."""
    workspace = str(tmp_path / "ws")
    extra = list(F.CANONICAL_POSITIONS) + [875.0]
    F.seal_all(workspace, positions_by_camera={C.CAMERA_LEFT_UP_TOP: extra})
    snapshot = F.run_global_half(
        REAL_ENGINE, global_assembly.restore_camera_results(
            {c: ev.load_evidence(workspace, c) for c in C.ALL_CAMERAS}),
        str(tmp_path / "engine_out"))
    capsys.readouterr()

    assert snapshot["master_camera"] == "left_up_top"
    assert snapshot["global_gap_count"] == len(extra)
    assert snapshot["global_wagon_count"] == len(extra) - 1


@needs_engine
def test_real_engine_extra_gap_on_a_support_camera_creates_no_wagon(tmp_path,
                                                                    capsys):
    """Only the master defines the count; a support camera's extra is diagnostic.

    LEFT_UP gets an extra detection but is kept BELOW the master's gap count, so
    it cannot win master selection -- which is the situation this protects.
    """
    workspace = str(tmp_path / "ws")
    positions = list(F.CANONICAL_POSITIONS)
    support = positions[:-1] + [690.0]          # same count, one displaced
    F.seal_all(workspace, positions_by_camera={C.CAMERA_LEFT_UP: support})
    snapshot = F.run_global_half(
        REAL_ENGINE, global_assembly.restore_camera_results(
            {c: ev.load_evidence(workspace, c) for c in C.ALL_CAMERAS}),
        str(tmp_path / "engine_out"))
    capsys.readouterr()

    assert snapshot["global_gap_count"] == len(positions)
    assert snapshot["global_wagon_count"] == len(positions) - 1


@needs_engine
def test_real_engine_recovers_a_gap_a_support_camera_missed(tmp_path, capsys):
    """A missing support detection must not shrink the canonical train."""
    workspace = str(tmp_path / "ws")
    positions = list(F.CANONICAL_POSITIONS)
    F.seal_all(workspace,
               positions_by_camera={C.CAMERA_LEFT_UP: positions[:-1]})
    snapshot = F.run_global_half(
        REAL_ENGINE, global_assembly.restore_camera_results(
            {c: ev.load_evidence(workspace, c) for c in C.ALL_CAMERAS}),
        str(tmp_path / "engine_out"))
    capsys.readouterr()

    assert snapshot["global_gap_count"] == len(positions)
    assert snapshot["global_wagon_count"] == len(positions) - 1
    assert snapshot["camera_gap_counts"]["left_up"] == len(positions) - 1


@needs_engine
def test_real_engine_reports_no_reversal_for_co_ordered_cameras(tmp_path,
                                                                capsys):
    """The reversal test must run and must not fire on co-ordered timelines.

    A camera is never reversed for being a left or a right camera; only the
    alignment error may decide that.
    """
    workspace = str(tmp_path / "ws")
    F.seal_all(workspace)
    snapshot = F.run_global_half(
        REAL_ENGINE, global_assembly.restore_camera_results(
            {c: ev.load_evidence(workspace, c) for c in C.ALL_CAMERAS}),
        str(tmp_path / "engine_out"))
    capsys.readouterr()

    assert snapshot["alignments"], "no camera was aligned"
    for camera, mapping in snapshot["alignments"].items():
        assert mapping["reversed"] is False, camera
        assert mapping["scale"] == pytest.approx(1.0, abs=1e-9)
        assert mapping["unmatched"] == []


# =============================================================================
# 6. Sequential runner orchestration
# =============================================================================

def test_camera_order_is_deterministic_and_configuration_driven():
    paths = {camera: "x" for camera in reversed(C.ALL_CAMERAS)}
    assert sequential_runner.camera_order(paths) == list(C.ALL_CAMERAS)


def _weights(tmp_path):
    directory = tmp_path / "recon_models"
    directory.mkdir(exist_ok=True)
    for filenames in gc_runner.MODEL_SLOTS.values():
        (directory / filenames[0]).write_bytes(b"w")
    return str(directory)


def test_runner_processes_cameras_in_order_then_assembles(monkeypatch,
                                                          tmp_path):
    events = []
    monkeypatch.setattr(
        camera_runner, "process_camera",
        lambda **kw: events.append(("camera", kw["camera_id"]))
        or camera_runner.CameraRunResult(camera_id=kw["camera_id"],
                                         status=ev.STATUS_SEALED))
    monkeypatch.setattr(
        global_assembly, "assemble",
        lambda **kw: events.append(("assembly", None))
        or global_assembly.AssemblyResult(ready=True))

    sequential_runner.run_sequential(
        video_paths={c: "v" for c in C.ALL_CAMERAS},
        workspace=str(tmp_path), repo_root=_REPO_ROOT,
        recon_models_dir=_weights(tmp_path), feat_models_dir=str(tmp_path),
        features=("door", "load", "damage"), batch_key="k", verbose=False)

    assert events == ([("camera", c) for c in C.ALL_CAMERAS]
                      + [("assembly", None)])
    assert len([e for e in events if e[0] == "assembly"]) == 1


def test_a_failing_camera_does_not_stop_the_others(monkeypatch, tmp_path):
    def _fake(**kwargs):
        if kwargs["camera_id"] == C.CAMERA_LEFT_UP:
            raise RuntimeError("boom")
        return camera_runner.CameraRunResult(camera_id=kwargs["camera_id"],
                                             status=ev.STATUS_SEALED)

    monkeypatch.setattr(camera_runner, "process_camera", _fake)
    monkeypatch.setattr(global_assembly, "assemble",
                        lambda **kw: global_assembly.AssemblyResult(True))
    outcome = sequential_runner.run_sequential(
        video_paths={c: "v" for c in C.ALL_CAMERAS},
        workspace=str(tmp_path), repo_root=_REPO_ROOT,
        recon_models_dir=_weights(tmp_path), feat_models_dir=str(tmp_path),
        features=("door",), batch_key="k", verbose=False)
    assert outcome.failed_cameras == [C.CAMERA_LEFT_UP]
    assert len(outcome.sealed_cameras) == 3


def test_skip_assembly_stops_after_camera_reports(monkeypatch, tmp_path):
    monkeypatch.setattr(camera_runner, "process_camera",
                        lambda **kw: camera_runner.CameraRunResult(
                            camera_id=kw["camera_id"],
                            status=ev.STATUS_SEALED))
    called = []
    monkeypatch.setattr(global_assembly, "assemble",
                        lambda **kw: called.append(1))
    outcome = sequential_runner.run_sequential(
        video_paths={C.CAMERA_RIGHT_UP: "v"}, workspace=str(tmp_path),
        repo_root=_REPO_ROOT, recon_models_dir=_weights(tmp_path),
        feat_models_dir=str(tmp_path), features=("door",), batch_key="k",
        skip_assembly=True, verbose=False)
    assert called == []
    assert outcome.assembly is None


def test_single_camera_run_reports_then_declines_assembly(stub_engine, inputs,
                                                          tmp_path):
    video_paths, models_dir = inputs
    workspace = tmp_path / "ws"
    outcome = sequential_runner.run_sequential(
        video_paths={C.CAMERA_LEFT_UP: video_paths[C.CAMERA_LEFT_UP]},
        workspace=str(workspace), repo_root=_REPO_ROOT,
        recon_models_dir=models_dir, feat_models_dir=models_dir,
        features=("door", "load", "damage"), batch_key="solo",
        engine_dir=str(stub_engine), verbose=False)

    assert outcome.sealed_cameras == [C.CAMERA_LEFT_UP]
    assert os.path.isfile(outcome.cameras[0].report_paths["json_path"])
    assert outcome.assembly is not None
    assert outcome.assembly.ready is False
    assert "not Batch-comparable" in outcome.assembly.reason


# =============================================================================
# 7. Static architecture audit
# =============================================================================

def _code(relative):
    """Source with docstrings and comments stripped, for auditing real calls."""
    text = open(os.path.join(_REPO_ROOT, relative), encoding="utf-8").read()
    without_docstrings = re.sub(r'"""(?:.|\n)*?"""', "", text)
    return "\n".join(line for line in without_docstrings.splitlines()
                     if not line.strip().startswith("#"))


def test_camera_local_code_builds_no_canonical_roster():
    for relative in ("sequential/camera_runner.py",
                     "sequential/camera_report.py"):
        code = _code(relative)
        assert not re.search(r'"GW_|\bGW_\d', code), relative
        for banned in ("build_global_train_state_document", "write_documents",
                       "verify_roster_integrity"):
            assert banned not in code, "%s: %s" % (relative, banned)


def test_only_global_assembly_writes_the_combined_report():
    assert "combined_train_report" in _code("sequential/global_assembly.py")
    for relative in ("sequential/camera_runner.py",
                     "sequential/camera_report.py",
                     "sequential/runner.py"):
        assert "combined_train_report" not in _code(relative), relative


def test_sequential_holds_no_mirrored_algorithm():
    """Every mirrored implementation is gone; the engine or Batch does the work.

    Replaces test_global_assembly_performs_no_inference_and_no_decode. Assembly
    now DOES infer -- through Batch's processors -- so the meaningful audit is
    no longer "does it infer" but "does it reimplement".
    """
    assembly = _code("sequential/global_assembly.py")
    for banned in ("monotonic_gap_match", "robust_linear_fit",
                   "_estimate_one_direction", "EvidenceAggregator",
                   "_filter_detections_for_top", "_LOADED_RATIO_THRESHOLD",
                   "_door_evidence_from_groups", "_pick_side_state",
                   "SnapshotStore", "classification_adapter"):
        assert banned not in assembly, (
            "assembly re-implements %r instead of calling Batch/the engine"
            % banned)
    assert not os.path.exists(os.path.join(_REPO_ROOT, "sequential",
                                           "classification_adapter.py")), (
        "the mirrored classification adapter must stay deleted")


def test_master_runner_does_not_duplicate_detector_logic():
    code = _code("orchestrator/master_runner.py")
    for banned in ("VideoCapture", "detect_gaps_in_frame", "load_yolo",
                   ".predict("):
        assert banned not in code, banned


def test_global_count_ec2_is_not_vendored():
    assert not os.path.exists(os.path.join(_REPO_ROOT,
                                           "global_wagon_pipeline.py"))
    for root, dirs, files in os.walk(os.path.join(_REPO_ROOT, "sequential")):
        dirs[:] = [d for d in dirs if d != "__pycache__"]
        assert "global_wagon_pipeline.py" not in files
