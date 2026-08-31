"""Stage 1 must hand the engine the TYPES its API expects.

Regression for the first real EC2 failure:

    global_count_ec2/models.py:171
        print("  size              :", human_size(path.stat().st_size))
    AttributeError: 'str' object has no attribute 'stat'

`load_all_models()` calls `.stat()` directly on the values it is handed. The
classification loader re-wraps its argument with `Path()` first, so a plain
`str` survives there and only blows up on the GAP models -- after both
classification models have already loaded, several minutes into a run.

Rather than assert on a string, this drives the real
`global_counting.runner.run()` against a STUB engine that records the exact
types it receives. The stub is a real directory of real modules loaded through
the real `engine_session()`, so the test needs no weights, no videos and no
installed engine -- it runs anywhere, including CI and a fresh EC2 box.

    python -m pytest tests/test_stage1_engine_call_contract.py -q
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

_TEST_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_TEST_DIR)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from core import constants as C
from global_counting import runner as gc_runner


# -----------------------------------------------------------------------------
# A stub engine: same module names, same call surface, no models.
# -----------------------------------------------------------------------------
# It writes what it was called with to $STUB_ENGINE_RECORD as JSON, so the test
# can assert on the ACTUAL arguments the production runner passed.

STUB_MODULES = {

"config.py": '''
import json, os
NORMALIZED_TIMELINE_SCALE = 1000.0
GENERATE_TRIM_DEBUG_VIDEO = True
GENERATE_GAP_ANNOTATED_VIDEO = True
_OVERRIDABLE = ("GENERATE_TRIM_DEBUG_VIDEO", "GENERATE_GAP_ANNOTATED_VIDEO",
                "MAX_FRAMES_TO_PROCESS")

def _record(key, value):
    path = os.environ["STUB_ENGINE_RECORD"]
    doc = {}
    if os.path.isfile(path):
        with open(path, encoding="utf-8") as handle:
            doc = json.load(handle)
    doc[key] = value
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(doc, handle)

def apply_overrides(**overrides):
    applied = {}
    for name, value in overrides.items():
        if value is None:
            continue
        if name not in _OVERRIDABLE:
            raise KeyError("%r is not an overridable setting" % name)
        globals()[name] = value
        applied[name] = value
    _record("overrides", {k: repr(v) for k, v in applied.items()})
    return applied
''',

"camera_map.py": '''
CAMERAS = ["right_up", "left_up", "right_up_top", "left_up_top"]
CAMERA_LABELS = {c: c.upper() for c in CAMERAS}
CLASSIFICATION_MODEL_FILENAMES = {"side": ["side_classification.pt"],
                                  "top": ["top_classification.pt"]}
GAP_MODEL_FILENAMES = {"right": ["right_up_wagon_gap.pt"],
                       "left": ["left_up_wagon_gap.pt"],
                       "top": ["top_gap.pt"]}
''',

"io_paths.py": '''
import os
from pathlib import Path
from camera_map import CAMERAS

VIDEO_PATHS = {}
OUTPUT_DIR = None

def resolve_inputs(video_arguments, model_arguments, input_dir=None, model_dir=None):
    """Mirrors the real engine: explicit paths become Path objects."""
    VIDEO_PATHS.clear()
    for camera in CAMERAS:
        VIDEO_PATHS[camera] = Path(video_arguments[camera]).expanduser()
    classification, gap = {}, {}
    for key, value in model_arguments.items():
        if key.startswith("classification_"):
            classification[key[len("classification_"):]] = Path(value).expanduser()
        elif key.startswith("gap_"):
            gap[key[len("gap_"):]] = Path(value).expanduser()
    return {"videos": dict(VIDEO_PATHS), "classification": classification,
            "gap": gap}

def prepare_output_dirs(output_dir):
    global OUTPUT_DIR
    OUTPUT_DIR = output_dir
    root = Path(output_dir) / "global"
    root.mkdir(parents=True, exist_ok=True)
    return {name: root / (name + ".csv") for name in (
        "normalized_gap_timelines", "camera_alignment_summary",
        "global_gap_timeline", "unmatched_extra_detections",
        "global_wagon_timeline", "global_wagon_snapshots")}
''',

"models.py": '''
import json, os
CLASSIFICATION_MODELS, GAP_MODELS = {}, {}

def _record(key, value):
    path = os.environ["STUB_ENGINE_RECORD"]
    doc = {}
    if os.path.isfile(path):
        with open(path, encoding="utf-8") as handle:
            doc = json.load(handle)
    doc[key] = value
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(doc, handle)

def load_all_models(classification_paths, gap_paths):
    """Record the TYPE of every value, then do what the engine does: .stat()."""
    _record("classification_types",
            {k: type(v).__name__ for k, v in classification_paths.items()})
    _record("gap_types", {k: type(v).__name__ for k, v in gap_paths.items()})
    # This is the exact operation that failed on EC2. A str has no .stat().
    for mapping in (classification_paths, gap_paths):
        for key, value in mapping.items():
            value.stat()
    _record("load_all_models_completed", True)
    return CLASSIFICATION_MODELS, GAP_MODELS

def build_class_maps():
    _record("build_class_maps_completed", True)
''',

"camera_pipeline.py": '''
from camera_map import CAMERAS
CAMERA_RESULTS = {}

def process_all_cameras(force=False):
    for index, camera in enumerate(CAMERAS):
        CAMERA_RESULTS[camera] = {
            "camera": camera,
            "status": "VALID",
            "video_info": {"fps": 15.0, "total_frames": 900},
            "n_frames": 900,
            "final_start_frame": 100 + index,
            "final_end_frame": 800,
            "trimmed_total_frames": 700,
            "unique_gap_count": 3,
            "timeline_df": None,
        }
    return CAMERA_RESULTS
''',

"global_alignment.py": '''
from camera_map import CAMERAS

MASTER_CAMERA = None
GLOBAL_GAP_COUNT = 0
CAMERA_ALIGNMENTS = {}
GLOBAL_GAP_STATUS = {}
MASTER_POSITIONS = [0.0, 500.0, 1000.0]
STATUS_DETECTED, STATUS_RECOVERED, STATUS_UNMATCHED = (
    "DETECTED", "RECOVERED", "UNMATCHED")

class _Alignment(object):
    def __init__(self, camera):
        self.camera = camera
        self.status = "RESOLVED"
        self.is_reversed = camera == "left_up_top"
        self.scale = -1.0 if self.is_reversed else 1.0
        self.offset = 0.0
        self.matches = [1, 2, 3]

def build_normalized_timelines(camera_results):
    return camera_results

def select_master_camera():
    global MASTER_CAMERA
    MASTER_CAMERA = "right_up_top"
    return MASTER_CAMERA

def validate_temporal_ordering():
    return True

def set_master_camera():
    for camera in CAMERAS:
        CAMERA_ALIGNMENTS[camera] = _Alignment(camera)

def match_all_cameras(verbose=True):
    return CAMERA_ALIGNMENTS

def report_alignment_mappings():
    return None

def recover_missing_gaps():
    return None

def collect_unmatched_extras(output_path=None):
    if output_path is not None:
        open(str(output_path), "w").close()

def build_global_gap_timeline():
    global GLOBAL_GAP_COUNT
    GLOBAL_GAP_COUNT = 3
    return GLOBAL_GAP_COUNT
''',

"reporting.py": '''
def _touch(path):
    open(str(path), "w").close()

def write_normalized_gap_timelines(path): _touch(path)
def write_camera_alignment_summary(path): _touch(path)
def write_global_gap_timeline(path): _touch(path)
''',

"wagon_mapping.py": '''
from camera_map import CAMERAS
import global_alignment as ga

GLOBAL_WAGONS = []
GLOBAL_WAGON_COUNT = 0

def build_global_wagon_timeline():
    global GLOBAL_WAGONS, GLOBAL_WAGON_COUNT
    GLOBAL_WAGON_COUNT = max(0, int(ga.GLOBAL_GAP_COUNT) - 1)
    GLOBAL_WAGONS = []
    for index in range(GLOBAL_WAGON_COUNT):
        cameras = {}
        for camera in CAMERAS:
            reverse = camera == "left_up_top"
            low, high = index * 200, (index + 1) * 200
            if reverse:
                low, high = 600 - high, 600 - low
            cameras[camera] = {
                "start_frame": low, "end_frame": high,
                "start_position": float(index * 500),
                "end_position": float((index + 1) * 500),
                "center_position": 0.0, "center_frame": 0,
                "start_source": "DETECTED", "end_source": "DETECTED",
                "interval_status": "DETECTED",
                "reversed": reverse,
            }
        GLOBAL_WAGONS.append({
            "global_wagon_id": "GW_%d" % (index + 1),
            "wagon_number": index + 1,
            "global_start_position_1000": float(index * 500),
            "global_end_position_1000": float((index + 1) * 500),
            "_cameras": cameras,
        })
    return None, GLOBAL_WAGONS

def write_global_wagon_timeline_csv(path):
    open(str(path), "w").close()
''',

# Present so locate_engine recognises the directory as the engine.
"global_wagon_pipeline.py": "def run(args):\n    return 0\n",
}


@pytest.fixture
def stub_engine(tmp_path, monkeypatch):
    """A directory that behaves like the engine and records its arguments."""
    engine_dir = tmp_path / "global_wagon_app"
    engine_dir.mkdir()
    for name, source in STUB_MODULES.items():
        (engine_dir / name).write_text(source, encoding="utf-8")

    record_path = tmp_path / "engine_calls.json"
    monkeypatch.setenv("STUB_ENGINE_RECORD", str(record_path))
    return engine_dir, record_path


@pytest.fixture
def inputs(tmp_path):
    """Four videos and five weights, real files so `.stat()` can succeed."""
    videos_dir = tmp_path / "videos"
    models_dir = tmp_path / "models"
    videos_dir.mkdir()
    models_dir.mkdir()

    video_paths = {}
    for camera in C.ALL_CAMERAS:
        path = videos_dir / ("%s.mp4" % camera)
        path.write_bytes(b"video")
        video_paths[camera] = str(path)

    for filenames in gc_runner.MODEL_SLOTS.values():
        (models_dir / filenames[0]).write_bytes(b"weights")
    return video_paths, str(models_dir)


def _run(stub_engine, inputs, tmp_path):
    engine_dir, record_path = stub_engine
    video_paths, models_dir = inputs
    result = gc_runner.run(
        video_paths=video_paths,
        models_dir=models_dir,
        output_dir=str(tmp_path / "stage1_out"),
        repo_root=_REPO_ROOT,
        engine_dir=str(engine_dir),
        verbose=False)
    with open(record_path, encoding="utf-8") as handle:
        return result, json.load(handle)


# -----------------------------------------------------------------------------
# The regression itself
# -----------------------------------------------------------------------------

def test_gap_model_paths_reach_the_engine_as_Path(stub_engine, inputs, tmp_path):
    """The exact EC2 failure: gap values must not be str."""
    _result, record = _run(stub_engine, inputs, tmp_path)

    assert record["gap_types"], "load_all_models was never called"
    for key, type_name in record["gap_types"].items():
        assert type_name != "str", (
            "gap model %r was passed as str -- this is the EC2 crash "
            "(path.stat() on a str)" % key)
        assert "Path" in type_name, "gap model %r arrived as %s" % (key, type_name)


def test_classification_model_paths_reach_the_engine_as_Path(
        stub_engine, inputs, tmp_path):
    """Classification only survived a str by accident; pin it too."""
    _result, record = _run(stub_engine, inputs, tmp_path)

    assert record["classification_types"]
    for key, type_name in record["classification_types"].items():
        assert type_name != "str", "classification model %r was passed as str" % key
        assert "Path" in type_name


def test_every_model_slot_is_delivered(stub_engine, inputs, tmp_path):
    """Both dictionaries, all five slots -- nothing silently dropped."""
    _result, record = _run(stub_engine, inputs, tmp_path)

    assert set(record["classification_types"]) == {"side", "top"}
    assert set(record["gap_types"]) == {"right", "left", "top"}
    assert record["load_all_models_completed"] is True
    assert record["build_class_maps_completed"] is True


def test_as_paths_coerces_and_is_idempotent():
    """The boundary helper, on its own."""
    coerced = gc_runner._as_paths({"a": "/tmp/x.pt", "b": Path("/tmp/y.pt")})
    assert all(isinstance(value, Path) for value in coerced.values())
    assert gc_runner._as_paths(coerced) == coerced


def test_resolve_models_still_reports_readable_strings():
    """The fix must not turn the operator-facing paths into repr() noise."""
    import inspect
    source = inspect.getsource(gc_runner.resolve_models)
    assert "os.path.join" in source          # unchanged, still str for messages


# -----------------------------------------------------------------------------
# ...and Stage 1 still completes, so the fix is not just type-correct
# -----------------------------------------------------------------------------

def test_stage1_completes_against_the_stub_engine(stub_engine, inputs, tmp_path):
    result, _record = _run(stub_engine, inputs, tmp_path)

    assert result.global_gap_count == 3
    assert result.global_wagon_count == result.global_gap_count - 1
    assert len(result.wagons) == result.global_wagon_count
    assert result.master_camera == C.CAMERA_RIGHT_UP_TOP
    assert set(result.cameras) == set(C.ALL_CAMERAS)


def test_stage1_shifts_frames_out_of_trimmed_space(stub_engine, inputs, tmp_path):
    """The crop offset must still be applied after the fix."""
    result, _record = _run(stub_engine, inputs, tmp_path)

    for wagon in result.wagons:
        for camera_id, interval in wagon.cameras.items():
            crop = result.cameras[camera_id].crop_start_frame
            assert crop > 0
            assert interval["start_frame"] >= crop


def test_stage1_writes_the_old_contract(stub_engine, inputs, tmp_path):
    """End to end: stub engine -> adapter -> the real loader."""
    from core.global_state_loader import (
        load_global_train_state, verify_roster_integrity)
    from global_counting import adapter

    result, _record = _run(stub_engine, inputs, tmp_path)
    out = tmp_path / "contract"
    state_path, tracking_path = adapter.write_documents(result, str(out))

    state = load_global_train_state(state_path)
    assert verify_roster_integrity(state) == []
    assert state.total_wagons == result.global_wagon_count
    assert state.uses_camera_frame_ranges
    assert os.path.isfile(tracking_path)


def test_only_the_two_artifact_toggles_are_overridden(stub_engine, inputs, tmp_path):
    """The engine stays frozen: no threshold or algorithm value is touched."""
    _result, record = _run(stub_engine, inputs, tmp_path)

    assert set(record["overrides"]) == {"GENERATE_TRIM_DEBUG_VIDEO",
                                        "GENERATE_GAP_ANNOTATED_VIDEO"}
    assert all(value == "False" for value in record["overrides"].values())


def test_session_is_cleaned_up_after_a_real_run(stub_engine, inputs, tmp_path):
    engine_dir, _record_path = stub_engine
    _run(stub_engine, inputs, tmp_path)

    assert str(engine_dir) not in sys.path

    # No surviving module may point INTO the stub engine. Merely being present
    # is not a leak: several engine module names (`global_alignment`,
    # `models`, `reporting` ...) legitimately exist in this process from
    # wagon_count/ or from this repo, and the session's job is to put those
    # back exactly as they were -- not to delete them.
    for name in gc_runner.ENGINE_MODULES:
        module = sys.modules.get(name)
        origin = getattr(module, "__file__", None) if module else None
        if origin:
            assert not os.path.abspath(origin).startswith(str(engine_dir)), (
                "%s still resolves into the stub engine after the session" % name)
    # our own packages must still be the real ones. `combined_train_report` is
    # a submodule, so it is only an attribute once imported -- assert on the
    # package's location, which is the thing the collision would change.
    import reporting
    assert os.path.abspath(reporting.__file__).startswith(_REPO_ROOT)
    from reporting import combined_train_report
    assert os.path.abspath(combined_train_report.__file__).startswith(_REPO_ROOT)
