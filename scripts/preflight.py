#!/usr/bin/env python3
"""Preflight: can this machine actually run the pipeline end to end?

Everything the orchestrator needs before it touches a video -- interpreter,
third-party wheels, the external counting engine, the five counting weights,
the feature weights for the features you selected, the four input videos with
recognisable camera names, and a writable workspace -- is checked here and
reported as one list.

No model is loaded and no frame is decoded, so it finishes in seconds.

    python scripts/preflight.py \
        --local-inputs ~/wagon_eye_inputs/fresh_train \
        --recon-models-dir ~/global_wagon_models \
        --features door,load,damage

Exit code 0 = ready to run, 1 = something is missing, 2 = bad arguments.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_SCRIPT_DIR)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from core import constants as C                                    # noqa: E402
from core.feature_config import FEATURE_KEYS                       # noqa: E402
from orchestrator import master_runner as mr                       # noqa: E402

# Weight filename per feature key. Mirrors the four processors, each of which
# does os.path.join(feature_models_dir, <constant>).
FEATURE_MODEL_FILES = {
    "door":   C.MODEL_DOOR_STATE,
    "ocr":    C.MODEL_WAGON_ID_COUNTING,
    "load":   C.MODEL_LOADED,
    "damage": C.MODEL_DAMAGE,
}

PASS, FAIL, WARN = "PASS", "FAIL", "WARN"
_RESULTS = []


def record(status, title, detail=""):
    _RESULTS.append((status, title, detail))
    print("[%s] %s" % (status, title))
    for line in str(detail).splitlines():
        if line.strip():
            print("       " + line)


# -----------------------------------------------------------------------------
# interpreter + wheels
# -----------------------------------------------------------------------------

def check_python():
    version = "%d.%d.%d" % sys.version_info[:3]
    if sys.version_info < (3, 10):
        record(FAIL, "Python >= 3.10", "found %s at %s" % (version, sys.executable))
    else:
        record(PASS, "Python %s" % version, sys.executable)
    if sys.prefix == getattr(sys, "base_prefix", sys.prefix):
        record(WARN, "not running inside a virtualenv",
               "expected .venv; system-wide installs are easy to break")


def check_packages(features):
    required = [
        ("numpy", "numpy"), ("cv2", "opencv-python-headless"),
        ("torch", "torch"), ("torchvision", "torchvision"),
        ("ultralytics", "ultralytics"), ("scipy", "scipy"),
        ("pandas", "pandas"),                   # the engine builds DataFrames
        ("PIL", "Pillow"), ("reportlab", "reportlab"), ("boto3", "boto3"),
    ]
    if "ocr" in features:
        required.append(("easyocr", "easyocr"))

    for module_name, package in required:
        try:
            module = __import__(module_name)
        except Exception as exc:
            record(FAIL, "import %s" % module_name,
                   "%s\n  pip install %s" % (exc, package))
            continue
        record(PASS, "%s %s" % (module_name, getattr(module, "__version__", "?")))

    if "ocr" not in features:
        record(PASS, "easyocr not required",
               "OCR is not selected, so it is never imported")
    try:
        import torch
        record(PASS, "torch device: %s"
               % ("cuda" if torch.cuda.is_available() else "cpu"))
    except Exception:
        pass


def check_ffmpeg():
    if shutil.which("ffmpeg"):
        record(PASS, "ffmpeg on PATH")
        return
    try:
        import imageio_ffmpeg
        record(PASS, "ffmpeg via imageio-ffmpeg", imageio_ffmpeg.get_ffmpeg_exe())
    except Exception:
        record(WARN, "no ffmpeg found",
               "the engine falls back to the OpenCV mp4v writer for its "
               "trimmed clips: larger files, same counting result")


# -----------------------------------------------------------------------------
# the external counting engine
# -----------------------------------------------------------------------------

def check_engine(explicit, stage1_engine):
    from reconstruction import runner as reconstruction_runner

    engine = reconstruction_runner.resolve_engine(stage1_engine)
    record(PASS, "Stage-1 engine: %s" % engine)
    if engine != reconstruction_runner.ENGINE_GLOBAL_APP:
        record(WARN, "the external global engine is not selected",
               "running the retained wagon_count counter; nothing below applies")
        return None

    from global_counting import runner as gc_runner

    searched = gc_runner.engine_search_paths(_REPO_ROOT, explicit)
    try:
        engine_dir = gc_runner.locate_engine(_REPO_ROOT, explicit)
    except gc_runner.GlobalCountingError as exc:
        record(FAIL, "global wagon engine", str(exc))
        return None

    record(PASS, "global wagon engine", engine_dir)
    print("       configured search order:")
    for origin, path in searched:
        marker = "  <-- used" if os.path.abspath(path) in (
            engine_dir, os.path.dirname(engine_dir)) else ""
        print("         %-22s %s%s" % (origin, path, marker))

    missing = [name for name in
               ("global_wagon_pipeline.py", "config.py", "global_alignment.py",
                "camera_pipeline.py", "wagon_mapping.py", "trimming.py",
                "gap_detection.py", "gap_tracking.py", "io_paths.py",
                "models.py", "reporting.py")
               if not os.path.isfile(os.path.join(engine_dir, name))]
    if missing:
        record(FAIL, "engine looks incomplete", "missing: %s" % ", ".join(missing))
    else:
        record(PASS, "engine modules present (11/11 checked)")
    return engine_dir


# -----------------------------------------------------------------------------
# weights
# -----------------------------------------------------------------------------

def check_counting_models(models_dir):
    from global_counting import runner as gc_runner

    models_dir = os.path.abspath(os.path.expanduser(models_dir))
    if not os.path.isdir(models_dir):
        record(FAIL, "counting models dir", "not a directory: %s" % models_dir)
        return
    try:
        resolved = gc_runner.resolve_models(models_dir)
    except gc_runner.GlobalCountingError as exc:
        record(FAIL, "global counting weights", str(exc))
        return

    lines = []
    for slot in sorted(resolved):
        filename = os.path.basename(resolved[slot])
        accepted = gc_runner.MODEL_SLOTS[slot]
        note = "" if filename == accepted[0] else "   (accepted alias of %s)" % accepted[0]
        lines.append("%-22s %s%s" % (slot, filename, note))
    record(PASS, "global counting weights (5/5) in %s" % models_dir,
           "\n".join(lines))


def check_feature_models(models_dir, features):
    models_dir = os.path.abspath(os.path.expanduser(models_dir))
    if not os.path.isdir(models_dir):
        record(FAIL, "feature models dir", "not a directory: %s" % models_dir)
        return
    present, missing = [], []
    for key in features:
        filename = FEATURE_MODEL_FILES[key]
        target = present if os.path.isfile(
            os.path.join(models_dir, filename)) else missing
        target.append("%-8s %s" % (key, filename))
    if present:
        record(PASS, "feature weights (%d)" % len(present), "\n".join(present))
    if missing:
        record(FAIL, "missing feature weights in %s" % models_dir,
               "\n".join(missing) + "\nRename on download if your copy uses "
               "different spellings; do not duplicate weights.")
    skipped = [k for k in FEATURE_KEYS if k not in features]
    if skipped:
        record(PASS, "weights not needed: %s" % ", ".join(
            "%s (%s)" % (k, FEATURE_MODEL_FILES[k]) for k in skipped))


# -----------------------------------------------------------------------------
# inputs + workspace
# -----------------------------------------------------------------------------

def check_videos(local_inputs):
    if not local_inputs:
        record(WARN, "no --local-inputs given",
               "skipping the video check (fine for an --auto/S3 run)")
        return
    local_inputs = os.path.abspath(os.path.expanduser(local_inputs))
    if not os.path.isdir(local_inputs):
        record(FAIL, "input video dir", "not a directory: %s" % local_inputs)
        return

    from core.batch import scan_local_video_dir

    found = scan_local_video_dir(local_inputs)
    lines = ["%-14s %s" % (cam, os.path.basename(found[cam]))
             for cam in C.ALL_CAMERAS if cam in found]
    missing = [cam for cam in C.ALL_CAMERAS if cam not in found]
    if missing:
        aliases = "\n".join(
            "  %-14s accepts: %s" % (cam, ", ".join(C.CAMERA_FILENAME_ALIASES[cam]))
            for cam in missing)
        record(FAIL, "input videos (%d/4) in %s" % (len(found), local_inputs),
               ("\n".join(lines) + "\n" if lines else "")
               + "missing: %s\nA filename must contain one of its camera's "
                 "accepted spellings:\n%s" % (", ".join(missing), aliases))
    else:
        record(PASS, "input videos (4/4) in %s" % local_inputs, "\n".join(lines))


def check_workspace(workspace):
    workspace = os.path.abspath(os.path.expanduser(
        workspace or os.path.join(_REPO_ROOT, "batch_outputs")))
    try:
        os.makedirs(workspace, exist_ok=True)
        probe = os.path.join(workspace, ".preflight_write_test")
        with open(probe, "w", encoding="utf-8") as handle:
            handle.write("ok")
        os.remove(probe)
    except Exception as exc:
        record(FAIL, "workspace not writable", "%s: %s" % (workspace, exc))
        return
    free_gb = shutil.disk_usage(workspace).free / (1024 ** 3)
    detail = "%s  (%.1f GB free)" % (workspace, free_gb)
    # Stage 2 writes one JPEG per wagon per camera per frame; a long train on
    # four cameras is tens of GB.
    record(WARN if free_gb < 20 else PASS,
           "workspace writable" + (", LOW DISK" if free_gb < 20 else ""), detail)


# -----------------------------------------------------------------------------

def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="scripts/preflight.py",
        description="Verify this machine can run the integrated pipeline.")
    parser.add_argument("--local-inputs", default=None)
    parser.add_argument("--recon-models-dir", default=mr.DEFAULT_RECON_MODELS_DIR)
    parser.add_argument("--feat-models-dir", default=mr.DEFAULT_FEAT_MODELS_DIR)
    parser.add_argument("--workspace", default=None)
    parser.add_argument("--global-engine-dir", default=None)
    parser.add_argument("--stage1-engine", default=None)
    parser.add_argument("--features", default=mr.FEATURES_ALL_KEYWORD)
    args = parser.parse_args(argv)

    try:
        features = mr.parse_features(args.features)
    except ValueError as exc:
        print("ERROR: %s" % exc, file=sys.stderr)
        return 2

    print("=" * 72)
    print("PREFLIGHT  --  global_wagon_app + WagonEye (f3d2d81) integration")
    print("=" * 72)
    print("features to run: %s" % ", ".join(features))
    print("")

    check_python()
    check_packages(features)
    check_ffmpeg()
    check_engine(args.global_engine_dir, args.stage1_engine)
    check_counting_models(args.recon_models_dir)
    check_feature_models(args.feat_models_dir, features)
    check_videos(args.local_inputs)
    check_workspace(args.workspace)

    failures = [item for item in _RESULTS if item[0] == FAIL]
    warnings = [item for item in _RESULTS if item[0] == WARN]
    print("")
    print("=" * 72)
    print("passed: %d   warnings: %d   failed: %d"
          % (len(_RESULTS) - len(failures) - len(warnings),
             len(warnings), len(failures)))
    for _status, title, _detail in failures:
        print("  FAILED: %s" % title)
    print("READY TO RUN" if not failures else "NOT READY - fix the above")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
