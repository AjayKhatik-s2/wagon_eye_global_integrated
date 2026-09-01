"""Deterministic fixtures for Batch-vs-Sequential parity.

The point of these helpers is to let a test hand the SAME engine camera results
to both paths:

  Batch      : camera_pipeline.CAMERA_RESULTS <- process_camera() (live)
  Sequential : camera_pipeline.CAMERA_RESULTS <- restore_camera_results()
               (rebuilt from persisted evidence)

Everything downstream of that dictionary is literally the same engine code, so
if the dictionaries agree the global halves must agree. These fixtures build the
dictionary deterministically -- no video, no models, no clock -- so the
comparison is exact and repeatable.
"""

from __future__ import annotations

import json
import os
import sys
from typing import Dict, List, Optional, Sequence

_TEST_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_TEST_DIR)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from core import constants as C
from sequential import camera_runner, evidence as ev

# A four-wagon train: five gaps at fixed normalized positions.
FPS = 15.0
TOTAL_FRAMES = 900
REGION_START = 100
REGION_END = 799
TRIMMED_FRAMES = REGION_END - REGION_START + 1
CANONICAL_POSITIONS = (0.0, 250.0, 500.0, 750.0, 1000.0)


def real_engine_dir():
    """The frozen engine checkout, if this machine has it."""
    parent = os.path.dirname(_REPO_ROOT)
    for candidate in (os.environ.get("WAGONEYE_ENGINE_DIR"),
                      os.path.join(parent, "global_wagon_app"),
                      os.path.join(parent, "global_count_ec2",
                                   "global_wagon_app")):
        if candidate and os.path.isfile(os.path.join(candidate,
                                                     "global_alignment.py")):
            return candidate
    return None


def engine_camera_result(camera_key, positions=CANONICAL_POSITIONS, *,
                         status="VALID", region_start=REGION_START,
                         region_end=REGION_END, video_path=""):
    """A per-camera result shaped exactly like the engine's own.

    `normalized_timeline` and `timeline_df` are pandas frames, as the engine
    produces; every other key matches what `camera_pipeline.process_camera`
    returns and what the global half plus `_harvest` read.
    """
    import pandas as pd

    trimmed = region_end - region_start + 1
    rows = []
    for index, position in enumerate(positions, start=1):
        frame = int(round(position / 1000.0 * (trimmed - 1)))
        rows.append({
            "camera": camera_key,
            "local_gap_id": "%s_G%d" % (camera_key, index),
            "confirmation_frame": frame,
            "first_seen_frame": frame,
            "last_seen_frame": frame,
            "normalized_confirmation_time": float(position),
            "normalized_first_time": float(position),
            "normalized_last_time": float(position),
            "normalized_duration": 8.0,
            "max_confidence": 0.9,
            "average_confidence": 0.85,
            "frame_count": 3,
        })
    return {
        "camera": camera_key,
        "status": status,
        "video_info": {"fps": FPS, "total_frames": TOTAL_FRAMES,
                       "width": 640, "height": 480},
        "trimmed_info": {"fps": FPS, "total_frames": trimmed},
        "final_start_frame": region_start,
        "final_end_frame": region_end,
        "trimmed_total_frames": trimmed,
        "unique_gap_count": len(positions),
        "n_frames": TOTAL_FRAMES,
        "trimmed_video_path": video_path or ("%s_trimmed.mp4" % camera_key),
        "normalized_timeline": pd.DataFrame(rows),
        "timeline_df": pd.DataFrame(
            [{"frame_id": i, "normalized_class": "wagon", "is_wagon": True}
             for i in range(TOTAL_FRAMES)]),
    }


def all_camera_results(positions_by_camera=None):
    """Engine results keyed by ENGINE camera key, for all four cameras."""
    from global_counting import runner as gc_runner

    out = {}
    for camera_id in C.ALL_CAMERAS:
        key = gc_runner.CAMERA_ID_TO_KEY[camera_id]
        positions = (positions_by_camera or {}).get(camera_id,
                                                    CANONICAL_POSITIONS)
        out[key] = engine_camera_result(key, positions)
    return out


def write_sequential_evidence(workspace, camera_id, result):
    """Persist `result` the way the camera stage does, and seal it."""
    record = camera_runner.engine_record(result)
    video_info = record["video_info"]
    timing = ev.CameraTiming(
        fps=video_info["fps"], total_frames=video_info["total_frames"],
        decoded_frames=record["n_frames"],
        wagon_region_start_frame=record["final_start_frame"],
        wagon_region_end_frame=record["final_end_frame"],
        wagon_region_frames=record["trimmed_total_frames"],
        duration_seconds=round(record["n_frames"] / (video_info["fps"] or 1.0),
                               3))
    gaps = camera_runner._gaps_from_record(record)
    fingerprint = {"path": "%s.mp4" % camera_id, "size": 1, "mtime": 0.0,
                   "digest": "deterministic-%s" % camera_id}
    feature_config = {"features": ["door", "damage", "load"],
                      "applied_in": "global_assembly"}

    camera_evidence = ev.CameraEvidence(
        camera_id=camera_id, status=ev.STATUS_SEALED, timing=timing, gaps=gaps,
        observations=[],
        classification_timeline=record["classification_timeline"],
        segments=camera_runner._segments(camera_id, gaps),
        provenance={"video": fingerprint, "models": {},
                    "config_fingerprint": "deterministic",
                    "frame_width": video_info["width"],
                    "frame_height": video_info["height"],
                    "produced_by": "engine camera_pipeline.process_camera"},
        feature_config=feature_config,
        diagnostics={"engine_status": record["status"],
                     "trimmed_video_path": record["trimmed_video_path"]},
        engine_result=record)

    path = ev.write_evidence(workspace, camera_evidence)
    ev.write_seal(workspace, camera_id=camera_id, status=ev.STATUS_SEALED,
                  timing=timing, video_fingerprint=fingerprint,
                  model_fingerprints={}, config_digest="deterministic",
                  feature_config=feature_config, processing_seconds=0.0,
                  report_paths={}, unique_gap_count=len(gaps),
                  observation_count=0)
    return path


def seal_all(workspace, positions_by_camera=None):
    """Persist evidence for all four cameras; return the engine results used."""
    from global_counting import runner as gc_runner

    results = all_camera_results(positions_by_camera)
    for camera_id in C.ALL_CAMERAS:
        key = gc_runner.CAMERA_ID_TO_KEY[camera_id]
        write_sequential_evidence(workspace, camera_id, results[key])
    return results


def run_global_half(engine_dir, camera_results, output_dir, verbose=False):
    """Drive the REAL engine's global half and snapshot every decision.

    Needs no video and no models: the global half reads only the per-camera
    results. That is what makes an exact structural comparison possible in a
    unit test.
    """
    from global_counting import runner as gc_runner
    from sequential import global_assembly

    os.makedirs(output_dir, exist_ok=True)
    with gc_runner.engine_session(engine_dir):
        import camera_pipeline
        import config
        import global_alignment
        import io_paths
        import reporting as engine_reporting
        import wagon_mapping

        config.apply_overrides(GENERATE_TRIM_DEBUG_VIDEO=False,
                               GENERATE_GAP_ANNOTATED_VIDEO=False)
        output_paths = io_paths.prepare_output_dirs(output_dir)

        camera_pipeline.CAMERA_RESULTS.clear()
        camera_pipeline.CAMERA_RESULTS.update(camera_results)

        global_assembly.run_engine_global_half(
            {"global_alignment": global_alignment,
             "wagon_mapping": wagon_mapping,
             "reporting": engine_reporting},
            camera_pipeline.CAMERA_RESULTS, output_paths, verbose)

        alignments = {}
        for camera, mapping in sorted(
                getattr(global_alignment, "CAMERA_ALIGNMENTS", {}).items()):
            alignments[camera] = {
                "scale": float(mapping.scale),
                "offset": float(mapping.offset),
                "reversed": bool(mapping.is_reversed),
                "status": str(mapping.status),
                "matched": len(mapping.matched_camera_indices),
                "unmatched": sorted(mapping.unmatched_camera_indices),
            }
        snapshot = {
            "master_camera": global_alignment.MASTER_CAMERA,
            "master_reason": getattr(global_alignment,
                                     "MASTER_SELECTION_REASON", ""),
            "global_gap_count": int(global_alignment.GLOBAL_GAP_COUNT),
            "global_gap_ids": list(global_alignment.GLOBAL_GAP_IDS),
            "camera_gap_counts": dict(
                getattr(global_alignment, "CAMERA_GAP_COUNTS", {})),
            "alignments": alignments,
            "global_wagon_count": int(wagon_mapping.GLOBAL_WAGON_COUNT),
            "global_wagons": [
                {k: w.get(k) for k in sorted(w) if k != "snapshots"}
                for w in getattr(wagon_mapping, "GLOBAL_WAGONS", [])],
        }
    return json.loads(json.dumps(snapshot, default=str))


# =============================================================================
# A full local assembly: REAL engine + REAL Batch stages, faked pixels/weights
# =============================================================================

WIDTH, HEIGHT = 640, 480

# A centred, high-confidence box that survives Batch's REAL gates on merit:
# conf 0.9 >= door 0.68 and >= damage 0.55; area ratio 0.104 inside
# [0.005, 0.40]; centre (0.47, 0.50) clear of every edge zone. Nothing is
# loosened to make detections pass -- they pass Batch's own thresholds.
BBOX = [100.0, 80.0, 200.0, 160.0]


class FakeCapture:
    """Paints each frame's own index into pixel (0,0,0) so calls are traceable."""

    open_count = 0

    def __init__(self, path):
        FakeCapture.open_count += 1
        self._index = 0

    def isOpened(self):
        return True

    def read(self):
        import numpy as np

        if self._index >= TOTAL_FRAMES:
            return False, None
        frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
        frame[0, 0, 0] = self._index % 256
        self._index += 1
        return True, frame

    def release(self):
        pass

    def get(self, prop):
        import cv2

        return {cv2.CAP_PROP_FRAME_COUNT: float(TOTAL_FRAMES),
                cv2.CAP_PROP_FPS: FPS,
                cv2.CAP_PROP_FRAME_WIDTH: float(WIDTH),
                cv2.CAP_PROP_FRAME_HEIGHT: float(HEIGHT)}.get(prop, 0.0)


def install_fake_pixel_stack(monkeypatch):
    """Fake decode + fake detectors. Batch's REAL gates still do the deciding."""
    import cv2
    import numpy as np

    FakeCapture.open_count = 0
    monkeypatch.setattr(cv2, "VideoCapture", FakeCapture)
    calls = {"door": [], "damage": [], "load": [], "ocr": []}

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
        if "wagon_id" in name:
            return _FakeYolo("ocr", {0: "digit"})
        return _FakeYolo("load", {0: "loaded"})

    def _fake_run_classification(model, frame):
        calls["load"].append(int(frame[0, 0, 0]))
        return ("loaded", 0.9)

    # Each processor does `from features._common import load_yolo`, which binds
    # its OWN reference at import time. Patching only `_common` works when the
    # processor has not been imported yet and silently misses it when it has --
    # which made these fixtures pass alone and fail in a full run. Patch every
    # bound copy.
    import importlib

    from features import _common

    targets = [_common]
    for name in ("door", "damage", "load", "ocr"):
        try:
            targets.append(importlib.import_module(
                "features.%s.processor" % name))
        except ImportError:                             # pragma: no cover
            continue

    for module in targets:
        for attribute, replacement in (
                ("load_yolo", _fake_load_yolo),
                ("run_classification", _fake_run_classification)):
            if hasattr(module, attribute):
                monkeypatch.setattr(module, attribute, replacement)
    return calls


def prepare_workspace(root, positions_by_camera=None, *, videos=None):
    """Seal all four cameras with engine records pointing at real video files."""
    from global_counting import runner as gc_runner
    from sequential import evidence as ev

    workspace = os.path.join(root, "ws")
    video_dir = videos or os.path.join(root, "videos")
    os.makedirs(video_dir, exist_ok=True)
    results = all_camera_results(positions_by_camera)

    for camera_id in C.ALL_CAMERAS:
        key = gc_runner.CAMERA_ID_TO_KEY[camera_id]
        video_path = os.path.join(video_dir, "%s.mp4" % camera_id)
        with open(video_path, "wb") as handle:
            handle.write(b"video")
        result = dict(results[key])
        result["trimmed_video_path"] = video_path
        write_sequential_evidence(workspace, camera_id, result)

        camera_evidence = ev.load_evidence(workspace, camera_id)
        camera_evidence.provenance["video"]["path"] = video_path
        ev.write_evidence(workspace, camera_evidence)
    return workspace, results


def feature_models(root):
    """Weight files Batch's feature registry expects. Contents are irrelevant."""
    directory = os.path.join(root, "feat_models")
    os.makedirs(directory, exist_ok=True)
    for name in ("door_state.pt", "damage.pt", "loaded.pt",
                 "wagon_id_counting.pt"):
        with open(os.path.join(directory, name), "wb") as handle:
            handle.write(b"w")
    return directory


def run_full_assembly(root, monkeypatch, *, positions_by_camera=None,
                      features=("door", "load", "damage")):
    """REAL engine global half + REAL Batch stages 2-5b, on faked pixels.

    Returns (result, workspace, calls). Every decision that matters -- master
    camera, gap timeline, wagon boundaries, ownership, feature gates, fusion,
    report contents -- is made by engine/Batch code, not by the test.
    """
    import contextlib
    import io

    from sequential import global_assembly

    engine_dir = real_engine_dir()
    if engine_dir is None:                       # pragma: no cover
        raise RuntimeError("the frozen global_wagon_app checkout is missing")

    calls = install_fake_pixel_stack(monkeypatch)
    workspace, _results = prepare_workspace(root, positions_by_camera)
    models_dir = feature_models(root)

    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        result = global_assembly.assemble(
            workspace=workspace, repo_root=_REPO_ROOT, batch_key="parity",
            engine_dir=engine_dir, feat_models_dir=models_dir,
            features=features, verbose=False)
    return result, workspace, calls
