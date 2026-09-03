"""Phase 1: Batch's Stage 2/3/4 run on ONE camera's own wagon windows.

WHAT THIS IS
The cache builder, the three feature processors and the fusion builder, called
with a camera-local `GlobalTrainState` instead of a canonical one. That is the
entire difference. No feature algorithm, threshold, stride, filter or verdict
rule is reimplemented here -- every one of them is Batch's, reached through the
same `load_feature_runner` dispatch Batch uses.

WHY PHASE 1 CAN RUN THEM AT ALL
`wagon_cache_builder` addresses frames through `wagon.camera_frame_ranges`, and
`_wagon_local_range` PREFERS that window over any master-time projection. The
feature processors then read `(cache_root, gw_id, camera_id)`. Nothing in that
chain needs a canonical roster -- only a state whose wagons carry windows, which
`local_state_adapter` supplies.

TWO THINGS COPIED FROM BATCH DELIBERATELY, NOT REINVENTED

  Run ORDER. Batch runs LOAD to completion first, then door / ocr / damage,
  because the damage processor reads the sibling `load` JSON to drop
  floor_damage tracks on LOADED wagons. Running them together raced the load
  writer -- handled fail-open, but nondeterministic. The same order is kept here
  for the same reason; a different order would give a camera-local damage
  verdict that Batch would not produce.

  DISABLED sentinels. A feature the operator turned off gets a
  DISABLED_BY_USER payload per wagon, so fusion and the report show
  "DISABLED BY USER" rather than silently treating the field as NO_DATA. That
  distinction is visible in the rendered report, so Phase 1 must make it too.

WHERE IT WRITES, AND WHY IT MATTERS
Everything lands under `camera_local/<CAMERA>/`. Phase 1 wagon ids are
camera-local (`LEFT_UP_W1`), and the canonical `wagon_cache/`, `wagon_states/`
and `evidence/` trees are what Phase 2 builds and what the Stage-6 upload
mirrors verbatim. Writing local ids into those trees would put directories in S3
that no canonical report references, and would make an artifact comparison
against Batch see files Batch cannot have.
"""

from __future__ import annotations

import os
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, Optional, Sequence

from core import constants as C
from core.global_state_loader import GlobalTrainState

#: One directory per camera, holding that camera's whole Phase-1 world.
CAMERA_LOCAL_DIRNAME = "camera_local"

#: Batch's Stage-3 order. LOAD first, to completion -- see the module docstring.
LOAD_FIRST = "load"
THEN_PARALLEL = ("door", "ocr", "damage")


def camera_local_root(workspace: str, camera_id: str) -> str:
    return os.path.join(workspace, CAMERA_LOCAL_DIRNAME, camera_id)


def paths_for(workspace: str, camera_id: str) -> Dict[str, str]:
    """The four Phase-1 directories for one camera, isolated from canonical."""
    root = camera_local_root(workspace, camera_id)
    return {
        "root": root,
        "cache_root": os.path.join(root, "wagon_cache"),
        "states_root": os.path.join(root, "wagon_states"),
        "evidence_root": os.path.join(root, "evidence"),
        "tracking_path": os.path.join(root, "per_camera_tracking.json"),
    }


def _mark_disabled(states_root: str, state: GlobalTrainState, name: str,
                   verbose: bool = True) -> Dict[str, str]:
    """Batch's own sentinel, for a feature the operator turned off.

    Uses `features._common`'s writer and payload builder rather than a
    hand-rolled dict, so a disabled feature looks identical to Batch's.
    """
    from features._common import empty_payload, write_per_wagon_json

    out = os.path.join(states_root, name)
    summary: Dict[str, str] = {}
    for wagon in state.wagons:
        write_per_wagon_json(out, wagon.global_id, empty_payload(
            wagon.global_id, name, C.STATUS_DISABLED, disabled_by_user=True))
        summary[wagon.global_id] = C.STATUS_DISABLED
    if verbose:
        print("[SEQ/P1/%s] DISABLED BY USER -- sentinel for %d local wagon(s)"
              % (name, len(summary)))
    return summary


def run_camera_local(
    *,
    state: GlobalTrainState,
    camera_id: str,
    video_path: str,
    workspace: str,
    feat_models_dir: str,
    features: Sequence[str],
    fps: float,
    door_stride: int = 3,
    damage_stride: int = 3,
    load_stride: int = 2,
    door_inference_mode: str = "sampled",
    damage_inference_mode: str = "sampled",
    load_inference_mode: str = "sampled",
    verbose: bool = True,
) -> Dict[str, Any]:
    """Cache -> features -> fusion for ONE camera, on its own windows.

    Returns `{paths, timings, feature_summary, unified, cache}`. Timings are
    per stage because the cost of this phase is the thing to measure, and an
    aggregate hides which model is expensive.

    Every stage is failure-isolated the way Batch isolates it: a crashed feature
    costs that feature, not the camera. The camera is still sealed, and its
    report shows the feature as unavailable rather than the run dying.
    """
    from materializer import wagon_cache_builder

    paths = paths_for(workspace, camera_id)
    for key in ("cache_root", "states_root", "evidence_root"):
        os.makedirs(paths[key], exist_ok=True)

    timings: Dict[str, float] = {}
    selected = [f for f in features]
    cache_stats: Dict[str, Any] = {}

    # ---- Batch Stage 2, one camera ---------------------------------------
    t0 = time.time()
    try:
        result = wagon_cache_builder.build(
            state=state,
            video_paths={camera_id: video_path},
            per_camera_fps={camera_id: fps},
            cache_root=paths["cache_root"],
            # A single camera has no clock delta to another camera, and the
            # window is explicit in camera_frame_ranges regardless.
            camera_offsets={camera_id: 0.0},
            verbose=verbose,
        )
        # `frames_written` is Dict[gw_id -> Dict[camera_id -> n_frames]], NOT
        # a scalar. Both are kept: the nested map is the real per-wagon,
        # per-camera record and is worth persisting for diagnosis, while
        # `total_frames` is the scalar a log line or a summary needs.
        #
        # `total_frames()` is CacheBuildResult's OWN accessor
        # (sum of per_camera_total), so the number reported here is the same
        # number Batch would report rather than a re-derivation.
        cache_stats = {
            "frames_written": dict(getattr(result, "frames_written", {}) or {}),
            "total_frames": int(result.total_frames()),
            "per_camera_total": dict(getattr(result, "per_camera_total", {})
                                     or {}),
            "missing_cameras": list(getattr(result, "missing_cameras", [])
                                    or []),
            "elapsed_seconds": float(getattr(result, "elapsed_seconds", 0.0)),
        }
    except Exception as exc:                                   # noqa: BLE001
        print("[SEQ/P1] wagon cache FAILED for %s: %s" % (camera_id, exc),
              file=sys.stderr)
        traceback.print_exc(limit=3)
    timings["cache"] = time.time() - t0

    # ---- Batch Stage 3, one camera ---------------------------------------
    from orchestrator.master_runner import load_feature_runner

    feature_kwargs = dict(
        state=state,
        cache_root=paths["cache_root"],
        feature_models_dir=feat_models_dir,
        output_dir=paths["states_root"],
        evidence_root=paths["evidence_root"],
        verbose=verbose,
    )
    extra: Dict[str, Dict[str, Any]] = {
        "door":   dict(inference_mode=door_inference_mode,
                       sample_stride=int(door_stride)),
        "damage": dict(inference_mode=damage_inference_mode,
                       sample_stride=int(damage_stride)),
        "load":   dict(inference_mode=load_inference_mode,
                       sample_stride=int(load_stride)),
    }

    feature_summary: Dict[str, Any] = {}

    def _run(name):
        t = time.time()
        try:
            return load_feature_runner(name)(**feature_kwargs,
                                             **extra.get(name, {}))
        except Exception as exc:                               # noqa: BLE001
            print("[SEQ/P1/%s] CRASHED: %s" % (name, exc), file=sys.stderr)
            traceback.print_exc(limit=3)
            return {}
        finally:
            timings["feature_%s" % name] = time.time() - t

    # LOAD first, to completion -- damage's load-aware floor_damage filter
    # reads the load JSON, and Batch orders it this way for that reason.
    if LOAD_FIRST in selected:
        feature_summary[LOAD_FIRST] = _run(LOAD_FIRST)
    else:
        feature_summary[LOAD_FIRST] = _mark_disabled(
            paths["states_root"], state, LOAD_FIRST, verbose)

    targets = [n for n in THEN_PARALLEL if n in selected]
    for name in THEN_PARALLEL:
        if name not in targets:
            feature_summary[name] = _mark_disabled(
                paths["states_root"], state, name, verbose)
    if targets:
        with ThreadPoolExecutor(max_workers=len(targets)) as pool:
            futures = {pool.submit(_run, name): name for name in targets}
            for future in as_completed(futures):
                feature_summary[futures[future]] = future.result()

    # ---- Batch Stage 4, one camera ---------------------------------------
    t0 = time.time()
    unified: Dict[str, Any] = {}
    try:
        from fusion import wagon_state_builder
        unified = wagon_state_builder.build(
            state=state, wagon_states_root=paths["states_root"],
            verbose=verbose)
    except Exception as exc:                                   # noqa: BLE001
        print("[SEQ/P1] fusion FAILED for %s: %s" % (camera_id, exc),
              file=sys.stderr)
        traceback.print_exc(limit=3)
    timings["fusion"] = time.time() - t0

    if verbose:
        print("[SEQ/P1/%s] cache=%.1fs (%d frames) %s fusion=%.1fs  "
              "local wagons=%d"
              % (camera_id, timings.get("cache", 0.0),
                 # The SCALAR, not the nested per-wagon map: formatting
                 # `frames_written` with %d raised
                 #   TypeError: %d format: a real number is required, not dict
                 # after fusion had already succeeded, so a camera whose whole
                 # Phase-1 pipeline had completed was marked FAILED and
                 # reprocessed. A log statement must never be able to do that.
                 cache_stats.get("total_frames", 0),
                 " ".join("%s=%.1fs" % (n.split("_", 1)[1], v)
                          for n, v in sorted(timings.items())
                          if n.startswith("feature_")),
                 timings.get("fusion", 0.0), len(state.wagons)))

    return {
        "paths": paths,
        "timings": timings,
        "feature_summary": feature_summary,
        "unified": unified,
        "cache": cache_stats,
    }
