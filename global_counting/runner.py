"""Stage 1 (NEW): drive the external `global_wagon_app` counting engine.

    4 raw videos
        -> global_wagon_app  (classification, wagon-region trimming, gap
                              detection, gap tracking, unique-gap confirmation,
                              0-1000 normalization, dynamic master selection,
                              cross-camera alignment, missing-gap recovery,
                              GLOBAL GAP TIMELINE, GLOBAL WAGON TIMELINE)
        -> adapter
        -> global_train_state.json + per_camera_tracking.json
        -> the unchanged downstream pipeline from commit f3d2d81

The engine is VALIDATED AND FROZEN. This module imports its stage functions and
calls them in exactly the order its own CLI does. It changes no threshold, no
algorithm and no configuration value except two artifact toggles (the trim-debug
and gap-annotated videos), because this pipeline owns frame extraction and
reporting. The engine's own snapshot extraction, figures and PDF are skipped for
the same reason.

Two hazards this module exists to contain:

1. **Module-name collision.** The engine ships top-level `reporting.py` and
   `models.py`; this repository has a `reporting/` package and a `models/`
   directory. Importing the engine naively either shadows ours or picks up ours
   instead of its own. `engine_session()` therefore swaps `sys.modules` and
   `sys.path` around the engine's whole execution and restores them afterwards.

2. **Engine objects escaping.** Nothing the engine builds may outlive the
   session, or a later attribute access would re-import a module that is no
   longer on the path. Everything is harvested into plain dicts and lists
   BEFORE the session exits.
"""

from __future__ import annotations

import contextlib
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from core import constants as C

# -----------------------------------------------------------------------------
# Engine identity and location
# -----------------------------------------------------------------------------

ENGINE_ENV_VAR = "GLOBAL_WAGON_APP_DIR"
ENGINE_MARKER = "global_wagon_pipeline.py"
ENGINE_PACKAGE_NAME = "global_wagon_app"
# The engine is its own project, deliberately NOT vendored here, so the
# validated counting algorithm has exactly one source of truth.
ENGINE_REPO_NAME = "global_count_ec2"
ENGINE_REPO_URL = "https://github.com/AjayKhatik-s2/global_count_ec2.git"

# Every top-level module the engine defines. `engine_session` stashes each of
# these so an engine module can never be left behind in sys.modules, and ours
# can never be shadowed after the session ends.
ENGINE_MODULES: Tuple[str, ...] = (
    "camera_map", "camera_pipeline", "classification", "config",
    "gap_annotation", "gap_detection", "gap_tracking", "global_alignment",
    "global_wagon_pipeline", "io_paths", "models", "pdf_report",
    "reporting", "runtime", "snapshot_extraction", "trimming", "utils",
    "visualization", "wagon_mapping",
)

# Camera id (this pipeline) <-> camera key (engine). Same four cameras.
CAMERA_ID_TO_KEY = {
    C.CAMERA_RIGHT_UP:     "right_up",
    C.CAMERA_LEFT_UP:      "left_up",
    C.CAMERA_RIGHT_UP_TOP: "right_up_top",
    C.CAMERA_LEFT_UP_TOP:  "left_up_top",
}
CAMERA_KEY_TO_ID = {key: cam for cam, key in CAMERA_ID_TO_KEY.items()}

# Engine class vocabulary -> this pipeline's classification vocabulary.
# Engine side models emit brakevan/empty_track/engine/wagon; top models emit
# brakevan/engine/track/wagon/wagon_loaded.
CLASS_MAP = {
    "engine":       C.CLASS_ENGINE,
    "brakevan":     C.CLASS_BRAKE_VAN,
    "wagon":        C.CLASS_WAGON,
    "wagon_loaded": C.CLASS_WAGON,
}

# Weight filenames per engine model slot, resolved inside --recon-models-dir.
# The canonical name from core.constants comes FIRST (this repository is the
# authority on what it ships), followed by the other spellings that occur in
# real model drops. The engine accepts all of them too.
MODEL_SLOTS: Dict[str, Tuple[str, ...]] = {
    "classification_side": (C.MODEL_SIDE_CLASSIFICATION,
                            "side_classify.pt", "side.pt"),
    "classification_top":  (C.MODEL_TOP_CLASSIFICATION,
                            "top_classify.pt", "top.pt"),
    "gap_right":           (C.MODEL_RIGHT_UP_GAP, "right_up_gap.pt",
                            "right_gap.pt", "right_wagon_gap.pt"),
    "gap_left":            (C.MODEL_LEFT_UP_GAP, "left_up_gap.pt",
                            "left_gap.pt", "left_wagon_gap.pt"),
    "gap_top":             (C.MODEL_TOP_GAP, "top_up_gap.pt", "top_wagon_gap.pt"),
}


class GlobalCountingError(RuntimeError):
    pass


def _expand(path: str) -> str:
    """`~` and `$VARS` -> absolute path.

    An env file, a systemd unit or a quoted export never expands `~`, so a
    perfectly reasonable GLOBAL_WAGON_APP_DIR="~/global_count_ec2" would
    otherwise silently fail to match.
    """
    return os.path.abspath(os.path.expanduser(os.path.expandvars(path)))


def _engine_root(candidate: str) -> Optional[str]:
    """The package dir, if `candidate` is - or contains - the engine."""
    if not candidate:
        return None
    for root in (candidate, os.path.join(candidate, ENGINE_PACKAGE_NAME)):
        if os.path.isfile(os.path.join(root, ENGINE_MARKER)):
            return os.path.abspath(root)
    return None


def engine_search_paths(repo_root: str, explicit: Optional[str] = None,
                        ) -> List[Tuple[str, str]]:
    """Ordered `(origin, absolute_path)` pairs `locate_engine` will try.

    Exposed so preflight tooling reports the same list the runtime uses.
    """
    raw: List[Tuple[str, str]] = []
    if explicit:
        raw.append(("--global-engine-dir", explicit))
    env = (os.environ.get(ENGINE_ENV_VAR) or "").strip()
    if env:
        raw.append(("$" + ENGINE_ENV_VAR, env))

    parent = os.path.dirname(os.path.abspath(repo_root))
    for label, base in (("beside repo", repo_root), ("above repo", parent),
                        ("home", os.path.expanduser("~"))):
        for name in (ENGINE_PACKAGE_NAME, ENGINE_REPO_NAME):
            raw.append((label, os.path.join(base, name)))

    seen, ordered = set(), []
    for origin, path in raw:
        expanded = _expand(path)
        if expanded not in seen:
            seen.add(expanded)
            ordered.append((origin, expanded))
    return ordered


def locate_engine(repo_root: str, explicit: Optional[str] = None) -> str:
    """Find the `global_wagon_app` package directory, or fail loudly.

    Each candidate is accepted either as the package itself or as a checkout
    containing it, so pointing at the clone works as well as pointing inside.
    """
    candidates = engine_search_paths(repo_root, explicit)
    for _origin, candidate in candidates:
        found = _engine_root(candidate)
        if found:
            return found

    tried = "\n".join("  %-22s %s" % (origin, path)
                      for origin, path in candidates)
    raise GlobalCountingError(
        "global wagon engine not found.\n\n"
        "The validated counting engine (%s) is a SEPARATE project and is "
        "deliberately not vendored in this repository.\n"
        "Looked for '%s' in these configured paths, either as the package "
        "itself or as a checkout containing it:\n%s\n\n"
        "Install it once:\n"
        "    git clone %s ~/%s\n"
        "    export %s=~/%s\n"
        "or point at it per run:\n"
        "    python -m orchestrator.master_runner --global-engine-dir "
        "/path/to/%s ...\n"
        % (ENGINE_PACKAGE_NAME, ENGINE_MARKER, tried, ENGINE_REPO_URL,
           ENGINE_REPO_NAME, ENGINE_ENV_VAR, ENGINE_REPO_NAME,
           ENGINE_PACKAGE_NAME))


# -----------------------------------------------------------------------------
# Import isolation
# -----------------------------------------------------------------------------

@contextlib.contextmanager
def engine_session(engine_dir: str):
    """Run a block with the engine importable and our own modules protected.

    `reporting` and `models` exist on BOTH sides. Inside the session the
    engine's win; on exit ours are restored exactly as they were, including the
    "was not imported at all" case.
    """
    saved_modules: Dict[str, Any] = {}
    for name in ENGINE_MODULES:
        if name in sys.modules:
            saved_modules[name] = sys.modules.pop(name)

    saved_path = list(sys.path)
    sys.path.insert(0, engine_dir)
    try:
        yield engine_dir
    finally:
        sys.path[:] = saved_path
        # Drop whatever the engine imported, then put ours back.
        for name in ENGINE_MODULES:
            sys.modules.pop(name, None)
        sys.modules.update(saved_modules)


# -----------------------------------------------------------------------------
# Model resolution
# -----------------------------------------------------------------------------

def resolve_models(models_dir: str) -> Dict[str, str]:
    """Map each engine model slot to a real file, or list everything missing."""
    models_dir = _expand(models_dir)
    resolved: Dict[str, str] = {}
    missing: List[str] = []
    for slot, filenames in MODEL_SLOTS.items():
        for filename in filenames:
            path = os.path.join(models_dir, filename)
            if os.path.isfile(path):
                resolved[slot] = path
                break
        else:
            missing.append("%-20s (looked for: %s)"
                           % (slot, ", ".join(filenames)))
    if missing:
        raise GlobalCountingError(
            "missing global counting weights in %s:\n  %s\n"
            "The engine needs FIVE weights: two classification models "
            "(side, top) and three gap models (right, left, top)."
            % (models_dir, "\n  ".join(missing)))
    return resolved


def _as_paths(mapping: Dict[str, Any]) -> Dict[str, Path]:
    """Every value as a `pathlib.Path`.

    The engine calls `.stat()` directly on the values handed to
    `load_all_models()`.  Its classification loader happens to re-wrap with
    `Path()` first, so a plain `str` survives there and then fails on the gap
    models -- a mismatch that only surfaces after both classification models
    have already loaded.  Coercing both dictionaries at the boundary makes the
    type guaranteed regardless of how the caller assembled them.
    """
    return {key: Path(value) for key, value in mapping.items()}


# -----------------------------------------------------------------------------
# Harvested result -- plain data only, no engine objects
# -----------------------------------------------------------------------------

@dataclass
class CameraHarvest:
    """What the engine established about one camera."""
    camera_id: str
    video_path: str = ""
    fps: float = 0.0
    total_frames: int = 0             # ORIGINAL video
    crop_start_frame: int = 0         # trimmed frame 0 == this original frame
    crop_end_frame: int = 0
    trimmed_total_frames: int = 0
    unique_gap_count: int = 0
    trim_status: str = ""
    alignment_status: str = ""
    is_reversed: bool = False
    scale: float = 1.0
    offset: float = 0.0
    matched_gaps: int = 0


@dataclass
class WagonHarvest:
    """One global wagon, with its interval in every camera."""
    wagon_number: int
    global_start_position: float
    global_end_position: float
    classification: str = C.CLASS_UNKNOWN
    classification_confidence: float = 0.0
    # camera_id -> interval, frames already shifted into ORIGINAL video space
    cameras: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    # Master-clock window, filled in by the adapter (original master frames).
    global_start_frame_master: int = 0
    global_end_frame_master: int = 0


@dataclass
class GlobalCountingResult:
    master_camera: str
    global_gap_count: int
    global_wagon_count: int
    wagons: List[WagonHarvest]
    cameras: Dict[str, CameraHarvest]
    engine_dir: str
    engine_output_dir: str
    normalized_scale: float = 1000.0
    # Non-wagon objects seen OUTSIDE the confirmed wagon region on the master
    # camera. The engine trims to the wagon region before counting, so the
    # locomotive and brake van are normally outside the wagon timeline by
    # design -- these counts keep the report KPIs honest without touching
    # Stage-1 counting. See the note in global_counting/README.md.
    leading_non_wagon: Dict[str, int] = field(default_factory=dict)
    trailing_non_wagon: Dict[str, int] = field(default_factory=dict)
    csv_paths: Dict[str, str] = field(default_factory=dict)
    elapsed_seconds: float = 0.0


# -----------------------------------------------------------------------------
# Harvest helpers (called INSIDE the session)
# -----------------------------------------------------------------------------

def _majority_class(timeline_df, start_frame: int, end_frame: int,
                    ) -> Tuple[str, float]:
    """Majority classification over an ORIGINAL-frame window.

    Uses the per-frame classification the trimming stage ALREADY produced --
    no model is re-run and no new algorithm is introduced.
    """
    if timeline_df is None or not len(timeline_df):
        return C.CLASS_UNKNOWN, 0.0
    window = timeline_df[(timeline_df["frame_id"] >= int(start_frame))
                         & (timeline_df["frame_id"] <= int(end_frame))]
    if not len(window):
        return C.CLASS_UNKNOWN, 0.0

    tally: Dict[str, int] = {}
    for raw in window["normalized_class"]:
        mapped = CLASS_MAP.get(str(raw), C.CLASS_UNKNOWN)
        tally[mapped] = tally.get(mapped, 0) + 1
    # Prefer a real class over UNKNOWN when both appear.
    known = {k: v for k, v in tally.items() if k != C.CLASS_UNKNOWN}
    chosen_pool = known or tally
    best = max(chosen_pool, key=lambda k: chosen_pool[k])
    return best, round(chosen_pool[best] / float(len(window)), 4)


def _non_wagon_tally(timeline_df, low: int, high: int) -> Dict[str, int]:
    """Count ENGINE / BRAKE_VAN *objects* in an original-frame window.

    A run of consecutive frames carrying the same non-wagon class is one
    object, not one per frame -- an eight-second locomotive must count once.
    """
    out: Dict[str, int] = {}
    if timeline_df is None or not len(timeline_df) or high < low:
        return out
    window = timeline_df[(timeline_df["frame_id"] >= int(low))
                         & (timeline_df["frame_id"] <= int(high))]
    if not len(window):
        return out

    previous = None
    for raw in window["normalized_class"]:
        mapped = CLASS_MAP.get(str(raw), C.CLASS_UNKNOWN)
        current = mapped if mapped in (C.CLASS_ENGINE, C.CLASS_BRAKE_VAN) else None
        if current is not None and current != previous:
            out[current] = out.get(current, 0) + 1
        previous = current
    return out


def _harvest(engine_modules, video_paths: Dict[str, str], engine_dir: str,
             engine_output_dir: str, csv_paths: Dict[str, str],
             verbose: bool) -> GlobalCountingResult:
    """Turn the engine's in-memory state into plain data."""
    ga = engine_modules["global_alignment"]
    wagon_mapping = engine_modules["wagon_mapping"]
    camera_results = engine_modules["camera_pipeline"].CAMERA_RESULTS
    config = engine_modules["config"]

    master_key = ga.MASTER_CAMERA
    if not master_key:
        raise GlobalCountingError(
            "the engine selected no master camera -- no camera produced "
            "confirmed unique gaps, so no global timeline exists")
    master_id = CAMERA_KEY_TO_ID[master_key]

    cameras: Dict[str, CameraHarvest] = {}
    for key, camera_id in CAMERA_KEY_TO_ID.items():
        result = camera_results.get(key) or {}
        info = result.get("video_info") or {}
        alignment = (ga.CAMERA_ALIGNMENTS or {}).get(key)
        cameras[camera_id] = CameraHarvest(
            camera_id=camera_id,
            video_path=video_paths.get(camera_id, ""),
            fps=float(info.get("fps") or 0.0),
            total_frames=int(info.get("total_frames") or result.get("n_frames") or 0),
            crop_start_frame=int(result.get("final_start_frame") or 0),
            crop_end_frame=int(result.get("final_end_frame") or 0),
            trimmed_total_frames=int(result.get("trimmed_total_frames") or 0),
            unique_gap_count=int(result.get("unique_gap_count") or 0),
            trim_status=str(result.get("status") or "NO_RESULT"),
            alignment_status=str(getattr(alignment, "status", "NOT_ESTIMATED")),
            is_reversed=bool(getattr(alignment, "is_reversed", False)),
            scale=float(getattr(alignment, "scale", 1.0) or 1.0),
            offset=float(getattr(alignment, "offset", 0.0) or 0.0),
            matched_gaps=len(getattr(alignment, "matches", ()) or ()),
        )

    master_timeline = (camera_results.get(master_key) or {}).get("timeline_df")

    wagons: List[WagonHarvest] = []
    for record in wagon_mapping.GLOBAL_WAGONS:
        harvest = WagonHarvest(
            wagon_number=int(record["wagon_number"]),
            global_start_position=float(record["global_start_position_1000"]),
            global_end_position=float(record["global_end_position_1000"]),
        )
        for key, camera_id in CAMERA_KEY_TO_ID.items():
            interval = (record.get("_cameras") or {}).get(key) or {}
            start_frame = interval.get("start_frame")
            end_frame = interval.get("end_frame")
            crop = cameras[camera_id].crop_start_frame
            # The engine works on the TRIMMED clip; Stage 2 opens the ORIGINAL
            # video. Shift every frame index by that camera's crop start.
            harvest.cameras[camera_id] = {
                "start_frame": (None if start_frame is None
                                else int(start_frame) + crop),
                "end_frame": (None if end_frame is None
                              else int(end_frame) + crop),
                "status": str(interval.get("interval_status") or "UNMATCHED"),
                "reversed": bool(interval.get("reversed", False)),
                "start_position": interval.get("start_position"),
                "end_position": interval.get("end_position"),
            }

        master_interval = harvest.cameras[master_id]
        if (master_interval["start_frame"] is not None
                and master_interval["end_frame"] is not None):
            harvest.classification, harvest.classification_confidence = \
                _majority_class(master_timeline,
                                master_interval["start_frame"],
                                master_interval["end_frame"])
        wagons.append(harvest)

    master = cameras[master_id]
    leading = _non_wagon_tally(master_timeline, 0, master.crop_start_frame - 1)
    trailing = _non_wagon_tally(
        master_timeline, master.crop_end_frame + 1,
        (master.total_frames - 1) if master.total_frames else master.crop_end_frame)

    if verbose:
        print("[GLOBAL] master camera      : %s" % master_id)
        print("[GLOBAL] global gap count   : %d" % int(ga.GLOBAL_GAP_COUNT))
        print("[GLOBAL] global wagon count : %d   (gaps - 1)"
              % int(wagon_mapping.GLOBAL_WAGON_COUNT))
        for camera_id, cam in cameras.items():
            print("[GLOBAL]   %-14s gaps=%-3d crop=[%d..%d] of %-6d "
                  "reversed=%-5s %s"
                  % (camera_id, cam.unique_gap_count, cam.crop_start_frame,
                     cam.crop_end_frame, cam.total_frames,
                     cam.is_reversed, cam.alignment_status))
        if leading or trailing:
            print("[GLOBAL] non-wagon objects outside the wagon region: "
                  "leading=%s trailing=%s" % (leading or "{}", trailing or "{}"))

    return GlobalCountingResult(
        master_camera=master_id,
        global_gap_count=int(ga.GLOBAL_GAP_COUNT),
        global_wagon_count=int(wagon_mapping.GLOBAL_WAGON_COUNT),
        wagons=wagons,
        cameras=cameras,
        engine_dir=engine_dir,
        engine_output_dir=engine_output_dir,
        normalized_scale=float(getattr(config, "NORMALIZED_TIMELINE_SCALE", 1000.0)),
        leading_non_wagon=leading,
        trailing_non_wagon=trailing,
        csv_paths=csv_paths,
    )


# -----------------------------------------------------------------------------
# Public entry
# -----------------------------------------------------------------------------

def run(
    *,
    video_paths: Dict[str, str],
    models_dir: str,
    output_dir: str,
    repo_root: str,
    engine_dir: Optional[str] = None,
    config_overrides: Optional[Dict[str, Any]] = None,
    verbose: bool = True,
) -> GlobalCountingResult:
    """Run the frozen engine end to end and harvest its global timeline.

    Args:
        video_paths: {camera_id -> path}; all four cameras are required.
        models_dir: the five counting weights live here.
        output_dir: the engine's audit artifacts are written under here.
        repo_root: this repository's root, used for engine discovery.
        engine_dir: explicit engine path; overrides the environment.
        config_overrides: extra engine config values (must be in the engine's
            own _OVERRIDABLE allow-list; anything else is refused by it).
    """
    import time

    missing = [cam for cam in C.ALL_CAMERAS if cam not in video_paths]
    if missing:
        raise GlobalCountingError(
            "global counting needs all 4 cameras; missing: %s" % missing)
    for camera_id, path in video_paths.items():
        if not os.path.isfile(path):
            raise GlobalCountingError(
                "video for %s does not exist: %s" % (camera_id, path))

    resolved_engine = locate_engine(repo_root, engine_dir)
    models = resolve_models(models_dir)
    engine_output_dir = os.path.join(output_dir, "global_counting")
    os.makedirs(engine_output_dir, exist_ok=True)

    if verbose:
        print("=" * 70)
        print("GLOBAL COUNTING ENGINE: NEW (global_wagon_app)")
        print("GLOBAL WAGON TIMELINE: ENABLED")
        print("OLD wagon_count BACKBONE: BYPASSED")
        print("=" * 70)
        print("engine        : %s" % resolved_engine)
        print("models        : %s" % _expand(models_dir))
        print("engine output : %s" % engine_output_dir)
        for slot in sorted(models):
            print("  %-22s %s" % (slot, os.path.basename(models[slot])))
        print("=" * 70)

    started = time.time()
    with engine_session(resolved_engine):
        import config

        # The ONLY configuration this integration changes: two debug videos
        # this pipeline does not consume. No threshold and no algorithm value
        # is touched. Overrides are validated by the engine's own allow-list.
        overrides = {"GENERATE_TRIM_DEBUG_VIDEO": False,
                     "GENERATE_GAP_ANNOTATED_VIDEO": False}
        overrides.update(config_overrides or {})
        config.apply_overrides(**overrides)

        import io_paths
        video_arguments = {CAMERA_ID_TO_KEY[cam]: path
                           for cam, path in video_paths.items()}
        # Use the dictionaries resolve_inputs RETURNS -- exactly what the
        # engine's own CLI passes on to load_all_models.  Re-deriving them from
        # our own string mapping skipped the engine's Path conversion.
        resolved = io_paths.resolve_inputs(video_arguments, models)
        output_paths = io_paths.prepare_output_dirs(engine_output_dir)

        from models import build_class_maps, load_all_models
        load_all_models(_as_paths(resolved["classification"]),
                        _as_paths(resolved["gap"]))
        build_class_maps()

        import camera_pipeline
        camera_pipeline.process_all_cameras()

        import global_alignment as ga
        ga.build_normalized_timelines(camera_pipeline.CAMERA_RESULTS)
        ga.select_master_camera()
        ga.validate_temporal_ordering()
        ga.set_master_camera()
        ga.match_all_cameras()
        ga.report_alignment_mappings()
        ga.recover_missing_gaps()
        ga.collect_unmatched_extras(output_paths["unmatched_extra_detections"])
        ga.build_global_gap_timeline()

        # The engine's own CSV audit trail. Snapshot extraction, the figures
        # and the engine PDF are deliberately skipped: this pipeline has its
        # own materializer and reporting layers.
        import reporting as engine_reporting
        engine_reporting.write_normalized_gap_timelines(
            output_paths["normalized_gap_timelines"])
        engine_reporting.write_camera_alignment_summary(
            output_paths["camera_alignment_summary"])
        engine_reporting.write_global_gap_timeline(
            output_paths["global_gap_timeline"])

        import wagon_mapping
        wagon_mapping.build_global_wagon_timeline()
        wagon_mapping.write_global_wagon_timeline_csv(
            output_paths["global_wagon_timeline"])

        csv_paths = {
            name: str(output_paths[name])
            for name in ("global_gap_timeline", "global_wagon_timeline",
                         "camera_alignment_summary", "normalized_gap_timelines",
                         "unmatched_extra_detections")
            if name in output_paths
        }

        result = _harvest(
            {"global_alignment": ga, "wagon_mapping": wagon_mapping,
             "camera_pipeline": camera_pipeline, "config": config},
            video_paths, resolved_engine, engine_output_dir, csv_paths, verbose)

    result.elapsed_seconds = time.time() - started
    if result.global_wagon_count <= 0:
        raise GlobalCountingError(
            "the engine formed no global wagons (global_gap_count=%d). "
            "At least two confirmed global gaps are required."
            % result.global_gap_count)
    return result
