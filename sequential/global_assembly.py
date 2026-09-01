"""Global Assembly: rebuild Batch's exact pipeline from sealed camera evidence.

    sealed per-camera engine results
      -> restore the engine's CAMERA_RESULTS
      -> the ENGINE'S OWN global half  (normalized timelines, master selection,
         alignment incl. reversal, missing-gap recovery, global gap timeline,
         wagon mapping)
      -> gc_runner._harvest + global_counting.adapter   -> GlobalTrainState
      -> Batch Stage 2  materializer.wagon_cache_builder
      -> Batch Stage 3  features.{load, door, ocr, damage}.processor
      -> Batch Stage 4  fusion.wagon_state_builder
      -> Batch Stage 5a reporting.camera_reports
      -> Batch Stage 5b reporting.combined_train_report

BATCH IS THE GOLDEN REFERENCE
Every step above is Batch's own function, called in Batch's own order. This
module contributes no algorithm: it restores state and sequences calls. That is
what makes the canonical output identical rather than merely similar --
alignment, reversal, recovery, wagon boundaries, ownership, gating, sampling,
aggregation and rendering are all executed by the same code Batch executes.

WHY THE FEATURES RUN HERE
Batch infers Door/Damage/Load over each wagon's stable interior of JPEG-90
cached frames. That frame set cannot exist before the canonical roster does, so
inferring during camera acquisition would use different frames AND different
pixels. The features therefore run here, after the roster and the materialized
cache exist, through Batch's own processors.

WHAT THIS COSTS
The original videos are decoded again here, by Batch's materializer, to build
that cache. That was accepted deliberately: exact parity outranks decode count.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from core import constants as C
from core.global_state_loader import (
    load_global_train_state, roster_fingerprint, verify_roster_integrity,
)
from global_counting import adapter
from global_counting import runner as gc_runner

from sequential import evidence as ev

ASSEMBLY_SCHEMA = "wagon_eye.global_assembly.v2"


class AssemblyNotReady(RuntimeError):
    """The evidence required for a canonical global train is absent."""


@dataclass
class AssemblyResult:
    ready: bool
    reason: str = ""
    cameras_used: List[str] = field(default_factory=list)
    cameras_missing: List[str] = field(default_factory=list)
    global_gap_count: int = 0
    global_wagon_count: int = 0
    master_camera: str = ""
    state_json_path: Optional[str] = None
    report_json_path: Optional[str] = None
    report_pdf_path: Optional[str] = None
    camera_report_paths: Dict[str, Optional[str]] = field(default_factory=dict)
    diagnostics: Dict[str, Any] = field(default_factory=dict)


# -----------------------------------------------------------------------------
# Readiness
# -----------------------------------------------------------------------------

def required_cameras() -> Tuple[str, ...]:
    """Every camera, because Batch's master is not fixed.

    `config.MASTER_CAMERA_SELECTION` is "max_unique_gaps", so the master -- and
    therefore the global gap count -- is whichever camera confirmed the most
    unique gaps. That cannot be known until all four have been processed, so a
    Batch-comparable assembly needs all four sealed.
    """
    return tuple(C.ALL_CAMERAS)


def readiness(workspace: str, *, cameras: Sequence[str] = C.ALL_CAMERAS,
              ) -> Tuple[bool, List[str], List[str], str]:
    """`(ready, sealed, missing, reason)` -- never fabricates."""
    sealed = ev.sealed_cameras(workspace, cameras)
    missing = [camera for camera in cameras if camera not in sealed]
    absent = [camera for camera in required_cameras() if camera not in sealed]
    if absent:
        return (False, sealed, missing,
                "not Batch-comparable: %s not sealed, and the master camera is "
                "the one with the most confirmed unique gaps, which cannot be "
                "known until every camera is processed" % ", ".join(absent))
    return (True, sealed, missing, "all cameras sealed")


# -----------------------------------------------------------------------------
# Restoring the engine's per-camera state
# -----------------------------------------------------------------------------

def restore_camera_results(evidences: Dict[str, ev.CameraEvidence],
                           ) -> Dict[str, Dict[str, Any]]:
    """Rebuild `camera_pipeline.CAMERA_RESULTS` from persisted engine records.

    The engine's global half reads only `normalized_timeline`,
    `trimmed_total_frames`, `trimmed_info["fps"]` and `status`; `_harvest`
    additionally reads `video_info`, `final_start_frame`, `final_end_frame`,
    `unique_gap_count` and `timeline_df`. All of those were persisted verbatim,
    so this restoration is lossless for everything downstream.
    """
    import pandas as pd

    results: Dict[str, Dict[str, Any]] = {}
    for camera_id, camera_evidence in evidences.items():
        record = camera_evidence.engine_result or {}
        if not record:
            raise AssemblyNotReady(
                "%s has no persisted engine result -- it was sealed by an "
                "older Sequential build; reprocess it with --force-cameras"
                % camera_id)
        camera_key = gc_runner.CAMERA_ID_TO_KEY[camera_id]
        results[camera_key] = {
            "camera": camera_key,
            "status": record.get("status", "UNKNOWN"),
            "video_info": dict(record.get("video_info") or {}),
            "trimmed_info": dict(record.get("trimmed_info") or {}),
            "final_start_frame": int(record.get("final_start_frame", 0) or 0),
            "final_end_frame": int(record.get("final_end_frame", 0) or 0),
            "trimmed_total_frames": int(
                record.get("trimmed_total_frames", 0) or 0),
            "unique_gap_count": int(record.get("unique_gap_count", 0) or 0),
            "n_frames": int(record.get("n_frames", 0) or 0),
            "normalized_timeline": pd.DataFrame(
                record.get("normalized_timeline") or []),
            "timeline_df": pd.DataFrame(
                record.get("classification_timeline") or []),
        }
    return results


def run_engine_global_half(engine_modules, camera_results, output_paths,
                           verbose: bool):
    """Call the engine's global stages, in the engine's own order.

    This is the same sequence `global_counting.runner.run` performs after
    `process_all_cameras()`, so the master camera, alignment, reversal,
    recovery, global gap timeline and wagon mapping are all decided by the
    engine exactly as they are in Batch.
    """
    ga = engine_modules["global_alignment"]
    wagon_mapping = engine_modules["wagon_mapping"]
    engine_reporting = engine_modules["reporting"]

    ga.build_normalized_timelines(camera_results)
    ga.select_master_camera()
    ga.validate_temporal_ordering()
    ga.set_master_camera()
    ga.match_all_cameras()
    ga.report_alignment_mappings()
    ga.recover_missing_gaps()
    ga.collect_unmatched_extras(output_paths["unmatched_extra_detections"])
    ga.build_global_gap_timeline()

    engine_reporting.write_normalized_gap_timelines(
        output_paths["normalized_gap_timelines"])
    engine_reporting.write_camera_alignment_summary(
        output_paths["camera_alignment_summary"])
    engine_reporting.write_global_gap_timeline(
        output_paths["global_gap_timeline"])

    wagon_mapping.build_global_wagon_timeline()
    wagon_mapping.write_global_wagon_timeline_csv(
        output_paths["global_wagon_timeline"])

    if verbose:
        print("[SEQ/ASSEMBLY] master camera : %s" % ga.MASTER_CAMERA)
        print("[SEQ/ASSEMBLY] global gaps   : %d" % int(ga.GLOBAL_GAP_COUNT))
        print("[SEQ/ASSEMBLY] global wagons : %d"
              % int(wagon_mapping.GLOBAL_WAGON_COUNT))


# -----------------------------------------------------------------------------
# Public entry
# -----------------------------------------------------------------------------

def assemble(*, workspace: str, repo_root: str, batch_key: str,
             engine_dir: Optional[str] = None,
             feat_models_dir: Optional[str] = None,
             features: Optional[Sequence[str]] = None,
             inference_opts: Optional[Dict[str, Any]] = None,
             cameras: Sequence[str] = C.ALL_CAMERAS,
             verbose: bool = True) -> AssemblyResult:
    """Build the canonical train and every final report, the Batch way."""
    if verbose:
        print("[SEQ] GLOBAL ASSEMBLY START")

    ready, sealed, missing, reason = readiness(workspace, cameras=cameras)
    if not ready:
        if verbose:
            print("[SEQ] GLOBAL ASSEMBLY NOT READY: %s" % reason)
            print("[SEQ] the per-camera reports remain valid on their own")
        return AssemblyResult(ready=False, reason=reason, cameras_used=sealed,
                              cameras_missing=missing)

    evidences = {camera_id: ev.load_evidence(workspace, camera_id)
                 for camera_id in sealed}
    video_paths = {
        camera_id: (camera_evidence.provenance.get("video") or {}).get("path")
        for camera_id, camera_evidence in evidences.items()
    }
    absent_videos = [camera for camera, path in video_paths.items()
                     if not path or not os.path.isfile(path)]

    resolved_engine = gc_runner.locate_engine(repo_root, engine_dir)
    engine_output_dir = os.path.join(workspace, "global_state",
                                     "global_counting")
    os.makedirs(engine_output_dir, exist_ok=True)

    # ---- the engine's own global half ------------------------------------
    with gc_runner.engine_session(resolved_engine):
        import camera_pipeline
        import config
        import global_alignment
        import io_paths
        import models
        import reporting as engine_reporting
        import wagon_mapping

        config.apply_overrides(GENERATE_TRIM_DEBUG_VIDEO=False,
                               GENERATE_GAP_ANNOTATED_VIDEO=False)
        output_paths = io_paths.prepare_output_dirs(engine_output_dir)

        camera_results = restore_camera_results(evidences)
        camera_pipeline.CAMERA_RESULTS.clear()
        camera_pipeline.CAMERA_RESULTS.update(camera_results)

        run_engine_global_half(
            {"global_alignment": global_alignment,
             "wagon_mapping": wagon_mapping,
             "reporting": engine_reporting},
            camera_pipeline.CAMERA_RESULTS, output_paths, verbose)

        csv_paths = {
            name: str(output_paths[name])
            for name in ("global_gap_timeline", "global_wagon_timeline",
                         "camera_alignment_summary",
                         "normalized_gap_timelines",
                         "unmatched_extra_detections")
            if name in output_paths
        }
        # Batch's own harvester, on the engine state the engine just produced.
        harvest = gc_runner._harvest(
            {"global_alignment": global_alignment,
             "wagon_mapping": wagon_mapping,
             "camera_pipeline": camera_pipeline, "config": config},
            {camera: path for camera, path in video_paths.items() if path},
            resolved_engine, engine_output_dir, csv_paths, verbose)

    if harvest.global_wagon_count <= 0:
        reason = ("the engine formed no global wagons (global_gap_count=%d)"
                  % harvest.global_gap_count)
        if verbose:
            print("[SEQ] GLOBAL ASSEMBLY NOT READY: %s" % reason)
        return AssemblyResult(ready=False, reason=reason,
                              cameras_used=sealed, cameras_missing=missing,
                              global_gap_count=harvest.global_gap_count,
                              master_camera=harvest.master_camera)

    # ---- the canonical contract, from Batch's own adapter -----------------
    global_state_dir = os.path.join(workspace, "global_state")
    state_path, tracking_path = adapter.write_documents(harvest,
                                                        global_state_dir)
    state = load_global_train_state(state_path)
    problems = verify_roster_integrity(state)
    if problems:
        raise AssemblyNotReady(
            "assembled roster is malformed: %s" % "; ".join(problems[:5]))
    guard = roster_fingerprint(state)

    from core.global_state_loader import (assert_roster_unchanged,
                                          load_per_camera_fps)
    per_camera_fps = load_per_camera_fps(tracking_path)

    if verbose:
        print("[SEQ/ASSEMBLY] canonical roster: GW_1..GW_%d  master=%s  "
              "fingerprint=%s" % (state.total_wagons, state.master_camera,
                                  guard[:16]))

    cache_root = os.path.join(workspace, "wagon_cache")
    states_root = os.path.join(workspace, "wagon_states")
    evidence_root = os.path.join(workspace, "evidence")
    reports_root = ev.combined_dir(workspace)

    if absent_videos:
        reason = ("cannot materialize the wagon cache: source video missing "
                  "for %s. Batch's feature stage reads that cache, so the "
                  "canonical features cannot be produced without it."
                  % ", ".join(sorted(absent_videos)))
        if verbose:
            print("[SEQ] GLOBAL ASSEMBLY INCOMPLETE: %s" % reason)
        return AssemblyResult(
            ready=False, reason=reason, cameras_used=sealed,
            cameras_missing=missing, master_camera=state.master_camera,
            global_gap_count=state.global_gap_count,
            global_wagon_count=state.total_wagons, state_json_path=state_path)

    # ---- Batch Stage 2: the SAME materializer ----------------------------
    from materializer import wagon_cache_builder

    if verbose:
        print("[SEQ] STAGE 2  wagon cache materialization (Batch's own)")
    wagon_cache_builder.build(
        state=state, video_paths=video_paths, per_camera_fps=per_camera_fps,
        cache_root=cache_root, camera_offsets=state.camera_time_offsets(),
        verbose=verbose)
    assert_roster_unchanged(state, guard, stage="Sequential Stage 2")

    # ---- Batch Stage 3: the SAME processors, the SAME order --------------
    options = dict(inference_opts or {})
    selected = tuple(features) if features is not None else ("door", "load",
                                                             "damage")
    feature_models_dir = feat_models_dir or os.path.join(
        repo_root, "models", "features")
    feature_kwargs = dict(state=state, cache_root=cache_root,
                          feature_models_dir=feature_models_dir,
                          output_dir=states_root, evidence_root=evidence_root,
                          verbose=verbose)
    if verbose:
        print("[SEQ] STAGE 3  feature inference (Batch's own processors)")
        print("[SEQ/STAGE3] features: %s" % ", ".join(selected))

    summary = _run_features(selected, options, feature_kwargs, state,
                            states_root, verbose)
    assert_roster_unchanged(state, guard, stage="Sequential Stage 3")

    # ---- Batch Stage 4 / 5a / 5b: the SAME builders and renderers --------
    from fusion import wagon_state_builder
    from reporting import camera_reports, combined_train_report

    if verbose:
        print("[SEQ] STAGE 4  fusion (Batch's own)")
    unified = wagon_state_builder.build(
        state=state, wagon_states_root=states_root, verbose=verbose)
    assert_roster_unchanged(state, guard, stage="Sequential Stage 4")

    if verbose:
        print("[SEQ] STAGE 5a camera reports (Batch's own renderer)")
    camera_report_paths: Dict[str, Optional[str]] = {}
    try:
        camera_report_paths = camera_reports.build_all(
            state=state, unified=unified, evidence_root=evidence_root,
            wagon_states_root=states_root, cache_root=cache_root,
            per_camera_tracking_path=tracking_path, output_dir=reports_root,
            batch_key=batch_key, verbose=verbose)
    except Exception as exc:                                # pragma: no cover
        print("[SEQ/STAGE5a] camera reports FAILED: %s" % exc, file=sys.stderr)

    if verbose:
        print("[SEQ] STAGE 5b combined report (Batch's own renderer)")
    report = combined_train_report.build(
        state=state, unified=unified, output_dir=reports_root,
        batch_key=batch_key, evidence_root=evidence_root,
        wagon_states_root=states_root, cache_root=cache_root,
        camera_pdf_urls={camera: os.path.basename(path)
                         for camera, path in camera_report_paths.items()
                         if path},
        verbose=verbose)

    diagnostics = {
        "schema": ASSEMBLY_SCHEMA,
        "cameras_used": sorted(evidences),
        "cameras_missing": missing,
        "master_camera": state.master_camera,
        "global_gap_count": state.global_gap_count,
        "global_wagon_count": state.total_wagons,
        "roster_fingerprint": guard,
        "feature_summary": summary,
        "produced_by": "engine global half + Batch stages 2-5b",
    }
    with open(os.path.join(reports_root, "global_assembly.json"), "w",
              encoding="utf-8") as handle:
        json.dump(diagnostics, handle, indent=2, default=str)

    if verbose:
        print("[SEQ] GLOBAL ASSEMBLY COMPLETE  master=%s gaps=%d wagons=%d"
              % (state.master_camera, state.global_gap_count,
                 state.total_wagons))

    return AssemblyResult(
        ready=True, reason="assembled", cameras_used=sorted(evidences),
        cameras_missing=missing, global_gap_count=state.global_gap_count,
        global_wagon_count=state.total_wagons,
        master_camera=state.master_camera, state_json_path=state_path,
        report_json_path=report.get("json_path"),
        report_pdf_path=report.get("pdf_path"),
        camera_report_paths=dict(camera_report_paths),
        diagnostics=diagnostics)


def _run_features(selected, options, feature_kwargs, state, states_root,
                  verbose) -> Dict[str, Any]:
    """Batch's Stage-3 ordering and per-feature arguments, reused verbatim.

    LOAD runs to completion first: the damage processor reads the sibling load
    JSON to drop floor_damage on LOADED wagons, and Batch orders it this way so
    that read is deterministic. Disabled features get Batch's own
    DISABLED_BY_USER sentinel, so fusion and the reports behave identically.
    """
    from orchestrator.master_runner import load_feature_runner

    extra = {
        "door": dict(inference_mode=options.get("door_inference_mode",
                                                "sampled"),
                     sample_stride=int(options.get("door_sample_stride", 3))),
        "damage": dict(inference_mode=options.get("damage_inference_mode",
                                                  "sampled"),
                       sample_stride=int(options.get("damage_sample_stride",
                                                     3))),
        "load": dict(inference_mode=options.get("load_inference_mode",
                                                "sampled"),
                     sample_stride=int(options.get("load_sample_stride", 2))),
    }

    def _mark_disabled(name):
        from features._common import empty_payload, write_per_wagon_json
        out = os.path.join(states_root, name)
        summary = {}
        for wagon in state.wagons:
            write_per_wagon_json(out, wagon.global_id, empty_payload(
                wagon.global_id, name, C.STATUS_DISABLED,
                disabled_by_user=True))
            summary[wagon.global_id] = C.STATUS_DISABLED
        if verbose:
            print("[SEQ/STAGE3/%s] DISABLED BY USER -- sentinel for %d wagons"
                  % (name, len(summary)))
        return summary

    summary: Dict[str, Any] = {}
    for name in ("load", "door", "ocr", "damage"):
        if name not in selected:
            summary[name] = _mark_disabled(name)
            continue
        try:
            summary[name] = load_feature_runner(name)(
                **feature_kwargs, **extra.get(name, {}))
        except Exception as exc:
            import traceback
            print("[SEQ/STAGE3/%s] CRASHED: %s" % (name, exc), file=sys.stderr)
            traceback.print_exc(limit=3)
            summary[name] = {}
    return summary
