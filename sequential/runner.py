"""Sequential mode: camera after camera, then ONE Global Assembly.

    Camera 1 complete -> local JSON/PDF -> seal -> release
    Camera 2 complete -> local JSON/PDF -> seal -> release
    ...
    Global Assembly -> canonical gaps -> canonical roster -> combined JSON/PDF

This is NOT Batch in a loop. Each camera runs a single decode that feeds GAP,
Door, Damage and Load, persists camera-local evidence carrying no canonical
wagon id, writes its own report so one available camera is immediately useful,
seals itself, and releases its models before the next camera starts. Only after
that does a single Global Assembly turn the evidence into meaning.
"""

from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from core import constants as C
from global_counting import runner as gc_runner

from sequential import camera_runner, evidence as ev, global_assembly


@dataclass
class SequentialOutcome:
    batch_key: str
    workspace: str
    cameras: List[camera_runner.CameraRunResult] = field(default_factory=list)
    assembly: Optional[global_assembly.AssemblyResult] = None
    seconds: float = 0.0

    @property
    def sealed_cameras(self) -> List[str]:
        return [result.camera_id for result in self.cameras if result.sealed]

    @property
    def failed_cameras(self) -> List[str]:
        return [result.camera_id for result in self.cameras if not result.sealed]


def resolve_reconstruction_models(recon_models_dir: str,
                                  repo_root: str) -> Dict[str, str]:
    """Resolve the five engine weights from the SUPPLIED directory, or explain.

    Sequential uses exactly the same contract as Batch/Stage-1:
    `global_counting.runner.resolve_models`, which expands the path and accepts
    the same filename spellings. There is no Sequential-specific search and no
    fallback that could override the CLI value.

    The failure message names the flag, because a run that simply omitted
    `--recon-models-dir` falls back to the in-repo default and would otherwise
    look identical to a resolution bug.
    """
    default_dir = os.path.join(repo_root, "models", "reconstruction")
    expanded = os.path.abspath(os.path.expanduser(
        os.path.expandvars(recon_models_dir or "")))
    try:
        return gc_runner.resolve_models(expanded)
    except gc_runner.GlobalCountingError as exc:
        hint = ""
        if expanded == os.path.abspath(default_dir):
            hint = ("\n\nNOTE: this is the in-repo DEFAULT directory, which "
                    "means no --recon-models-dir was supplied on the command "
                    "line (and $WAGONEYE_RECON_MODELS_DIR was not set). The "
                    "validated weights are not stored in the repository. Pass "
                    "the directory that holds them, e.g.\n"
                    "    --recon-models-dir ~/global_wagon_models")
        raise camera_runner.CameraRunError(str(exc) + hint) from exc


def camera_order(video_paths: Dict[str, str],
                 cameras: Sequence[str] = C.ALL_CAMERAS) -> List[str]:
    """Deterministic processing order from the authoritative configuration.

    `C.ALL_CAMERAS` -- RIGHT_UP, LEFT_UP, RIGHT_UP_TOP, LEFT_UP_TOP -- never
    filesystem order, so two runs on the same inputs process in the same order.
    """
    return [camera_id for camera_id in cameras if camera_id in video_paths]


def run_sequential(
    *,
    video_paths: Dict[str, str],
    workspace: str,
    repo_root: str,
    recon_models_dir: str,
    feat_models_dir: str,
    features: Sequence[str],
    batch_key: str,
    engine_dir: Optional[str] = None,
    door_stride: int = 3,
    damage_stride: int = 3,
    load_stride: int = 2,
    force_cameras: bool = False,
    skip_assembly: bool = False,
    verbose: bool = True,
) -> SequentialOutcome:
    started = time.time()
    order = camera_order(video_paths)
    outcome = SequentialOutcome(batch_key=batch_key, workspace=workspace)

    selected = list(features)
    if verbose:
        print("=" * 78)
        print("  SEQUENTIAL  batch=%s" % batch_key)
        print("=" * 78)
        print("[SEQ] camera order    : %s" % ", ".join(order))
        absent = [c for c in C.ALL_CAMERAS if c not in order]
        if absent:
            print("[SEQ] cameras absent  : %s  (each present camera is still "
                  "processed and reported independently)" % ", ".join(absent))
        # Print the SELECTED set and, explicitly, what was left out -- so a run
        # that silently defaulted to every feature is obvious in the log.
        skipped = [name for name in camera_runner.DEFAULT_STRIDES
                   if name not in selected]
        print("[SEQ] features        : %s" % (", ".join(selected) or "(none)"))
        print("[SEQ] features OFF    : %s"
              % (", ".join(skipped) if skipped
                 else "(none -- every feature is enabled)"))
        if "ocr" not in selected:
            print("[SEQ] OCR             : DISABLED (never imported, no model "
                  "loaded)")
        print("[SEQ] strides         : door=%d damage=%d load=%d"
              % (door_stride, damage_stride, load_stride))
        print("[SEQ] recon models    : %s"
              % os.path.abspath(os.path.expanduser(recon_models_dir or "")))
        print("[SEQ] feature models  : %s"
              % os.path.abspath(os.path.expanduser(feat_models_dir or "")))

    if not order:
        if verbose:
            print("[SEQ] no camera videos found -- nothing to do")
        outcome.seconds = time.time() - started
        return outcome

    # Resolve the five engine weights ONCE, from the supplied directory, before
    # any video is opened: a wrong directory should fail in a second, not after
    # the first camera has decoded.
    try:
        counting_models = resolve_reconstruction_models(recon_models_dir,
                                                        repo_root)
    except camera_runner.CameraRunError as exc:
        print("[SEQ] cannot resolve the reconstruction models:\n%s" % exc,
              file=sys.stderr)
        outcome.seconds = time.time() - started
        return outcome
    if verbose:
        for slot in sorted(counting_models):
            print("[SEQ]   %-22s -> %s" % (slot, counting_models[slot]))

    for camera_id in order:
        try:
            result = camera_runner.process_camera(
                camera_id=camera_id, video_path=video_paths[camera_id],
                workspace=workspace, repo_root=repo_root,
                recon_models_dir=recon_models_dir,
                feat_models_dir=feat_models_dir, features=features,
                engine_dir=engine_dir, door_stride=door_stride,
                damage_stride=damage_stride, load_stride=load_stride,
                force=force_cameras, verbose=verbose, batch_key=batch_key)
        except Exception as exc:                     # one camera must not
            import traceback                         # take down the others
            print("[SEQ] Camera %s FAILED: %s" % (camera_id, exc))
            traceback.print_exc(limit=4)
            result = camera_runner.CameraRunResult(
                camera_id=camera_id, status=ev.STATUS_FAILED, reason=str(exc))
        outcome.cameras.append(result)

    required = global_assembly.required_cameras()
    sealed = outcome.sealed_cameras
    if verbose:
        if all(camera_id in sealed for camera_id in required):
            print("[SEQ] ALL REQUIRED CAMERAS SEALED  (%s)" % ", ".join(sealed))
        else:
            print("[SEQ] REQUIRED CAMERAS NOT SEALED  sealed=%s required=%s"
                  % (sealed, list(required)))

    if skip_assembly:
        if verbose:
            print("[SEQ] --skip-assembly: camera reports only")
        outcome.seconds = time.time() - started
        return outcome

    outcome.assembly = global_assembly.assemble(
        workspace=workspace, repo_root=repo_root, batch_key=batch_key,
        engine_dir=engine_dir, verbose=verbose)

    outcome.seconds = time.time() - started
    if verbose:
        print("-" * 78)
        for result in outcome.cameras:
            print("[SEQ] %-14s %-8s gaps=%-3d observations=%-6d %s"
                  % (result.camera_id, result.status, result.unique_gap_count,
                     result.observation_count,
                     "(resumed)" if result.reused else ""))
        assembly = outcome.assembly
        if assembly and assembly.ready:
            print("[SEQ] combined JSON : %s" % assembly.report_json_path)
            print("[SEQ] combined PDF  : %s" % assembly.report_pdf_path)
        print("[SEQ] total %.1fs" % outcome.seconds)
    return outcome
