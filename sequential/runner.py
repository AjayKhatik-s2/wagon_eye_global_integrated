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
    uploaded: Dict[str, int] = field(default_factory=dict)
    dashboard: Dict[str, Any] = field(default_factory=dict)
    global_ingest: Dict[str, Any] = field(default_factory=dict)
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


# -----------------------------------------------------------------------------
# Delivery (Batch's Stage 6 / 6b / 6c, reached from the sequential flow)
# -----------------------------------------------------------------------------

def _deliver(outcome, *, workspace: str, batch_key: str,
             skip_upload: bool, skip_email: bool, verbose: bool = True) -> None:
    """Upload this batch's artifacts, then feed both dashboard endpoints.

    Reads only FINISHED files -- runs no model, opens no video, and recounts
    nothing.  The order is the contract:

        1. S3 upload of the whole batch tree
        2. per-camera dashboard ingest   POST /inspections/ingest      (x4)
        3. global dashboard ingest       POST /inspections/ingest-global (x1)

    Step 2 hands the receiver an `s3://` URI it fetches for itself, and step 3
    posts the fused report INLINE.  Both reference objects step 1 uploaded, which
    is why neither runs before it.  Step 3 runs after step 2 because the receiver
    stores the fused document as a virtual fifth camera that supersedes the four.

    `skip_upload` suppresses steps 1-3 together: without the upload there is no
    object for a URL to name, so posting would advertise files that do not exist.
    `skip_email` suppresses ONLY the email -- the two are independent controls.

    Every step is isolated.  A receiver outage costs a dashboard row, not the
    inspection, so nothing here can fail the batch.
    """
    from core import constants as C

    assembly = outcome.assembly
    reports_dir = ev.COMBINED_DIRNAME       # sequential's own reports folder

    if skip_upload:
        if verbose:
            print("[SEQ] DELIVERY skipped (--skip-upload): no S3 upload, "
                  "no dashboard ingest")
        if not skip_email:
            _send_email(outcome, batch_key=batch_key, verbose=verbose)
        return

    try:
        import boto3
        s3 = boto3.client("s3", region_name=C.S3_REGION)
    except Exception as exc:                                  # noqa: BLE001
        print("[SEQ/DELIVERY] no S3 client (%s) -- nothing delivered" % exc,
              file=sys.stderr)
        return

    # ---- 1. S3 upload -----------------------------------------------------
    from delivery import s3_upload
    if verbose:
        print("[SEQ] DELIVERY 1/3  S3 upload -> s3://%s/%s/%s/"
              % (C.S3_OUTPUT_BUCKET, C.S3_TRAIN_BATCH_PREFIX, batch_key))
    uploaded = {}
    for label, subdir, skip_ext in (
            ("global_state",     "global_state",     {".jpg", ".jpeg"}),
            ("wagon_states",     "wagon_states",     None),
            (reports_dir,        reports_dir,        None),
            ("evidence",         "evidence",         None),
            ("processed_videos", "processed_videos", None),
    ):
        local = os.path.join(workspace, subdir)
        if not os.path.isdir(local):
            continue
        try:
            uploaded[label] = s3_upload.upload_tree(
                s3, local, batch_key, sub_prefix=subdir,
                skip_extensions=skip_ext)
        except Exception as exc:                              # noqa: BLE001
            print("[SEQ/DELIVERY] upload %s failed: %s" % (label, exc),
                  file=sys.stderr)
    outcome.uploaded = uploaded
    if verbose:
        print("[SEQ] DELIVERY 1/3  uploaded: %s"
              % ", ".join("%s=%d" % kv for kv in sorted(uploaded.items())))

    # ---- 2. per-camera dashboard ingest -----------------------------------
    if verbose:
        print("[SEQ] DELIVERY 2/3  per-camera dashboard ingest")
    try:
        from delivery import dashboard_ingest
        outcome.dashboard = dashboard_ingest.run(
            batch_root=workspace, s3_client=s3, skip_upload=False,
            reports_dir=reports_dir)
        for camera, info in (outcome.dashboard.get("cameras") or {}).items():
            print("[SEQ]   %-14s %s%s"
                  % (camera, info.get("status"),
                     "  run_id=%s" % info["run_id"] if info.get("run_id") else ""))
    except Exception as exc:                                  # noqa: BLE001
        outcome.dashboard = {"error": str(exc)}
        print("[SEQ/DELIVERY] per-camera ingest failed: %s" % exc,
              file=sys.stderr)

    # ---- 3. global (fused) dashboard ingest -------------------------------
    if verbose:
        print("[SEQ] DELIVERY 3/3  global dashboard ingest")
    try:
        from delivery import global_train_webhook
        outcome.global_ingest = global_train_webhook.publish(
            report_json_path=getattr(assembly, "report_json_path", "") or "",
            batch_key=batch_key, verbose=verbose).to_dict()
    except Exception as exc:                                  # noqa: BLE001
        outcome.global_ingest = {"error": str(exc)}
        print("[SEQ/DELIVERY] global ingest failed: %s" % exc, file=sys.stderr)

    if not skip_email:
        _send_email(outcome, batch_key=batch_key, verbose=verbose)


def _send_email(outcome, *, batch_key: str, verbose: bool = True) -> None:
    """One email per batch.  Independent of the dashboard feed by design --
    `--skip-email` must not disable ingest, and `--skip-upload` must not be the
    only way to stop a mail."""
    try:
        from core.unified_wagon_state import summarize_wagons
        from delivery import notification
        assembly = outcome.assembly
        notification.send_email(
            batch_key=batch_key,
            report_pdf_url=getattr(assembly, "report_pdf_path", "") or "",
            report_json_url=getattr(assembly, "report_json_path", "") or "",
            summary=summarize_wagons([]),
            cameras_present=list(getattr(assembly, "cameras_used", []) or []),
            cameras_missing=list(getattr(assembly, "cameras_missing", []) or []),
            final_status="completed",
        )
    except Exception as exc:                                  # noqa: BLE001
        print("[SEQ/DELIVERY] email failed: %s" % exc, file=sys.stderr)


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
    door_inference_mode: str = "sampled",
    damage_inference_mode: str = "sampled",
    load_inference_mode: str = "sampled",
    force_cameras: bool = False,
    skip_assembly: bool = False,
    skip_upload: bool = True,
    skip_email: bool = True,
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

    # Global Assembly now runs Batch's Stages 2-5b, so it needs the feature
    # selection, the feature weights and the inference options the CLI chose.
    # Deterministic S3 prefix for this batch's evidence tree.  Assembly needs it
    # BEFORE it serializes the global report, so the report can name wagon-frame
    # and door URLs; it mirrors `s3_upload.upload_tree`'s own key layout
    # (<prefix>/<batch_key>/evidence/<rel>) rather than observing S3, because the
    # upload happens after assembly returns.
    #
    # None when this run will not upload -- the report then omits every URL
    # instead of promising objects that never land.
    evidence_url_base = (
        None if skip_upload else
        "https://%s.s3.%s.amazonaws.com/%s/%s/evidence"
        % (C.S3_OUTPUT_BUCKET, C.S3_REGION, C.S3_TRAIN_BATCH_PREFIX, batch_key)
    )

    outcome.assembly = global_assembly.assemble(
        workspace=workspace, repo_root=repo_root, batch_key=batch_key,
        engine_dir=engine_dir, feat_models_dir=feat_models_dir,
        features=features, evidence_url_base=evidence_url_base,
        inference_opts={
            "door_inference_mode": door_inference_mode,
            "door_sample_stride": door_stride,
            "damage_inference_mode": damage_inference_mode,
            "damage_sample_stride": damage_stride,
            "load_inference_mode": load_inference_mode,
            "load_sample_stride": load_stride,
        },
        verbose=verbose)

    # ---- DELIVERY: S3 -> per-camera ingest -> global ingest ---------------
    # Placed here, after assemble() has returned, because that is the first
    # point at which every artifact the dashboard references is complete on
    # disk: the sealed state, the fused per-wagon states, the evidence tree
    # (including the wagon frames assembly just materialized), and both report
    # families.  Ordering is strict -- nothing is POSTed until the object its URL
    # names has been uploaded.
    #
    # Nothing here reprocesses: it reads finished files and talks to S3 and the
    # receiver.  Every step is failure-isolated, because a delivered inspection
    # is worth more than a delivery receipt.
    if outcome.assembly is not None and outcome.assembly.ready:
        _deliver(outcome, workspace=workspace, batch_key=batch_key,
                 skip_upload=skip_upload, skip_email=skip_email,
                 verbose=verbose)

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
