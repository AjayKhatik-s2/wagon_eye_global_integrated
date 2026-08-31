"""Global Assembly: the ONE place a global interpretation is created.

    sealed camera evidence  ->  canonical gap sequence  ->  canonical roster
      ->  projection into every camera  ->  ownership  ->  aggregation
      ->  fusion  ->  combined_train_report.{json,pdf}

Hard rules this module obeys, and that `tests/test_sequential_architecture.py`
enforces by inspecting its source:

* NO inference. No `cv2.VideoCapture`, no YOLO `predict`, no `load_yolo`, no
  model construction, no GapTracker. Every number comes from evidence that was
  persisted during the single per-camera decode. The pipeline is
  `INFERENCE ONCE -> PERSIST -> INTERPRET`, never `INTERPRET -> INFER AGAIN`.
* Exactly ONE canonical roster. RIGHT_UP (`C.MASTER_CAMERA`) is the canonical
  gap authority; the other three corroborate. A support camera that saw an
  extra gap contributes a DIAGNOSTIC, never a wagon; a support camera that
  missed a canonical gap still gets that gap projected into its timeline.
* Alignment uses the engine's own validated maths -- `monotonic_gap_match` and
  `robust_linear_fit` are pure functions over position lists, so they are
  called directly on the persisted normalized positions. No alignment rule is
  reimplemented here.
* The canonical contract is built by `global_counting.adapter`, the same
  builder Batch uses, so both modes emit the identical schema (including
  `camera_frame_ranges` and `global_gaps`) and stay comparable.

CONFIDENCE GATING -- identical to Batch, by reuse
------------------------------------------------
The camera stage gates detections with Batch's OWN code before persisting them:
Door with `TrackerConfig().closed_confidence_threshold` and Damage with
`features.damage.processor._filter_detections_for_top` (floor, skip classes,
area ratio, edge-zone suppression). So the surviving detection set for a given
frame is the same set Batch would keep, and the aggregation below then feeds the
same `EvidenceAggregator` groups through the same Door helpers. Load uses
Batch's `_LOADED_RATIO_THRESHOLD` and its `used` denominator.
`tests/test_batch_sequential_parity.py` compares both paths at each of those
decision points.

Alignment, including reversal, is the engine's own `estimate_alignment` driven
from persisted timelines, and projection is the alignment's own
`project_to_camera`, so a reversed support camera behaves exactly as it does in
Batch.

Still outstanding: NO real end-to-end run has been performed in Sequential
mode. Every test uses a stub engine and a fake video capture.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from core import constants as C
from core import wagon_ownership
from core.global_state_loader import (
    load_global_train_state, verify_roster_integrity,
)
from global_counting import adapter
from global_counting import runner as gc_runner

from sequential import evidence as ev

ASSEMBLY_SCHEMA = "wagon_eye.global_assembly.v1"


class AssemblyNotReady(RuntimeError):
    """Raised when the evidence required for a canonical global train is absent.

    Global Assembly refuses to fabricate a combined train from partial
    evidence: a report that silently described half a train would be worse than
    no report.
    """


@dataclass
class AssemblyResult:
    ready: bool
    reason: str = ""
    cameras_used: List[str] = field(default_factory=list)
    cameras_missing: List[str] = field(default_factory=list)
    global_gap_count: int = 0
    global_wagon_count: int = 0
    state_json_path: Optional[str] = None
    report_json_path: Optional[str] = None
    report_pdf_path: Optional[str] = None
    diagnostics: Dict[str, Any] = field(default_factory=dict)


# -----------------------------------------------------------------------------
# Readiness
# -----------------------------------------------------------------------------

def required_cameras() -> Tuple[str, ...]:
    """The cameras a canonical global train needs.

    RIGHT_UP is indispensable because it is the canonical gap authority.  The
    support cameras improve the result but a train can be assembled without
    all of them, so only the authority is strictly required.
    """
    return (C.MASTER_CAMERA,)


def readiness(workspace: str, *, cameras: Sequence[str] = C.ALL_CAMERAS,
              ) -> Tuple[bool, List[str], List[str], str]:
    """`(ready, sealed, missing, reason)` -- never fabricates."""
    sealed = ev.sealed_cameras(workspace, cameras)
    missing = [camera for camera in cameras if camera not in sealed]
    for camera in required_cameras():
        if camera not in sealed:
            return (False, sealed, missing,
                    "canonical gap authority %s is not sealed" % camera)
    return (True, sealed, missing, "canonical authority sealed")


# -----------------------------------------------------------------------------
# Canonical gaps + projection
# -----------------------------------------------------------------------------

def _frame_for_position(camera_evidence: ev.CameraEvidence,
                        position: float) -> Optional[int]:
    """Camera-local ORIGINAL frame for a position on that camera's 0-1000 axis.

    The inverse of the engine's normalization: position 0 is the first frame of
    the camera's confirmed wagon region and 1000 is the last.
    """
    timing = camera_evidence.timing
    span = max(1, timing.wagon_region_frames - 1)
    if timing.wagon_region_frames <= 0:
        return None
    offset = int(round((float(position) / 1000.0) * span))
    offset = max(0, min(span, offset))
    return timing.wagon_region_start_frame + offset


def _gap_frame(camera_evidence: ev.CameraEvidence):
    """One camera's persisted gap timeline as the DataFrame the engine expects."""
    import pandas as pd

    return pd.DataFrame([
        {"local_gap_id": gap.local_gap_id,
         "normalized_confirmation_time": float(gap.normalized_position),
         "normalized_duration": float(gap.normalized_duration)}
        for gap in camera_evidence.gaps
    ], columns=["local_gap_id", "normalized_confirmation_time",
                "normalized_duration"])


def populate_engine_alignment_state(global_alignment, evidences, master_id):
    """Load the persisted timelines into the engine's alignment state.

    This is state population, NOT algorithm reimplementation: the engine's
    `estimate_alignment` reads `GAP_TIMELINES` and the `MASTER_*` arrays from
    its own module, so filling those from sealed evidence lets Sequential call
    the SAME validated function Batch's engine run calls -- including its
    scale-grid seeding, its iterative robust fit, its duration-weighted
    matching and its reversed-orientation test. Nothing is decoded and no model
    is touched.
    """
    import numpy as np

    master_key = gc_runner.CAMERA_ID_TO_KEY[master_id]

    global_alignment.GAP_TIMELINES.clear()
    global_alignment.CAMERA_ALIGNMENTS.clear()
    for camera_id, camera_evidence in evidences.items():
        global_alignment.GAP_TIMELINES[
            gc_runner.CAMERA_ID_TO_KEY[camera_id]] = _gap_frame(camera_evidence)

    master_frame = global_alignment.GAP_TIMELINES[master_key]
    global_alignment.MASTER_CAMERA = master_key
    global_alignment.MASTER_TIMELINE = master_frame
    global_alignment.MASTER_GAP_COUNT = int(len(master_frame))
    global_alignment.MASTER_POSITIONS = master_frame[
        "normalized_confirmation_time"].to_numpy(dtype=float)
    global_alignment.MASTER_DURATIONS = master_frame[
        "normalized_duration"].to_numpy(dtype=float)
    global_alignment.MASTER_GAP_ID_LIST = list(master_frame["local_gap_id"])
    global_alignment.GLOBAL_GAP_IDS = [
        "GLOBAL_G%03d" % (index + 1)
        for index in range(global_alignment.MASTER_GAP_COUNT)]
    global_alignment.GLOBAL_GAP_COUNT = global_alignment.MASTER_GAP_COUNT

    # The master is aligned to itself by definition; `estimate_alignment`
    # returns this identity registration rather than matching it to itself.
    identity = global_alignment.CameraAlignment(master_key, master_key)
    identity.scale, identity.offset = 1.0, 0.0
    identity.is_reversed = False
    identity.status = "MASTER"
    identity.fit_note = "identity (canonical gap authority)"
    identity.n_master = identity.n_camera = global_alignment.MASTER_GAP_COUNT
    identity.matches = [
        {"master_index": index, "camera_index": index,
         "master_position": float(global_alignment.MASTER_POSITIONS[index]),
         "camera_position": float(global_alignment.MASTER_POSITIONS[index]),
         "error": 0.0}
        for index in range(global_alignment.MASTER_GAP_COUNT)]
    identity.errors = np.zeros(global_alignment.MASTER_GAP_COUNT, dtype=float)
    identity.matched_master_indices = set(
        range(global_alignment.MASTER_GAP_COUNT))
    identity.matched_camera_indices = set(
        range(global_alignment.MASTER_GAP_COUNT))
    identity.unmatched_camera_indices = []
    global_alignment.CAMERA_ALIGNMENTS[master_key] = identity
    return identity


def align_camera(engine, camera_id: str, camera_evidence: ev.CameraEvidence,
                 ) -> Dict[str, Any]:
    """Align one support camera by calling the engine's own estimator.

    Forward vs reversed is decided by `estimate_alignment`, which tests both
    orientations and adopts a reversal only when the gap sequence proves it.
    Projection back into the camera uses the alignment's own
    `project_to_camera`, which applies the mirror for a reversed camera -- so a
    canonical RIGHT_UP gap lands on the correct frame of a camera that runs
    backwards, with no second algorithm involved.
    """
    global_alignment = engine["global_alignment"]
    camera_key = gc_runner.CAMERA_ID_TO_KEY[camera_id]

    alignment = global_alignment.estimate_alignment(camera_key, verbose=False)
    canonical_positions = list(global_alignment.MASTER_POSITIONS)
    camera_positions = [gap.normalized_position for gap in camera_evidence.gaps]

    detected = {match["master_index"]: match["camera_index"]
                for match in (alignment.matches or [])}
    projected: List[float] = []
    for index, canonical in enumerate(canonical_positions):
        camera_index = detected.get(index)
        if camera_index is not None and camera_index < len(camera_positions):
            # This camera measured the gap: prefer its own observation, exactly
            # as the engine's wagon mapping prefers a DETECTED boundary.
            projected.append(float(camera_positions[camera_index]))
        else:
            # Missed here -> the canonical gap stands and is projected in.
            projected.append(float(alignment.project_to_camera(canonical)))

    extra = [
        {"local_gap_id": camera_evidence.gaps[index].local_gap_id,
         "normalized_position": camera_evidence.gaps[index].normalized_position,
         "note": "seen only by this camera; DIAGNOSTIC, never a canonical wagon"}
        for index in (alignment.unmatched_camera_indices or [])
        if index < len(camera_evidence.gaps)
    ]

    return {
        "matches": [{"master_index": m["master_index"],
                     "camera_index": m["camera_index"],
                     "error": m.get("error", 0.0)}
                    for m in (alignment.matches or [])],
        "scale": float(alignment.scale), "offset": float(alignment.offset),
        "is_reversed": bool(alignment.is_reversed),
        "fit_note": str(alignment.fit_note),
        "status": str(alignment.status),
        "mean_error": (float(alignment.mean_error)
                       if alignment.matched_count else None),
        "extra_camera_gaps": extra,
        "projected_positions": projected,
        "matched_count": int(alignment.matched_count),
    }


def build_harvest(engine, evidences: Dict[str, ev.CameraEvidence],
                 ) -> Tuple[Any, Dict[str, Any]]:
    """Turn sealed evidence into the same harvest shape Batch's Stage 1 emits.

    Reusing `global_counting.runner`'s dataclasses and
    `global_counting.adapter` means Sequential and Batch produce the identical
    canonical contract -- which is what makes the parity tests meaningful.
    """
    master_id = C.MASTER_CAMERA
    master = evidences[master_id]

    # Load the persisted timelines into the engine's alignment state, then let
    # the engine's own estimator decide each camera's direction and mapping.
    populate_engine_alignment_state(
        engine["global_alignment"], evidences, master_id)
    canonical_positions = [float(p) for p in
                           engine["global_alignment"].MASTER_POSITIONS]
    gap_count = len(canonical_positions)

    alignments: Dict[str, Dict[str, Any]] = {}
    cameras: Dict[str, gc_runner.CameraHarvest] = {}
    for camera_id in C.ALL_CAMERAS:
        camera_evidence = evidences.get(camera_id)
        if camera_evidence is None:
            cameras[camera_id] = gc_runner.CameraHarvest(
                camera_id=camera_id, trim_status="NOT_SEALED",
                alignment_status="ABSENT")
            alignments[camera_id] = {"status": "ABSENT", "matches": [],
                                     "extra_camera_gaps": [],
                                     "projected_positions": [],
                                     "matched_count": 0, "is_reversed": False,
                                     "scale": 1.0, "offset": 0.0}
            continue

        if camera_id == master_id:
            alignment = {"matches": [{"master_index": i, "camera_index": i,
                                      "error": 0.0}
                                     for i in range(gap_count)],
                         "scale": 1.0, "offset": 0.0,
                         "fit_note": "identity (canonical gap authority)",
                         "status": "MASTER", "is_reversed": False,
                         "extra_camera_gaps": [],
                         "projected_positions": list(canonical_positions),
                         "matched_count": gap_count}
        else:
            alignment = align_camera(engine, camera_id, camera_evidence)
        alignments[camera_id] = alignment

        timing = camera_evidence.timing
        cameras[camera_id] = gc_runner.CameraHarvest(
            camera_id=camera_id,
            video_path=(camera_evidence.provenance.get("video") or {}).get("path", ""),
            fps=timing.fps, total_frames=timing.total_frames,
            crop_start_frame=timing.wagon_region_start_frame,
            crop_end_frame=timing.wagon_region_end_frame,
            trimmed_total_frames=timing.wagon_region_frames,
            unique_gap_count=camera_evidence.unique_gap_count,
            trim_status=camera_evidence.status,
            alignment_status=str(alignment.get("status")),
            is_reversed=bool(alignment.get("is_reversed", False)),
            scale=float(alignment.get("scale", 1.0)),
            offset=float(alignment.get("offset", 0.0)),
            matched_gaps=int(alignment.get("matched_count", 0)))

    # ---- the ONE canonical roster: N gaps -> N-1 wagons ------------------
    wagons: List[gc_runner.WagonHarvest] = []
    for index in range(max(0, gap_count - 1)):
        wagon = gc_runner.WagonHarvest(
            wagon_number=index + 1,
            global_start_position=canonical_positions[index],
            global_end_position=canonical_positions[index + 1])
        for camera_id in C.ALL_CAMERAS:
            camera_evidence = evidences.get(camera_id)
            alignment = alignments[camera_id]
            projected = alignment.get("projected_positions") or []
            if (camera_evidence is None or len(projected) <= index + 1
                    or camera_evidence.timing.wagon_region_frames <= 0):
                wagon.cameras[camera_id] = {
                    "start_frame": None, "end_frame": None,
                    "status": adapter.STATUS_UNMATCHED, "reversed": False,
                    "start_position": None, "end_position": None}
                continue
            low_position = min(projected[index], projected[index + 1])
            high_position = max(projected[index], projected[index + 1])
            start_frame = _frame_for_position(camera_evidence, low_position)
            end_frame = _frame_for_position(camera_evidence, high_position)
            matched = {m["master_index"]
                       for m in (alignment.get("matches") or [])}
            status = (adapter.STATUS_DETECTED
                      if {index, index + 1} <= matched
                      else adapter.STATUS_RECOVERED)
            wagon.cameras[camera_id] = {
                "start_frame": start_frame, "end_frame": end_frame,
                "status": status,
                "reversed": bool(alignment.get("is_reversed", False)),
                "start_position": low_position, "end_position": high_position}
        wagons.append(wagon)

    # Non-wagon objects outside the master's confirmed region, from the
    # persisted classification timeline (no classifier is re-run).
    leading, trailing = _non_wagon_counts(master)

    harvest = gc_runner.GlobalCountingResult(
        master_camera=master_id,
        global_gap_count=gap_count,
        global_wagon_count=max(0, gap_count - 1),
        wagons=wagons, cameras=cameras,
        engine_dir=str(master.provenance.get("engine_dir", "")),
        engine_output_dir="",
        leading_non_wagon=leading, trailing_non_wagon=trailing)
    return harvest, alignments


def _non_wagon_counts(master: ev.CameraEvidence,
                      ) -> Tuple[Dict[str, int], Dict[str, int]]:
    """ENGINE / BRAKE_VAN runs before and after the master's wagon region."""
    timeline = master.classification_timeline
    if not timeline:
        return {}, {}

    def tally(records) -> Dict[str, int]:
        out: Dict[str, int] = {}
        previous = None
        for record in records:
            mapped = gc_runner.CLASS_MAP.get(
                str(record.get("normalized_class")), C.CLASS_UNKNOWN)
            current = mapped if mapped in (C.CLASS_ENGINE,
                                           C.CLASS_BRAKE_VAN) else None
            if current is not None and current != previous:
                out[current] = out.get(current, 0) + 1
            previous = current
        return out

    start = master.timing.wagon_region_start_frame
    end = master.timing.wagon_region_end_frame
    return (tally([r for r in timeline if int(r.get("frame_id", 0)) < start]),
            tally([r for r in timeline if int(r.get("frame_id", 0)) > end]))


# -----------------------------------------------------------------------------
# Assigning persisted observations to canonical wagons
# -----------------------------------------------------------------------------

def assign_observations(state, evidences: Dict[str, ev.CameraEvidence],
                        ) -> Dict[str, Dict[str, Dict[str, List[Any]]]]:
    """`{gw_id: {feature: {camera_id: [observation, ...]}}}`.

    Ownership is `core.wagon_ownership` -- the same rule Batch uses, so an
    observation before a gap belongs to the previous wagon, after a gap to the
    next, and exactly on a gap to the next. An observation owned by no wagon
    (outside the wagon region) is dropped rather than attached to the nearest
    one.
    """
    ownership = wagon_ownership.for_state(state)
    assigned: Dict[str, Dict[str, Dict[str, List[Any]]]] = {
        wagon.global_id: {} for wagon in state.wagons}

    for camera_id, camera_evidence in evidences.items():
        for observation in camera_evidence.observations:
            gw_id = (ownership.owner_of_camera_frame(camera_id,
                                                    observation.frame_idx)
                     if ownership is not None else None)
            if gw_id is None or gw_id not in assigned:
                continue
            assigned[gw_id].setdefault(observation.feature, {}).setdefault(
                camera_id, []).append(observation)
    return assigned


# -----------------------------------------------------------------------------
# Aggregation -- the EXISTING pure aggregators, on persisted evidence
# -----------------------------------------------------------------------------

def _aggregate_door(gw_id: str, per_camera: Dict[str, List[Any]],
                    frame_size: Tuple[int, int]) -> Dict[str, Any]:
    """Door verdict + every DISTINCT door, using the existing machinery.

    `EvidenceAggregator` groups this wagon's observations into candidates (one
    per physical door) exactly as the sampled Door path does, and the door
    processor's own helpers turn those candidates into `doors[]`, so the
    multi-door behaviour from b6f67b5 is preserved rather than re-derived.
    """
    from features.door import processor as door_proc
    from features.evidence_aggregator import EvidenceAggregator, Observation

    width, height = frame_size
    all_doors: List[Dict[str, Any]] = []
    side_states: Dict[str, Tuple[str, float]] = {}

    for camera_id, observations in sorted(per_camera.items()):
        aggregator = EvidenceAggregator(frame_width=width, frame_height=height,
                                        stride=3)
        by_frame: Dict[int, List[Observation]] = {}
        for observation in observations:
            state = C.DOOR_LABEL_TO_STATE.get(
                observation.raw_class, door_proc._canonical(observation.raw_class))
            bbox = tuple(observation.bbox or (0.0, 0.0, 0.0, 0.0))
            by_frame.setdefault(observation.frame_idx, []).append(Observation(
                frame_idx=observation.frame_idx, state=state,
                confidence=observation.confidence, bbox=bbox,
                score=observation.score))
        for frame_idx in sorted(by_frame):
            aggregator.add_frame(frame_idx, by_frame[frame_idx])

        result = aggregator.finalize()
        doors = door_proc._door_evidence_from_groups(
            camera_id, result["accepted"], {}, width, height)
        all_doors.extend(doors)

        decisions = [{"state": d["state"], "confidence": d["confidence"],
                      "total_hits": d["total_hits"]} for d in doors]
        side_states[camera_id] = door_proc._pick_side_state(decisions)

    ordered = door_proc.order_doors(all_doors)
    indexed = door_proc._indexed_doors(ordered)
    left = side_states.get(C.CAMERA_LEFT_UP, (C.NO_DATA, 0.0))
    right = side_states.get(C.CAMERA_RIGHT_UP, (C.NO_DATA, 0.0))

    return {
        "global_id": gw_id, "feature": "door",
        "status": C.STATUS_OK if all_doors else C.NO_DATA,
        "left_door": left[0], "left_door_confidence": round(left[1], 4),
        "right_door": right[0], "right_door_confidence": round(right[1], 4),
        "doors": [{k: v for k, v in door.items() if k != "_snapshot"}
                  for door in indexed],
        "door_status": door_proc.wagon_door_status(ordered),
        "supporting_cameras": sorted(per_camera),
        "frame_count": sum(len(v) for v in per_camera.values()),
        "source": "global_assembly_from_persisted_evidence",
    }


def _aggregate_damage(gw_id: str, per_camera: Dict[str, List[Any]],
                      frame_size: Tuple[int, int]) -> Dict[str, Any]:
    from features.evidence_aggregator import EvidenceAggregator, Observation

    width, height = frame_size
    tracks: List[Dict[str, Any]] = []
    for camera_id, observations in sorted(per_camera.items()):
        aggregator = EvidenceAggregator(frame_width=width, frame_height=height,
                                        stride=3)
        by_frame: Dict[int, List[Observation]] = {}
        for observation in observations:
            bbox = tuple(observation.bbox or (0.0, 0.0, 0.0, 0.0))
            by_frame.setdefault(observation.frame_idx, []).append(Observation(
                frame_idx=observation.frame_idx, state=observation.raw_class,
                confidence=observation.confidence, bbox=bbox,
                score=observation.score))
        for frame_idx in sorted(by_frame):
            aggregator.add_frame(frame_idx, by_frame[frame_idx])
        for group in aggregator.finalize()["accepted"]:
            best = group.get("best")
            tracks.append({
                "camera_id": camera_id,
                "track_id": int(group["candidate_id"]),
                "class_name": str(group["state"]),
                "confidence": float(group["confidence"]),
                "best_confidence": float(group["confidence"]),
                "best_frame_idx": int(getattr(best, "frame_idx", -1)
                                      if best is not None else -1),
                "bbox": list(getattr(best, "bbox", ()) or ()) or None,
            })
    return {
        "global_id": gw_id, "feature": "damage",
        "status": C.STATUS_OK if per_camera else C.NO_DATA,
        "top_damage": C.DAMAGE_PRESENT if tracks else (
            C.DAMAGE_OK if per_camera else C.NO_DATA),
        "top_damage_details": tracks,
        "supporting_cameras": sorted(per_camera),
        "source": "global_assembly_from_persisted_evidence",
    }


def _aggregate_load(gw_id: str, per_camera: Dict[str, List[Any]],
                    ) -> Dict[str, Any]:
    """Batch's Load rule, with Batch's own threshold constant and branch order.

    `total` counts EVERY sampled observation -- Batch uses `used`, not
    `n_loaded + n_empty` -- so a frame classified as neither loaded nor empty
    still dilutes the ratio. Getting that wrong would shift the verdict on
    wagons with third-class frames, so the constant and the branches are taken
    from the Batch module rather than restated.
    """
    from features.load.processor import _LOADED_RATIO_THRESHOLD, _canonical_load

    loaded_confs: List[float] = []
    empty_confs: List[float] = []
    used = 0
    for observations in per_camera.values():
        for observation in observations:
            used += 1
            canonical = _canonical_load(observation.raw_class)
            if canonical == C.LOAD_LOADED:
                loaded_confs.append(float(observation.confidence))
            elif canonical == C.LOAD_EMPTY:
                empty_confs.append(float(observation.confidence))

    base = {"global_id": gw_id, "feature": "load",
            "frames_used": used, "loaded_frames": len(loaded_confs),
            "empty_frames": len(empty_confs),
            "supporting_cameras": sorted(per_camera),
            "source": "global_assembly_from_persisted_evidence"}

    if used == 0:
        base.update({"status": C.NO_DATA, "load_status": C.NO_DATA,
                     "load_confidence": 0.0})
        return base

    total = max(1, used)
    loaded_ratio = len(loaded_confs) / total
    if loaded_ratio > _LOADED_RATIO_THRESHOLD and loaded_confs:
        base.update({
            "status": C.STATUS_OK, "load_status": C.LOAD_LOADED,
            "load_confidence": round(sum(loaded_confs) / len(loaded_confs), 4)})
        return base
    if empty_confs:
        base.update({
            "status": C.STATUS_OK, "load_status": C.LOAD_EMPTY,
            "load_confidence": round(sum(empty_confs) / len(empty_confs), 4)})
        return base
    base.update({"status": C.NO_DATA, "load_status": C.NO_DATA,
                 "load_confidence": 0.0})
    return base


# -----------------------------------------------------------------------------
# Public entry
# -----------------------------------------------------------------------------

def assemble(*, workspace: str, repo_root: str, batch_key: str,
             engine_dir: Optional[str] = None,
             cameras: Sequence[str] = C.ALL_CAMERAS,
             verbose: bool = True) -> AssemblyResult:
    """Build the ONE canonical global train from sealed camera evidence."""
    if verbose:
        print("[SEQ] GLOBAL ASSEMBLY START")

    ready, sealed, missing, reason = readiness(workspace, cameras=cameras)
    if not ready:
        if verbose:
            print("[SEQ] GLOBAL ASSEMBLY NOT READY: %s" % reason)
            print("[SEQ] sealed=%s missing=%s" % (sealed, missing))
            print("[SEQ] no combined report is produced; the per-camera "
                  "reports remain valid on their own")
        return AssemblyResult(ready=False, reason=reason, cameras_used=sealed,
                              cameras_missing=missing)

    evidences = {camera_id: ev.load_evidence(workspace, camera_id)
                 for camera_id in sealed}
    usable = {camera_id: e for camera_id, e in evidences.items()
              if e.status == ev.STATUS_SEALED and e.timing.wagon_region_frames > 0}
    if C.MASTER_CAMERA not in usable:
        reason = ("%s has no confirmed wagon region, so no canonical gap "
                  "sequence exists" % C.MASTER_CAMERA)
        if verbose:
            print("[SEQ] GLOBAL ASSEMBLY NOT READY: %s" % reason)
        return AssemblyResult(ready=False, reason=reason,
                              cameras_used=sorted(usable),
                              cameras_missing=missing)

    resolved_engine = gc_runner.locate_engine(repo_root, engine_dir)
    with gc_runner.engine_session(resolved_engine):
        import config, global_alignment
        harvest, alignments = build_harvest(
            {"global_alignment": global_alignment, "config": config}, usable)

    if harvest.global_wagon_count <= 0:
        reason = ("only %d canonical gap(s) on %s; at least two are needed to "
                  "form a wagon" % (harvest.global_gap_count, C.MASTER_CAMERA))
        if verbose:
            print("[SEQ] GLOBAL ASSEMBLY NOT READY: %s" % reason)
        return AssemblyResult(ready=False, reason=reason,
                              cameras_used=sorted(usable),
                              cameras_missing=missing,
                              global_gap_count=harvest.global_gap_count)

    # ---- the canonical contract, built by the SAME adapter Batch uses ----
    global_state_dir = os.path.join(workspace, "global_state")
    state_path, tracking_path = adapter.write_documents(harvest, global_state_dir)
    state = load_global_train_state(state_path)
    problems = verify_roster_integrity(state)
    if problems:
        raise AssemblyNotReady(
            "assembled roster is malformed: %s" % "; ".join(problems[:5]))

    if verbose:
        print("[SEQ/ASSEMBLY] canonical gaps   : %d" % state.global_gap_count)
        print("[SEQ/ASSEMBLY] canonical roster : GW_1..GW_%d" % state.total_wagons)
        print("[SEQ/ASSEMBLY] gap authority    : %s" % state.master_camera)
        for camera_id in C.ALL_CAMERAS:
            alignment = alignments.get(camera_id) or {}
            extra = alignment.get("extra_camera_gaps") or []
            print("[SEQ/ASSEMBLY]   %-14s matched=%-3s extra(diagnostic)=%-3d %s"
                  % (camera_id, alignment.get("matched_count", 0), len(extra),
                     alignment.get("status")))

    # ---- assign + aggregate (no inference) ------------------------------
    assigned = assign_observations(state, usable)
    master = usable[C.MASTER_CAMERA]
    frame_size = (int(master.provenance.get("frame_width") or 1920),
                  int(master.provenance.get("frame_height") or 1080))

    states_root = os.path.join(workspace, "wagon_states")
    counts = {"door": 0, "damage": 0, "load": 0}
    for wagon in state.wagons:
        features = assigned.get(wagon.global_id) or {}
        payloads = []
        if features.get("door"):
            payloads.append(_aggregate_door(wagon.global_id, features["door"],
                                            frame_size))
            counts["door"] += 1
        if features.get("damage"):
            payloads.append(_aggregate_damage(wagon.global_id,
                                              features["damage"], frame_size))
            counts["damage"] += 1
        if features.get("load"):
            payloads.append(_aggregate_load(wagon.global_id, features["load"]))
            counts["load"] += 1
        for payload in payloads:
            directory = os.path.join(states_root, payload["feature"])
            os.makedirs(directory, exist_ok=True)
            with open(os.path.join(directory, "%s.json" % wagon.global_id),
                      "w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2, default=str)

    if verbose:
        print("[SEQ/ASSEMBLY] wagons with evidence: door=%d damage=%d load=%d"
              % (counts["door"], counts["damage"], counts["load"]))

    # ---- fusion + the single combined report (Batch's own stages) --------
    from fusion import wagon_state_builder
    from reporting import combined_train_report

    unified = wagon_state_builder.build(
        state=state, wagon_states_root=states_root, verbose=verbose)

    output_dir = ev.combined_dir(workspace)
    report = combined_train_report.build(
        state=state, unified=unified, output_dir=output_dir,
        batch_key=batch_key, wagon_states_root=states_root, verbose=verbose)

    diagnostics = {
        "schema": ASSEMBLY_SCHEMA,
        "cameras_used": sorted(usable),
        "cameras_missing": missing,
        "alignments": {
            camera_id: {k: v for k, v in (alignments.get(camera_id) or {}).items()
                        if k != "matches"}
            for camera_id in C.ALL_CAMERAS},
        "assigned_wagons": counts,
    }
    with open(os.path.join(output_dir, "global_assembly.json"), "w",
              encoding="utf-8") as handle:
        json.dump(diagnostics, handle, indent=2, default=str)

    if verbose:
        print("[SEQ] GLOBAL ASSEMBLY COMPLETE  gaps=%d wagons=%d"
              % (state.global_gap_count, state.total_wagons))

    return AssemblyResult(
        ready=True, reason="assembled", cameras_used=sorted(usable),
        cameras_missing=missing, global_gap_count=state.global_gap_count,
        global_wagon_count=state.total_wagons, state_json_path=state_path,
        report_json_path=report.get("json_path"),
        report_pdf_path=report.get("pdf_path"), diagnostics=diagnostics)
