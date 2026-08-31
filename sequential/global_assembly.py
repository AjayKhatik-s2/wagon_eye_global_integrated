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

KNOWN GAP -- read before trusting Sequential feature numbers
------------------------------------------------------------
The camera stage stores RAW detections, which is deliberate: filtering is a
decision, and decisions belong here. But Batch's Door and Damage processors
apply their own per-model confidence gate BEFORE aggregating (`min_conf` for
door, `confidence_floor` for damage), and this module currently relies on
`EvidenceAggregator`'s acceptance rules alone. Gap detection, the canonical
timeline, the roster, ownership and Load are unaffected; Door and Damage
verdicts could differ from Batch on marginal detections.

So Sequential's STRUCTURE (gap count, roster, boundaries, ownership) is
test-covered, while its Door/Damage verdicts are NOT yet proven equal to
Batch. A Batch-vs-Sequential parity run on real footage is the outstanding
validation step; until it passes, treat Batch as the authority for feature
facts. Persisting the gates in the evidence and applying them here is the
intended fix.
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


def align_camera(engine, canonical_positions: Sequence[float],
                 camera_evidence: ev.CameraEvidence,
                 ) -> Dict[str, Any]:
    """Match one support camera's local gaps against the canonical sequence.

    Uses the engine's validated `monotonic_gap_match` (order preserving, no
    crossing matches) and `robust_linear_fit`, both pure. Unmatched CAMERA gaps
    are recorded as diagnostics and never become wagons.
    """
    global_alignment = engine["global_alignment"]
    config = engine["config"]

    camera_positions = [gap.normalized_position for gap in camera_evidence.gaps]
    if not canonical_positions or not camera_positions:
        return {"matches": [], "scale": 1.0, "offset": 0.0,
                "status": "NO_GAPS", "extra_camera_gaps": [],
                "projected_positions": list(canonical_positions),
                "matched_count": 0}

    matches = global_alignment.monotonic_gap_match(
        list(canonical_positions), camera_positions,
        float(config.GLOBAL_ALIGNMENT_TOLERANCE))

    scale, offset, used, note = 1.0, 0.0, 0, "identity"
    if len(matches) >= 2:
        scale, offset, used, note = global_alignment.robust_linear_fit(
            [canonical_positions[m["master_index"]] for m in matches],
            [camera_positions[m["camera_index"]] for m in matches])
    elif len(matches) == 1:
        offset = (camera_positions[matches[0]["camera_index"]]
                  - canonical_positions[matches[0]["master_index"]])
        note = "single match: offset only"

    matched_camera = {m["camera_index"] for m in matches}
    extra = [
        {"local_gap_id": camera_evidence.gaps[index].local_gap_id,
         "normalized_position": camera_evidence.gaps[index].normalized_position,
         "note": "seen only by this camera; DIAGNOSTIC, never a canonical wagon"}
        for index in range(len(camera_positions)) if index not in matched_camera
    ]

    by_master = {m["master_index"]: m for m in matches}
    projected: List[float] = []
    for index, canonical in enumerate(canonical_positions):
        match = by_master.get(index)
        if match is not None:
            # This camera measured it: prefer its own observation.
            projected.append(camera_positions[match["camera_index"]])
        else:
            # Missed here -> the canonical gap still holds and is projected in.
            projected.append(float(scale) * float(canonical) + float(offset))

    return {
        "matches": matches,
        "scale": float(scale), "offset": float(offset),
        "fit_note": note, "fit_points": int(used),
        "status": "RESOLVED" if matches else "UNRESOLVED",
        "extra_camera_gaps": extra,
        "projected_positions": projected,
        "matched_count": len(matches),
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
    canonical_positions = [gap.normalized_position for gap in master.gaps]
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
                                     "matched_count": 0,
                                     "scale": 1.0, "offset": 0.0}
            continue

        if camera_id == master_id:
            alignment = {"matches": [{"master_index": i, "camera_index": i,
                                      "error": 0.0, "score": 1.0}
                                     for i in range(gap_count)],
                         "scale": 1.0, "offset": 0.0, "fit_note": "authority",
                         "fit_points": gap_count, "status": "MASTER",
                         "extra_camera_gaps": [],
                         "projected_positions": list(canonical_positions),
                         "matched_count": gap_count}
        else:
            alignment = align_camera(engine, canonical_positions, camera_evidence)
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
            is_reversed=False,
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
                "status": status, "reversed": False,
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
    """The existing Load rule: `loaded / total > 0.35`, confidence = winner mean."""
    from features.load import processor as load_proc

    loaded, empty = [], []
    for observations in per_camera.values():
        for observation in observations:
            canonical = load_proc._canonical_load(observation.raw_class)
            (loaded if canonical == C.LOAD_LOADED else empty).append(
                observation.confidence)
    total = len(loaded) + len(empty)
    if not total:
        return {"global_id": gw_id, "feature": "load", "status": C.NO_DATA,
                "load_status": C.NO_DATA, "load_confidence": 0.0,
                "source": "global_assembly_from_persisted_evidence"}
    is_loaded = (len(loaded) / float(total)) > 0.35
    winners = loaded if is_loaded else empty
    return {
        "global_id": gw_id, "feature": "load", "status": C.STATUS_OK,
        "load_status": C.LOAD_LOADED if is_loaded else C.LOAD_EMPTY,
        "load_confidence": round(sum(winners) / max(1, len(winners)), 4),
        "frames_used": total, "loaded_frames": len(loaded),
        "empty_frames": len(empty),
        "supporting_cameras": sorted(per_camera),
        "source": "global_assembly_from_persisted_evidence",
    }


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
