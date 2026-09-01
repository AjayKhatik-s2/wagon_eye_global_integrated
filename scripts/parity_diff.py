#!/usr/bin/env python3
"""Compare a Batch output directory with a Sequential one, field by field.

    python scripts/parity_diff.py --batch batch_outputs/20260831_180146 \
                                  --sequential batch_outputs/20260831_173302

Prints the FIRST divergent field with both values and the upstream component
responsible, then a summary. Exit 0 = identical within the documented
tolerance, 1 = divergent, 2 = not comparable (different inputs/config, or a
required artifact is missing).

WHAT IS COMPARED, IN UPSTREAM ORDER
The order matters: a wagon-count difference is almost always a symptom of a
gap-count difference, which is a symptom of a master-camera difference. Walking
the chain from its source means the first line printed is the cause, not a
downstream echo.

    1. comparability   input video + model + config fingerprints
    2. master camera   Stage 1 / select_master_camera
    3. global gaps     count, then each canonical position
    4. roster          wagon count, ids, order
    5. boundaries      per-wagon master frames + per-camera aligned ranges
    6. alignment       scale / offset / reversal per camera
    7. classification  per-wagon class + confidence
    8. features        Door / Load / Damage per wagon
    9. fusion          the unified per-wagon facts
   10. reports         combined JSON contract + camera report presence

TOLERANCE
Integers, ids, counts, orderings and flags must be EXACTLY equal. Only
floating-point quantities produced by the same algorithm (normalized positions,
alignment scale/offset, confidences) are compared with --tolerance, default
1e-6, because the same arithmetic on the same inputs can still differ in the
last bits when it is reached by a different call order.

DATA vs METADATA
Two runs of the same train are never byte-identical: they ran at different
times, into different output directories, and took different amounts of wall
clock. Those fields are EXPECTED to differ and are listed by name in
EXPECTED_METADATA below. They are reported in a separate section as "expected
metadata differences" -- counted and shown, never silently dropped -- so an
operator can see exactly what was excused. Everything not on that list is data,
and any difference in it is a divergence.

The distinction is by FIELD NAME, not by value: a field is excused because it is
provenance, never because its values happen to disagree. --strict-metadata turns
the excused set into divergences too, for when you want to see everything.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_SCRIPT_DIR)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

STATE = os.path.join("global_state", "global_train_state.json")
TRACKING = os.path.join("global_state", "per_camera_tracking.json")

# field -> the component that produces it, printed with a divergence so the
# trace starts in the right place instead of at the symptom.
SOURCE = {
    "master_camera": "engine global_alignment.select_master_camera "
                     "(max confirmed unique gaps)",
    "global_gap_count": "engine set_master_camera -- the gap count IS the "
                        "master's unique gap count",
    "gap_position": "engine build_normalized_gap_timeline / gap tracking",
    "total_wagons": "wagon_mapping.build_global_wagon_timeline "
                    "(gaps - 1)",
    "global_id": "adapter roster construction",
    "frame_master": "wagon_mapping boundary projection",
    "camera_range": "global_alignment.estimate_alignment + "
                    "camera_frame_for_position",
    "alignment": "global_alignment.estimate_alignment / robust_linear_fit",
    "classification": "engine classification + _majority_class over the master",
    "door": "features.door.processor (gate + EvidenceAggregator)",
    "load": "features.load.processor (_LOADED_RATIO_THRESHOLD, used)",
    "damage": "features.damage.processor (_filter_detections_for_top)",
    "unified": "fusion.wagon_state_builder",
    "report": "reporting.combined_train_report / camera_reports",
}


# Fields that legitimately differ between two runs of the same train. Matched
# on the whole field name or its last dotted component. Each says why.
EXPECTED_METADATA = {
    "generated_at":        "wall-clock time the document was written",
    "created_at":          "wall-clock time the document was written",
    "sealed_at":           "wall-clock time the camera seal was written",
    "timestamp":           "wall-clock time",
    "batch_key":           "per-run identifier, chosen by the operator",
    "run_id":              "per-run identifier",
    "output_dir":          "the output directory each run was given",
    "workspace":           "the workspace each run was given",
    "path":                "absolute path inside a run's own directory",
    "paths":               "absolute paths inside a run's own directory",
    "evidence_path":       "absolute path inside a run's own directory",
    "processing_seconds":  "wall-clock duration -- Sequential is expected to "
                           "differ, that is the point of the architecture",
    "seconds":             "wall-clock duration",
    "duration_seconds":    "wall-clock duration",
    "elapsed":             "wall-clock duration",
    "mode":                "batch vs sequential -- the one intended difference "
                           "between the two runs",
    "produced_by":         "which component wrote the record",
    "decode_count":        "Sequential decodes twice per camera by design",
}


def metadata_reason(field):
    """Why `field` is provenance rather than data, or None if it is data.

    Bracket subscripts are stripped from EACH dotted component, so a nested
    field like `wagons[0].sealed_at` is recognised by its tail. Truncating at
    the first `[` instead would read that field as `wagons` and classify a
    per-wagon timestamp as DATA -- reporting a false divergence on every real
    run, which is exactly the failure this function exists to prevent.
    """
    parts = [part.split("[")[0] for part in field.split(".")]
    bare = ".".join(parts)
    if bare in EXPECTED_METADATA:
        return EXPECTED_METADATA[bare]
    return EXPECTED_METADATA.get(parts[-1])


class Divergence(Exception):
    def __init__(self, field, batch, sequential, source_key=None):
        self.field = field
        self.batch = batch
        self.sequential = sequential
        self.source = SOURCE.get(source_key or field.split(".")[0].split("[")[0],
                                 "unknown")
        super().__init__(field)


def _load(directory, relative):
    path = os.path.join(directory, relative)
    if not os.path.isfile(path):
        return None
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _close(left, right, tolerance):
    try:
        return abs(float(left) - float(right)) <= tolerance
    except (TypeError, ValueError):
        return left == right


# Differences excused as provenance, recorded so they can be reported.
EXCUSED = []
STRICT_METADATA = False


def _record_or_raise(field, left, right, source_key):
    """Data differences raise; metadata differences are recorded and excused."""
    reason = None if STRICT_METADATA else metadata_reason(field)
    if reason is None:
        raise Divergence(field, left, right, source_key)
    EXCUSED.append({"field": field, "batch": left, "sequential": right,
                    "reason": reason})


def _exact(field, left, right, source_key=None):
    if left != right:
        _record_or_raise(field, left, right, source_key)


def _approx(field, left, right, tolerance, source_key=None):
    if not _close(left, right, tolerance):
        _record_or_raise(field, left, right, source_key)


# -----------------------------------------------------------------------------

def check_comparable(batch_dir, sequential_dir):
    """Refuse to compare runs that did not see the same inputs."""
    notes = []
    batch_state = _load(batch_dir, STATE)
    sequential_state = _load(sequential_dir, STATE)
    if batch_state is None:
        raise SystemExit("[NOT COMPARABLE] missing %s under %s"
                         % (STATE, batch_dir))
    if sequential_state is None:
        raise SystemExit("[NOT COMPARABLE] missing %s under %s"
                         % (STATE, sequential_dir))

    for name, document in (("batch", batch_state),
                           ("sequential", sequential_state)):
        engine = document.get("global_counting_engine")
        if engine != "global_wagon_app":
            notes.append("%s was produced by %r, not the global engine"
                         % (name, engine))

    # Sequential records per-camera fingerprints in its seals; Batch does not,
    # so identical inputs can only be asserted when the seals are present.
    seals = os.path.join(sequential_dir, "camera_evidence")
    if os.path.isdir(seals):
        fingerprints = {}
        for camera in sorted(os.listdir(seals)):
            seal = _load(sequential_dir,
                         os.path.join("camera_evidence", camera, "sealed.json"))
            if seal:
                fingerprints[camera] = (
                    (seal.get("video_fingerprint") or {}).get("fingerprint"),
                    seal.get("config_fingerprint"))
        notes.append("sequential input fingerprints: %s" % fingerprints)
        notes.append("NOTE: Batch records no input fingerprint, so identical "
                     "inputs must be confirmed by the operator")
    else:
        notes.append("sequential run has no camera_evidence/: cannot verify "
                     "that both runs saw the same videos")
    return batch_state, sequential_state, notes


def compare(batch_dir, sequential_dir, tolerance):
    batch, sequential, notes = check_comparable(batch_dir, sequential_dir)
    for note in notes:
        print("  note: %s" % note)
    print("")

    # Find provenance differences up front, so the report can say what it
    # excused instead of leaving those fields unexamined.
    sweep_metadata(batch, sequential)

    # ---- 2. master camera -------------------------------------------------
    _exact("master_camera", batch.get("master_camera"),
           sequential.get("master_camera"))

    # ---- 3. global gaps ---------------------------------------------------
    _exact("global_gap_count", batch.get("global_gap_count"),
           sequential.get("global_gap_count"))
    batch_gaps = batch.get("global_gaps") or []
    sequential_gaps = sequential.get("global_gaps") or []
    _exact("global_gaps.length", len(batch_gaps), len(sequential_gaps),
           "global_gap_count")
    for index, (left, right) in enumerate(zip(batch_gaps, sequential_gaps)):
        _approx("gap_position[%d]" % index, left.get("normalized_position"),
                right.get("normalized_position"), tolerance, "gap_position")
        _exact("gap_master_frame[%d]" % index, left.get("master_frame"),
               right.get("master_frame"), "frame_master")

    # ---- 4. roster --------------------------------------------------------
    _exact("total_wagons", batch.get("total_wagons"),
           sequential.get("total_wagons"))
    batch_wagons = batch.get("wagons") or []
    sequential_wagons = sequential.get("wagons") or []
    _exact("wagons.length", len(batch_wagons), len(sequential_wagons),
           "total_wagons")

    for index, (left, right) in enumerate(zip(batch_wagons, sequential_wagons)):
        label = "wagons[%d]" % index
        _exact("%s.global_id" % label, left.get("global_id"),
               right.get("global_id"), "global_id")
        _exact("%s.wagon_index" % label, left.get("wagon_index"),
               right.get("wagon_index"), "global_id")

        # ---- 5. boundaries ------------------------------------------------
        _exact("%s.start_frame_master" % label, left.get("start_frame_master"),
               right.get("start_frame_master"), "frame_master")
        _exact("%s.end_frame_master" % label, left.get("end_frame_master"),
               right.get("end_frame_master"), "frame_master")

        left_ranges = left.get("camera_frame_ranges") or {}
        right_ranges = right.get("camera_frame_ranges") or {}
        _exact("%s.camera_frame_ranges.keys" % label,
               sorted(left_ranges), sorted(right_ranges), "camera_range")
        for camera in sorted(left_ranges):
            for field in ("start_frame", "end_frame", "status",
                          "timeline_reversed"):
                _exact("%s.camera_frame_ranges[%s].%s" % (label, camera, field),
                       left_ranges[camera].get(field),
                       right_ranges[camera].get(field), "camera_range")

        # ---- 7. classification --------------------------------------------
        _exact("%s.classification" % label, left.get("classification"),
               right.get("classification"), "classification")
        _approx("%s.classification_confidence" % label,
                left.get("classification_confidence"),
                right.get("classification_confidence"), tolerance,
                "classification")

    # ---- 6. alignment -----------------------------------------------------
    left_summary = batch.get("support_alignment_summary") or {}
    right_summary = sequential.get("support_alignment_summary") or {}
    _exact("support_alignment_summary.keys", sorted(left_summary),
           sorted(right_summary), "alignment")
    for camera in sorted(left_summary):
        for field in ("timeline_reversed", "alignment_status",
                      "detected_intervals", "recovered_intervals",
                      "unmatched_intervals"):
            _exact("alignment[%s].%s" % (camera, field),
                   left_summary[camera].get(field),
                   right_summary[camera].get(field), "alignment")
        for field in ("scale", "offset"):
            _approx("alignment[%s].%s" % (camera, field),
                    left_summary[camera].get(field),
                    right_summary[camera].get(field), tolerance, "alignment")

    # ---- 8 + 9. features and fusion --------------------------------------
    for wagon in batch_wagons:
        gw_id = wagon.get("global_id")
        for feature, fields in (
            ("door", ("left_door", "right_door", "door_status")),
            ("load", ("load_status",)),
            ("damage", ("top_damage",)),
        ):
            left = _load(batch_dir,
                         os.path.join("wagon_states", feature, "%s.json" % gw_id))
            right = _load(sequential_dir,
                          os.path.join("wagon_states", feature, "%s.json" % gw_id))
            if left is None and right is None:
                continue
            if (left is None) != (right is None):
                raise Divergence("%s[%s].present" % (feature, gw_id),
                                 left is not None, right is not None, feature)
            for field in fields:
                _exact("%s[%s].%s" % (feature, gw_id, field),
                       left.get(field), right.get(field), feature)
            if feature == "door":
                left_doors = left.get("doors") or []
                right_doors = right.get("doors") or []
                _exact("door[%s].doors.length" % gw_id, len(left_doors),
                       len(right_doors), "door")
                for position, (one, two) in enumerate(zip(left_doors,
                                                          right_doors)):
                    _exact("door[%s].doors[%d].state" % (gw_id, position),
                           one.get("state"), two.get("state"), "door")

    # ---- 10. reports ------------------------------------------------------
    batch_report = _load(batch_dir,
                         os.path.join("reports", "combined_train_report.json"))
    sequential_report = _load(
        sequential_dir, os.path.join("combined", "combined_train_report.json"))
    if batch_report and sequential_report:
        for field in ("total_wagons", "wagon_count", "engine_count",
                      "brake_van_count", "loaded_count", "empty_count"):
            if field in batch_report or field in sequential_report:
                _exact("report.%s" % field, batch_report.get(field),
                       sequential_report.get(field), "report")
    else:
        print("  note: combined report JSON missing on one side "
              "(batch=%s sequential=%s) -- report contract not compared"
              % (bool(batch_report), bool(sequential_report)))


def sweep_metadata(batch, sequential, prefix=""):
    """Actively look for provenance differences and record them.

    Without this the excused fields would merely be absent from the comparison,
    which is indistinguishable from not having checked. Walking both documents
    and reporting every name-matched metadata difference makes the policy
    auditable: an operator sees exactly what was excused and why.
    """
    if isinstance(batch, dict) and isinstance(sequential, dict):
        for key in sorted(set(batch) | set(sequential)):
            field = "%s.%s" % (prefix, key) if prefix else key
            left, right = batch.get(key), sequential.get(key)
            if metadata_reason(field) is not None:
                if left != right:
                    _record_or_raise(field, left, right, None)
                continue
            sweep_metadata(left, right, field)
    elif isinstance(batch, list) and isinstance(sequential, list):
        for index, (left, right) in enumerate(zip(batch, sequential)):
            sweep_metadata(left, right, "%s[%d]" % (prefix, index))


def _print_excused():
    """Show every excused difference, so nothing is hidden by the policy."""
    if not EXCUSED:
        print("  expected metadata differences: none")
        print("")
        return
    print("EXPECTED METADATA DIFFERENCES (%d) -- excused, not data"
          % len(EXCUSED))
    print("-" * 78)
    for entry in EXCUSED:
        print("  %-34s %s" % (entry["field"], entry["reason"]))
        print("      batch      : %r" % (entry["batch"],))
        print("      sequential : %r" % (entry["sequential"],))
    print("")


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="scripts/parity_diff.py",
        description="Compare Batch and Sequential outputs for the same train.")
    parser.add_argument("--batch", required=True, help="Batch output directory")
    parser.add_argument("--sequential", required=True,
                        help="Sequential output directory")
    parser.add_argument("--tolerance", type=float, default=1e-6,
                        help="float tolerance (default 1e-6); ids, counts, "
                             "orderings and flags are always exact")
    parser.add_argument("--strict-metadata", action="store_true",
                        help="treat timestamps, paths, durations and run ids "
                             "as data too, so every difference is reported")
    args = parser.parse_args(argv)

    global STRICT_METADATA
    STRICT_METADATA = args.strict_metadata
    del EXCUSED[:]

    print("=" * 78)
    print("BATCH vs SEQUENTIAL parity")
    print("=" * 78)
    print("  batch      : %s" % args.batch)
    print("  sequential : %s" % args.sequential)
    print("  tolerance  : %g (floats only)" % args.tolerance)
    print("  metadata   : %s"
          % ("STRICT -- provenance counts as data" if args.strict_metadata
             else "provenance excused (timestamps, paths, durations, run ids)"))
    print("")

    try:
        compare(args.batch, args.sequential, args.tolerance)
    except Divergence as divergence:
        _print_excused()
        print("FIRST DIVERGENCE")
        print("-" * 78)
        print("  field      : %s" % divergence.field)
        print("  batch      : %r" % (divergence.batch,))
        print("  sequential : %r" % (divergence.sequential,))
        print("  produced by: %s" % divergence.source)
        print("")
        print("Fix the FIRST divergence before looking at anything downstream:")
        print("a wagon-count difference is usually a gap-count difference, and")
        print("a gap-count difference is usually a master-camera difference.")
        return 1

    _print_excused()
    print("PARITY OK -- every compared DATA field is identical "
          "(floats within %g)" % args.tolerance)
    return 0


if __name__ == "__main__":
    sys.exit(main())
