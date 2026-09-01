#!/usr/bin/env python3
"""Validate a combined_train_report.json against the schema the backend agreed.

Checks only what the backend contract promises, and reports an OMITTED field as
such rather than as a failure -- the writer omits `problem_frames` when a train
has no damage and `s3_url` when a door has no snapshot, by design.

usage: python3 validate_global_json.py <path to combined_train_report.json>
"""
import json
import re
import sys

ANGLES = ("left_up", "right_up", "right_top", "left_top")
POSITIONS = ("start", "mid1", "mid2", "end")

ok = []
warn = []
bad = []


def check(cond, msg, soft=False):
    (ok if cond else (warn if soft else bad)).append(msg)


def main(path):
    doc = json.load(open(path))
    print("=" * 74)
    print("  GLOBAL JSON VALIDATION  %s" % path.rsplit("/", 1)[-1])
    print("=" * 74)

    # ---- 1. batch_key -----------------------------------------------------
    bk = doc.get("batch_key")
    check(bool(bk), "batch_key present: %r" % bk)
    if bk:
        check(bool(re.fullmatch(r"\d{8}_\d{6}", bk)),
              "batch_key is YYYYMMDD_HHMMSS")

    # ---- 2. per-camera source URLs (the timestamp fix) -------------------
    tm = doc.get("train_metadata") or {}
    svu = tm.get("source_video_urls") or {}
    check(len(svu) == 4, "train_metadata.source_video_urls has 4 cameras (got %d)"
          % len(svu))
    stamps = {}
    for cam, url in sorted(svu.items()):
        m = re.search(r"(\d{8}_\d{6})", url.rsplit("/", 1)[-1])
        stamps[cam] = m.group(1) if m else None
        print("   %-14s %s" % (cam, url.rsplit("/", 1)[-1]))
    distinct = len({v for v in stamps.values() if v})
    check(distinct > 1,
          "per-camera timestamps DIFFER (%d distinct) -- camera skew preserved"
          % distinct)

    # ---- 3. wagons --------------------------------------------------------
    wagons = doc.get("wagons") or []
    check(bool(wagons), "wagons present: %d" % len(wagons))

    wf_wagons = [w for w in wagons if w.get("wagon_frames")]
    check(bool(wf_wagons), "wagons carrying wagon_frames: %d/%d"
          % (len(wf_wagons), len(wagons)))

    # angle + position coverage
    angle_hits = {a: 0 for a in ANGLES}
    pos_problems = []
    url_problems = []
    for w in wf_wagons:
        frames = w["wagon_frames"]
        for angle, entries in frames.items():
            if angle in angle_hits:
                angle_hits[angle] += 1
            positions = [e.get("position") for e in entries]
            if positions and positions != list(POSITIONS):
                pos_problems.append("%s/%s -> %s"
                                    % (w.get("global_id"), angle, positions))
            for e in entries:
                u = e.get("s3_url") or ""
                if not u.startswith("https://"):
                    url_problems.append("%s/%s/%s" % (w.get("global_id"), angle,
                                                      e.get("position")))
    for a in ANGLES:
        check(angle_hits[a] > 0, "angle %-10s on %d wagons" % (a, angle_hits[a]),
              soft=(angle_hits[a] == 0))
    check(not pos_problems,
          "every wagon_frames angle has start/mid1/mid2/end in order"
          + ("" if not pos_problems else "  BAD: %s" % pos_problems[:3]))
    check(not url_problems,
          "every wagon_frames entry has an https s3_url"
          + ("" if not url_problems else "  BAD: %s" % url_problems[:3]))

    # ---- 4. doors ---------------------------------------------------------
    door_total = door_with_url = 0
    for w in wagons:
        for d in (w.get("doors") or []):
            door_total += 1
            if d.get("s3_url"):
                door_with_url += 1
    if door_total:
        check(door_with_url > 0,
              "doors[].s3_url on %d/%d doors" % (door_with_url, door_total),
              soft=(door_with_url == 0))
    else:
        warn.append("no doors[] in this report at all")

    # ---- 5. problem_frames (absent is CORRECT on a clean train) ----------
    pf = sum(len(w.get("problem_frames") or []) for w in wagons)
    if pf:
        ok.append("problem_frames: %d entries" % pf)
    else:
        warn.append("problem_frames: none -- correct IF this train had no damage")

    # ---- 6. bucket sanity -------------------------------------------------
    blob = json.dumps(doc)
    for stale in ("sarva", "biro-", "wagon-eye-models", "complete-train"):
        check(stale not in blob, "no stale %r reference in the document" % stale)
    buckets = set(re.findall(r"https://([a-z0-9.-]+)\.s3\.", blob))
    print("\n   S3 hosts referenced: %s" % (sorted(buckets) or "(none)"))

    # ---- report -----------------------------------------------------------
    print()
    for m in ok:
        print("[PASS] %s" % m)
    for m in warn:
        print("[WARN] %s" % m)
    for m in bad:
        print("[FAIL] %s" % m)
    print("\npassed: %d   warnings: %d   failed: %d" % (len(ok), len(warn), len(bad)))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
