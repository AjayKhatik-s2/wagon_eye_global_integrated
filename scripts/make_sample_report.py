#!/usr/bin/env python3
"""Trim a real combined_train_report.json down to a shareable sample.

Produces the same document, with the same keys in the same order, carrying two
real wagons instead of sixty: one clean and -- when the train has any -- one with
damage, so `problem_frames` and `top_damage_details` are present rather than
having to be described in prose.

Nothing is invented. Every value, URL and frame number is what the pipeline
actually wrote, which is the point: a hand-written sample drifts from the real
schema the moment either changes, and a backend built against a fabricated
`processed_video_urls` would expect links the sequential pipeline never produces.

usage: python3 scripts/make_sample_report.py <combined_train_report.json> [out.json]
"""
import json
import sys


def pick(wagons):
    """One clean wagon and one damaged one, in roster order."""
    damaged = [w for w in wagons if w.get("problem_frames")
               or (w.get("top_damage_details") or w.get("side_damage_details"))]
    clean = [w for w in wagons if w not in damaged]
    out = []
    if clean:
        out.append(clean[0])
    if damaged:
        out.append(damaged[0])
    # A train with no damage at all still gets two wagons, so the sample shows
    # the repeating shape rather than a single object.
    if len(out) < 2:
        out = wagons[:2]
    return sorted(out, key=lambda w: w.get("wagon_index") or 0)


def main(src, dst):
    doc = json.load(open(src))
    wagons = doc.get("wagons") or []
    keep = pick(wagons)
    kept_ids = {w.get("global_id") for w in keep}

    doc["wagons"] = keep
    # evidence_pages is keyed by wagon, so trim it to match -- a sample whose
    # index references wagons it does not contain is a misleading sample.
    if isinstance(doc.get("evidence_pages"), dict):
        doc["evidence_pages"] = {k: v for k, v in doc["evidence_pages"].items()
                                 if k in kept_ids}
    doc["_sample_note"] = (
        "Trimmed from the real report for %s: %d of %d wagons kept, every other "
        "field verbatim. Full document: train_batch/%s/combined/"
        "combined_train_report.json"
        % (doc.get("batch_key"), len(keep), len(wagons), doc.get("batch_key")))

    with open(dst, "w") as fh:
        json.dump(doc, fh, indent=2)

    dmg = sum(len(w.get("problem_frames") or []) for w in keep)
    print("wrote %s" % dst)
    print("  batch_key        : %s" % doc.get("batch_key"))
    print("  wagons kept      : %s (of %d)"
          % (", ".join(sorted(kept_ids)), len(wagons)))
    print("  problem_frames   : %d" % dmg)
    print("  wagon_frames     : %s"
          % ", ".join(sorted((keep[0].get("wagon_frames") or {}).keys())))
    if not (doc.get("train_metadata", {}) or {}).get("processed_video_urls"):
        print("  NOTE: processed_video_urls is EMPTY -- the sequential pipeline "
              "renders no annotated videos (no Stage 4b). Any earlier sample "
              "showing those links was illustrative, not real.")
    return 0


if __name__ == "__main__":
    src = sys.argv[1]
    dst = sys.argv[2] if len(sys.argv) > 2 else "SAMPLE_global_train_report.json"
    sys.exit(main(src, dst))
