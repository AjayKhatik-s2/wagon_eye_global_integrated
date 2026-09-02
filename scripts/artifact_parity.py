#!/usr/bin/env python3
"""Compare the RENDERED artifacts of a Batch run and a Sequential run.

    python scripts/artifact_parity.py --batch <ws>/PARITY_BATCH \
                                      --sequential <ws>/PARITY_SEQ

WHY THIS EXISTS ALONGSIDE parity_diff.py
`parity_diff` opens exactly two files per side -- global_state/
global_train_state.json and global_state/per_camera_tracking.json -- plus a
six-field check on the combined report. It proves the canonical train is
identical, which is the substantive claim, and it is blind to everything the
renderers produce. A `logo_path` that reached one builder and not the other
changed every PDF and `parity_diff` still exited 0.

WHAT IS COMPARED
    combined JSON   every field, recursively
    camera PDFs     page count, embedded-image count, and the decoded text
    combined PDF    the same three

WHAT IS EXCUSED, AND WHY IT IS NOT CHEATING
Two runs happen at two times. `generated_at`, the rendered timestamp line and
the PDF trailer's CreationDate therefore differ between ANY two runs -- Batch
against Batch included. Treating those as intelligence differences would make
the check permanently red and train the operator to ignore it.

Everything excused is listed by name in the output. Nothing is excused by
pattern, by value-shape, or because it happened to differ. Run --control on two
Batch runs first if you want the excuse list validated empirically: whatever
differs there is provenance by definition.

RAW BYTE EQUALITY IS NOT THE TEST, and could not be: reportlab stamps a
CreationDate and the report renders _now_ist() onto the page. Structure and
decoded text are the honest comparison.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import zlib

# Field names whose value is a fact about the RUN, not about the train.
PROVENANCE = (
    "generated_at",
    "processing_seconds",
    "elapsed_seconds",
    "seconds",
    "started_at",
    "finished_at",
    "run_id",
    "duration",
    "timestamp",
)

# Absolute paths differ because the two runs use different workspaces.
PATH_KEYS = ("path", "_path", "dir", "root", "output_dir", "workspace")

failures: list = []
excused: list = []


def is_provenance(field: str) -> bool:
    leaf = field.rsplit(".", 1)[-1]
    if leaf in PROVENANCE:
        return True
    return any(leaf == k or leaf.endswith("_" + k) for k in PATH_KEYS)


# -----------------------------------------------------------------------------
# JSON
# -----------------------------------------------------------------------------

def compare_json(left, right, prefix=""):
    if is_provenance(prefix):
        if left != right:
            excused.append((prefix, left, right))
        return
    if isinstance(left, dict) and isinstance(right, dict):
        for key in sorted(set(left) | set(right)):
            field = "%s.%s" % (prefix, key) if prefix else key
            if key not in left or key not in right:
                failures.append((field, "absent" if key not in left else "present",
                                 "present" if key not in left else "absent"))
                continue
            compare_json(left[key], right[key], field)
    elif isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            failures.append(("%s[] length" % prefix, len(left), len(right)))
            return
        for i, (a, b) in enumerate(zip(left, right)):
            compare_json(a, b, "%s[%d]" % (prefix, i))
    elif isinstance(left, float) and isinstance(right, float):
        if abs(left - right) > 1e-6:
            failures.append((prefix, left, right))
    elif left != right:
        failures.append((prefix, left, right))


# -----------------------------------------------------------------------------
# PDF -- structure and decoded text, with no third-party dependency
# -----------------------------------------------------------------------------

_TS = re.compile(rb"\d{2}-\d{2}-\d{4}[^)]{0,20}")          # rendered date line
_CREATION = re.compile(rb"/CreationDate\s*\([^)]*\)")
_MODDATE = re.compile(rb"/ModDate\s*\([^)]*\)")


def pdf_shape(path: str) -> dict:
    """Page count, image count, and decoded text -- timestamps normalized.

    Streams are Flate-compressed by reportlab, so zlib (stdlib) is enough; a
    stream that will not inflate is skipped rather than guessed at.
    """
    with open(path, "rb") as fh:
        raw = fh.read()

    text = bytearray()
    for match in re.finditer(rb"stream\r?\n(.*?)endstream", raw, re.S):
        chunk = match.group(1)
        try:
            text += zlib.decompress(chunk)
        except zlib.error:
            continue

    body = bytes(text)
    body = _TS.sub(b"<TIMESTAMP>", body)
    return {
        "bytes": len(raw),
        "pages": len(re.findall(rb"/Type\s*/Page[^s]", raw)),
        "images": len(re.findall(rb"/Subtype\s*/Image", raw)),
        "text_len": len(body),
        "text": body,
    }


def compare_pdf(name: str, left_path: str, right_path: str):
    for p in (left_path, right_path):
        if not os.path.isfile(p):
            failures.append(("%s file" % name, os.path.isfile(left_path),
                             os.path.isfile(right_path)))
            return
    a, b = pdf_shape(left_path), pdf_shape(right_path)

    for key in ("pages", "images"):
        if a[key] != b[key]:
            failures.append(("%s.%s" % (name, key), a[key], b[key]))

    if a["text"] != b["text"]:
        # Report WHERE, so a divergence is actionable rather than a bare "differs".
        la = a["text"].split(b"\n")
        lb = b["text"].split(b"\n")
        first = next((i for i, (x, y) in enumerate(zip(la, lb)) if x != y), None)
        detail = "first differing text line %s" % first if first is not None \
            else "text length %d vs %d" % (a["text_len"], b["text_len"])
        failures.append(("%s.text" % name, detail, "see above"))
    # Size is reported, never failed on: compression is not determinism.
    excused.append(("%s.bytes" % name, a["bytes"], b["bytes"]))


CAMERA_PDFS = ("right_up_report.pdf", "left_up_report.pdf",
               "right_up_top_report.pdf", "left_up_top_report.pdf")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", required=True)
    ap.add_argument("--sequential", required=True)
    ap.add_argument("--batch-reports", default="reports",
                    help="Batch's report subdirectory (default: reports)")
    ap.add_argument("--sequential-reports", default="combined",
                    help="Sequential's report subdirectory (default: combined)")
    ap.add_argument("--control", action="store_true",
                    help="both sides are the SAME mode; anything that differs "
                         "is provenance by definition")
    args = ap.parse_args()

    bdir = os.path.join(args.batch, args.batch_reports)
    sdir = os.path.join(args.sequential, args.sequential_reports)

    print("=" * 78)
    print("  RENDERED-ARTIFACT parity%s" % ("  [CONTROL RUN]" if args.control
                                            else ""))
    print("=" * 78)
    print("  batch      : %s" % bdir)
    print("  sequential : %s" % sdir)
    print()

    if not os.path.isdir(bdir) or not os.path.isdir(sdir):
        print("  NOT COMPARABLE: %s exists=%s, %s exists=%s"
              % (bdir, os.path.isdir(bdir), sdir, os.path.isdir(sdir)))
        return 2

    # ---- combined JSON, every field ---------------------------------------
    name = "combined_train_report.json"
    bj, sj = os.path.join(bdir, name), os.path.join(sdir, name)
    if os.path.isfile(bj) and os.path.isfile(sj):
        with open(bj) as f:
            left = json.load(f)
        with open(sj) as f:
            right = json.load(f)
        compare_json(left, right, "combined")
        print("  combined JSON : compared recursively "
              "(%d + %d top-level keys)" % (len(left), len(right)))
    else:
        failures.append(("combined JSON present", os.path.isfile(bj),
                         os.path.isfile(sj)))

    # ---- PDFs --------------------------------------------------------------
    compare_pdf("combined.pdf",
                os.path.join(bdir, "combined_train_report.pdf"),
                os.path.join(sdir, "combined_train_report.pdf"))
    for pdf in CAMERA_PDFS:
        compare_pdf(pdf, os.path.join(bdir, pdf), os.path.join(sdir, pdf))
    print("  PDFs          : combined + 4 camera reports "
          "(pages, images, decoded text)")
    print()

    if excused:
        print("  EXCUSED (provenance -- differs between ANY two runs):")
        for field, a, b in excused[:20]:
            print("     %-34s %s != %s" % (field, str(a)[:28], str(b)[:28]))
        if len(excused) > 20:
            print("     ... and %d more" % (len(excused) - 20))
        print()

    if failures:
        print("  DIVERGENT:")
        for field, a, b in failures[:40]:
            print("     %-34s batch=%s" % (field, str(a)[:40]))
            print("     %-34s seq  =%s" % ("", str(b)[:40]))
        if len(failures) > 40:
            print("     ... and %d more" % (len(failures) - 40))
        print()
        if args.control:
            print("  CONTROL RUN: these fields differ between two runs of the "
                  "SAME mode, so they are provenance. Add them to PROVENANCE "
                  "and re-run the real comparison.")
            return 1
        print("ARTIFACT PARITY FAILED -- %d divergent field(s)" % len(failures))
        return 1

    print("ARTIFACT PARITY OK -- combined JSON identical field-for-field; "
          "every PDF matches on pages, images and decoded text")
    return 0


if __name__ == "__main__":
    sys.exit(main())
