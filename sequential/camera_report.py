"""The single-camera report: what THIS camera saw, and nothing more.

Written the moment a camera finishes, so one available camera is immediately
useful even when the other three do not exist yet.

Two things this report must never do:

* claim canonical wagon ids. Segments are `<CAMERA>_SEG_n` and every page says
  so, because the canonical global roster is created once, later, by Global
  Assembly. A reader who mistakes a local segment for a global wagon would
  draw the wrong conclusion about the train.
* aggregate features into per-wagon verdicts. A camera does not know which
  wagon a frame belongs to, so the report shows observation COUNTS and the
  strongest observation per feature -- evidence, not meaning.

`reporting/camera_reports.py` is deliberately NOT reused: it is Batch's
post-global Stage 5a and requires a GlobalTrainState full of GW ids.
`reporting/_brand.py` IS reused, so both reports look like the same product.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

from core import constants as C

from sequential import evidence as ev

TITLE = "Single-Camera Evidence Report"
NOT_CANONICAL = (
    "Camera-local view only. Segment numbers below are THIS camera's own "
    "segments between the gaps it detected; they are NOT canonical global "
    "wagon numbers. The canonical global wagon roster is produced once by the "
    "Global Assembly stage, after all required cameras are sealed."
)


def _observation_summary(camera_evidence: ev.CameraEvidence) -> Dict[str, Any]:
    """Counts and the strongest raw observation per feature -- no verdicts."""
    summary: Dict[str, Any] = {}
    for feature in ("door", "damage", "load"):
        observations = camera_evidence.observations_for(feature)
        if not observations:
            continue
        strongest = max(observations, key=lambda o: o.confidence)
        classes: Dict[str, int] = {}
        for observation in observations:
            classes[observation.raw_class] = classes.get(
                observation.raw_class, 0) + 1
        summary[feature] = {
            "observation_count": len(observations),
            "frames_with_observations": len({o.frame_idx for o in observations}),
            "raw_class_counts": dict(sorted(classes.items())),
            "strongest": {
                "raw_class": strongest.raw_class,
                "confidence": round(strongest.confidence, 4),
                "frame_idx": strongest.frame_idx,
            },
            "note": "raw observations; per-wagon aggregation happens in "
                    "Global Assembly",
        }
    return summary


def build_document(camera_evidence: ev.CameraEvidence, *, batch_key: str,
                   ) -> Dict[str, Any]:
    """The camera-local JSON report."""
    timing = camera_evidence.timing
    document = {
        "schema": "wagon_eye.camera_report.v1",
        "report_type": "single_camera",
        "canonical": False,
        "disclaimer": NOT_CANONICAL,
        "batch_key": batch_key,
        "camera_id": camera_evidence.camera_id,
        "status": camera_evidence.status,
        "timing": {
            "fps": timing.fps,
            "total_frames": timing.total_frames,
            "decoded_frames": timing.decoded_frames,
            "duration_seconds": timing.duration_seconds,
            "wagon_region_start_frame": timing.wagon_region_start_frame,
            "wagon_region_end_frame": timing.wagon_region_end_frame,
            "wagon_region_frames": timing.wagon_region_frames,
        },
        "gap_authority": (
            "canonical gap authority (RIGHT_UP)"
            if camera_evidence.camera_id == C.MASTER_CAMERA
            else "support camera: corroborating evidence only"),
        "local_gaps": {
            "count": camera_evidence.unique_gap_count,
            "gaps": [
                {"local_gap_id": gap.local_gap_id,
                 "confirmation_frame": gap.confirmation_frame,
                 "normalized_position": gap.normalized_position,
                 "max_confidence": gap.max_confidence}
                for gap in camera_evidence.gaps
            ],
        },
        "local_segments": {
            "count": len(camera_evidence.segments),
            "note": "segments between consecutive local gaps; NOT canonical "
                    "wagons",
            "segments": camera_evidence.segments,
        },
        "observations": _observation_summary(camera_evidence),
        "feature_config": camera_evidence.feature_config,
        "provenance": camera_evidence.provenance,
        "diagnostics": camera_evidence.diagnostics,
    }
    ev.assert_no_canonical_ids(
        document, where="camera report for %s" % camera_evidence.camera_id)
    return document


def _build_pdf(document: Dict[str, Any], output_pdf: str,
               verbose: bool) -> Optional[str]:
    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.units import inch
        from reportlab.platypus import (
            Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle)
    except Exception as exc:                                # pragma: no cover
        if verbose:
            print("[SEQ/%s] reportlab unavailable, PDF skipped: %s"
                  % (document["camera_id"], exc))
        return None

    from reporting import _brand

    styles = _brand.build_styles()
    story: List[Any] = []

    story.append(Paragraph("<b>%s</b>" % TITLE, styles["ReportTitle"]))
    story.append(Paragraph("Camera: <b>%s</b>" % document["camera_id"],
                           styles["ReportSubtitle"]))
    story.append(Spacer(1, 0.12 * inch))
    story.append(Paragraph("<i>%s</i>" % document["disclaimer"],
                           styles["ReportSubtitle"]))
    story.append(Spacer(1, 0.2 * inch))

    timing = document["timing"]
    overview = [
        ["Status", document["status"]],
        ["Gap authority", document["gap_authority"]],
        ["FPS", "%.3f" % timing["fps"]],
        ["Frames decoded", str(timing["decoded_frames"])],
        ["Duration (s)", "%.1f" % timing["duration_seconds"]],
        ["Wagon region (frames)",
         "%d .. %d  (%d frames)" % (timing["wagon_region_start_frame"],
                                    timing["wagon_region_end_frame"],
                                    timing["wagon_region_frames"])],
        ["Local gaps detected", str(document["local_gaps"]["count"])],
        ["Local segments", str(document["local_segments"]["count"])],
    ]
    table = Table([["Field", "Value"]] + overview,
                  colWidths=[2.4 * inch, 4.6 * inch])
    table.setStyle(TableStyle([
        ("GRID", (0, 0), (-1, -1), 0.5, colors.black),
        ("BACKGROUND", (0, 0), (-1, 0), _brand.HEADER_GRAY),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    story.append(table)
    story.append(Spacer(1, 0.25 * inch))

    observations = document.get("observations") or {}
    story.append(Paragraph("<b>Feature observations (raw)</b>",
                           styles["ReportSubtitle"]))
    story.append(Spacer(1, 0.08 * inch))
    if observations:
        rows = [["Feature", "Observations", "Frames", "Strongest class",
                 "Confidence"]]
        for feature in sorted(observations):
            info = observations[feature]
            rows.append([
                feature.upper(), str(info["observation_count"]),
                str(info["frames_with_observations"]),
                str(info["strongest"]["raw_class"]),
                "%.3f" % info["strongest"]["confidence"],
            ])
        feature_table = Table(rows, colWidths=[1.3 * inch, 1.3 * inch,
                                               1.1 * inch, 2.2 * inch,
                                               1.1 * inch])
        feature_table.setStyle(TableStyle([
            ("GRID", (0, 0), (-1, -1), 0.5, colors.black),
            ("BACKGROUND", (0, 0), (-1, 0), _brand.HEADER_GRAY),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("ALIGN", (1, 1), (-1, -1), "CENTER"),
        ]))
        story.append(feature_table)
        story.append(Spacer(1, 0.08 * inch))
        story.append(Paragraph(
            "<i>Per-wagon Door / Damage / Load verdicts are intentionally "
            "absent: assigning an observation to a wagon requires the "
            "canonical global timeline, which this camera does not have.</i>",
            styles["ReportSubtitle"]))
    else:
        story.append(Paragraph("No feature observations for this camera.",
                               styles["ReportSubtitle"]))

    story.append(Spacer(1, 0.25 * inch))
    segments = document["local_segments"]["segments"]
    story.append(Paragraph("<b>Camera-local segments (NOT canonical wagons)</b>",
                           styles["ReportSubtitle"]))
    story.append(Spacer(1, 0.08 * inch))
    if segments:
        rows = [["Segment", "Start frame", "End frame", "Start pos", "End pos"]]
        for segment in segments:
            rows.append([
                segment["segment_id"], str(segment["start_frame"]),
                str(segment["end_frame"]),
                "%.1f" % float(segment["start_normalized"]),
                "%.1f" % float(segment["end_normalized"]),
            ])
        segment_table = Table(rows, colWidths=[2.2 * inch, 1.2 * inch,
                                               1.2 * inch, 1.2 * inch,
                                               1.2 * inch])
        segment_table.setStyle(TableStyle([
            ("GRID", (0, 0), (-1, -1), 0.5, colors.black),
            ("BACKGROUND", (0, 0), (-1, 0), _brand.HEADER_GRAY),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("ALIGN", (1, 1), (-1, -1), "CENTER"),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
        ]))
        story.append(segment_table)
    else:
        story.append(Paragraph(
            "No segment could be formed: at least two confirmed local gaps "
            "are required.", styles["ReportSubtitle"]))

    os.makedirs(os.path.dirname(output_pdf), exist_ok=True)
    SimpleDocTemplate(output_pdf, pagesize=A4,
                      leftMargin=0.5 * inch, rightMargin=0.5 * inch,
                      topMargin=0.5 * inch, bottomMargin=0.5 * inch).build(story)
    return output_pdf


def build(*, workspace: str, evidence: ev.CameraEvidence, batch_key: str,
          verbose: bool = True) -> Dict[str, Optional[str]]:
    """Write `<CAMERA>_report.json` and `<CAMERA>_report.pdf`."""
    directory = ev.camera_report_dir(workspace, evidence.camera_id)
    os.makedirs(directory, exist_ok=True)

    document = build_document(evidence, batch_key=batch_key)
    json_path = os.path.join(directory, "%s_report.json" % evidence.camera_id)
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(document, handle, indent=2, default=str)

    pdf_path = None
    try:
        pdf_path = _build_pdf(
            document, os.path.join(directory, "%s_report.pdf"
                                   % evidence.camera_id), verbose)
    except Exception as exc:                                # pragma: no cover
        print("[SEQ/%s] camera PDF failed: %s" % (evidence.camera_id, exc))

    if verbose:
        print("[SEQ/%s] camera report JSON: %s" % (evidence.camera_id, json_path))
        if pdf_path:
            print("[SEQ/%s] camera report PDF : %s"
                  % (evidence.camera_id, pdf_path))

    return {"json_path": json_path, "pdf_path": pdf_path}
