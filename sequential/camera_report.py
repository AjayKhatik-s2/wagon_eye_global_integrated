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
import sys
from typing import Any, Dict, List, Optional


from sequential import evidence as ev

#: Repo root, for resolving the shared report logo the way Batch does.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

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


def _inspection_items(camera_id: str, local_state,
                      local_result: Dict[str, Any]) -> List[Dict[str, Any]]:
    """The SAME per-wagon item list Batch's camera PDF renders from.

    `camera_reports._build_camera_items` is Batch's own function -- it produces
    `{sr, gw_id, classification, classification_conf, visible, detections,
    anomalies, primary_confidence}` per wagon, where `detections` carries the
    Door / Load / Damage labels, states, confidences and snapshot paths.

    Serializing that list is what makes the camera JSON the complete inspection
    record rather than a parallel invention: the JSON and the PDF then describe
    the same wagons with the same verdicts, because they come from one call.

    Batch has no camera-report JSON of its own, so there is no Batch JSON
    contract this could match -- this is Sequential-defined, built from Batch's
    data.
    """
    if local_state is None or not local_result:
        return []
    unified = local_result.get("unified") or {}
    if not unified:
        return []
    paths = local_result.get("paths") or {}
    try:
        from reporting import camera_reports
        items = camera_reports._build_camera_items(
            camera_id=camera_id, state=local_state, unified=unified,
            evidence_root=paths.get("evidence_root"),
            wagon_states_root=paths.get("states_root"),
            cache_root=paths.get("cache_root"))
    except Exception as exc:                                # pragma: no cover
        print("[SEQ/P1/%s] item build FAILED: %s" % (camera_id, exc),
              file=sys.stderr)
        return []

    out: List[Dict[str, Any]] = []
    for item in items:
        out.append({
            "sr": item.get("sr"),
            # This camera's OWN wagon id. The key keeps Batch's name so the
            # shape matches, but the VALUE is local (`LEFT_UP_W1`).
            "local_wagon_id": item.get("gw_id"),
            "classification": item.get("classification"),
            "classification_confidence": item.get("classification_conf"),
            "visible": item.get("visible"),
            "primary_confidence": item.get("primary_confidence"),
            # [(label, state, confidence, snapshot_path), ...] -- the Door /
            # Load / Damage results, with their evidence frame references.
            "detections": [
                {"label": d[0], "state": d[1], "confidence": d[2],
                 "snapshot": d[3]}
                for d in (item.get("detections") or []) if len(d) >= 4
            ],
            "anomalies": [{"severity": a[0], "text": a[1]}
                          for a in (item.get("anomalies") or [])
                          if len(a) >= 2],
        })
    return out


def build_document(camera_evidence: ev.CameraEvidence, *, batch_key: str,
                   local_state=None,
                   local_result: Optional[Dict[str, Any]] = None,
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
        # NO global-authority claim. `C.MASTER_CAMERA` is a static constant
        # (RIGHT_UP), but the real master is whichever camera has the most
        # confirmed unique gaps -- selected in Global Assembly, and on real
        # footage it is often NOT RIGHT_UP. Labelling RIGHT_UP the "canonical
        # gap authority" here was therefore a global claim inside a
        # camera-local document, wrong on any train whose master is another
        # camera, and it demoted the other three to "corroborating evidence
        # only" when in Phase 1 all four cameras are equal and independent.
        #
        # In Phase 1 no camera is the authority. That is decided once, later,
        # from all four sealed evidences.
        "gap_authority": (
            "none in this phase: every camera reports independently, and the "
            "master camera is selected in Global Assembly from all four "
            "sealed evidences (most confirmed unique gaps)"),
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
        # The complete per-wagon inspection record, from Batch's own item
        # builder -- Door / Load / Damage verdicts, confidences, evidence
        # snapshots and anomalies, per CAMERA-LOCAL wagon.
        "inspection": {
            "wagons": _inspection_items(camera_evidence.camera_id, local_state,
                                        local_result or {}),
            "note": "camera-local wagons between this camera's own confirmed "
                    "gaps; verdicts computed on those windows. The canonical "
                    "train and its GW_n roster are built in Global Assembly.",
        },
        "phase1_timings": dict((local_result or {}).get("timings") or {}),
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


def _batch_rendered_pdf(*, workspace: str, camera_id: str, batch_key: str,
                        local_state, local_result: Dict[str, Any],
                        output_pdf: str, verbose: bool) -> Optional[str]:
    """Render the camera report with BATCH'S OWN renderer, camera-locally.

    `reporting.camera_reports.build_camera_report` is the report Batch produces:
    the summary KPI page, the detection summary, the anomaly summary, the
    evidence pages and the per-wagon pages. It takes `state` and `unified` and
    addresses everything by `wagon.global_id`, so handing it a camera-local
    state and that camera's own fused states yields the SAME report -- the same
    sections, layout, labels and evidence presentation -- naming this camera's
    own wagons instead of canonical ones.

    Nothing about the renderer is forked. This is a call, not a copy.

    Returns None when the local state or fusion is unavailable, and the caller
    then falls back to the camera-local counting report: a camera with fewer
    than two confirmed gaps bounds no wagon, so there is nothing for a
    wagon-oriented report to describe, and inventing a page for it would be
    worse than not rendering one.
    """
    if local_state is None or not local_result:
        return None
    unified = local_result.get("unified") or {}
    if not unified:
        return None
    paths = local_result.get("paths") or {}
    try:
        from reporting import camera_reports
        # Resolved the way Batch resolves it, so the Phase-1 report carries the
        # same logo as every other report in the system.
        logo = os.path.join(_REPO_ROOT, "reporting", "assets", "Logo.jpeg")
        return camera_reports.build_camera_report(
            camera_id=camera_id,
            state=local_state,
            unified=unified,
            evidence_root=paths.get("evidence_root"),
            wagon_states_root=paths.get("states_root"),
            cache_root=paths.get("cache_root"),
            per_camera_tracking_path=paths.get("tracking_path"),
            output_pdf=output_pdf,
            batch_key=batch_key,
            logo_path=logo if os.path.isfile(logo) else None,
            verbose=verbose,
        )
    except Exception as exc:                                # pragma: no cover
        print("[SEQ/P1/%s] Batch camera renderer FAILED: %s"
              % (camera_id, exc), file=sys.stderr)
        return None


def build(*, workspace: str, evidence: ev.CameraEvidence, batch_key: str,
          local_state=None, local_result: Optional[Dict[str, Any]] = None,
          verbose: bool = True) -> Dict[str, Optional[str]]:
    """Write `<CAMERA>_report.json` and `<CAMERA>_report.pdf`.

    The PDF is Batch's own camera-report renderer whenever this camera produced
    local wagons and fused states; otherwise it is the camera-local counting
    report, which is all a camera with no bounded wagon can honestly show.
    """
    directory = ev.camera_report_dir(workspace, evidence.camera_id)
    os.makedirs(directory, exist_ok=True)
    local_result = local_result or {}

    document = build_document(evidence, batch_key=batch_key,
                              local_state=local_state,
                              local_result=local_result)
    json_path = os.path.join(directory, "%s_report.json" % evidence.camera_id)
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(document, handle, indent=2, default=str)

    output_pdf = os.path.join(directory, "%s_report.pdf" % evidence.camera_id)
    pdf_path = _batch_rendered_pdf(
        workspace=workspace, camera_id=evidence.camera_id, batch_key=batch_key,
        local_state=local_state, local_result=local_result,
        output_pdf=output_pdf, verbose=verbose)
    if pdf_path:
        return {"json_path": json_path, "pdf_path": pdf_path}

    try:
        pdf_path = _build_pdf(document, output_pdf, verbose)
    except Exception as exc:                                # pragma: no cover
        print("[SEQ/%s] camera PDF failed: %s" % (evidence.camera_id, exc))

    if verbose:
        print("[SEQ/%s] camera report JSON: %s" % (evidence.camera_id, json_path))
        if pdf_path:
            print("[SEQ/%s] camera report PDF : %s"
                  % (evidence.camera_id, pdf_path))

    return {"json_path": json_path, "pdf_path": pdf_path}
