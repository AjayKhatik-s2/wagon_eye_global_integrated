"""Four positional wagon frames per camera angle -- start / mid1 / mid2 / end.

    wagon_cache/<GW_n>/<camera>/frame_NNNNNN.jpg      Stage 2 wrote these
              |
              |  materialize()   copies four per wagon per camera
              v
    evidence/<GW_n>/wagon_frames/<angle>/w<N>_frame_<IDX>.jpg
              |
              |  Stage 6's existing upload_tree(evidence_root, ...)
              v
    s3://<bucket>/train_batch/<batch_key>/evidence/<GW_n>/wagon_frames/<angle>/...

Why they go through `evidence/` rather than being uploaded on their own: Stage 6
already mirrors that whole tree, so these objects land with no new upload code
and no second bucket, and the URL is the SAME base the door snapshots use. The
frames cannot simply be linked where they already are -- `wagon_cache/` is
deliberately never uploaded (it is the bulk of a train's several GB and the
historical runner reclaims it after each batch).

Which frames
------------
Four samples at 12.5 / 37.5 / 62.5 / 87.5% through the wagon's own frame range,
per camera -- the same quartiles `reporting._evidence_lookup` uses for the PDF's
2x2 wagon overview, so the JSON and the PDF show the same pictures. The extreme
edges are avoided on purpose: those frames are the ones most likely to be
clipped by a neighbouring wagon or by the gap itself.

These are TEMPORAL positions. `start` is near the wagon's leading edge, `end`
near its trailing edge.

Camera angle keys
-----------------
The keys are the DASHBOARD's angle vocabulary, not this package's folder names:
`right_top` / `left_top`, not `right_up_top` / `left_up_top`. That is what
`CCTV-watch-report-BE/app/helpers/wagon_assembly.py: camera_angle` produces and
what the site's own rig folders are named (`..._5_RIGHT_TOP`), so a reader can
line an angle up with a camera without a translation table.
"""

from __future__ import annotations

import os
import shutil
from typing import Any, Dict, List, Optional, Tuple

from . import constants as C

#: Labels, in order, for the four sample points.
POSITIONS: Tuple[str, ...] = ("start", "mid1", "mid2", "end")

#: Where in the wagon each is sampled.  Not 0/33/66/100 -- see the module note.
FRACTIONS: Tuple[float, ...] = (0.125, 0.375, 0.625, 0.875)

#: Camera id -> the angle key published in the JSON.  The dashboard's own
#: vocabulary; `C.CAMERA_S3_FOLDER` is the authority for which physical rig each
#: camera is, and these agree with it (`..._2_RIGHT_UP` -> `right_up`,
#: `..._5_RIGHT_TOP` -> `right_top`).
ANGLE_BY_CAMERA: Dict[str, str] = {
    C.CAMERA_LEFT_UP:      "left_up",
    C.CAMERA_RIGHT_UP:     "right_up",
    C.CAMERA_RIGHT_UP_TOP: "right_top",
    C.CAMERA_LEFT_UP_TOP:  "left_top",
}

#: Reverse lookup, so a consumer can get back from an angle to a camera id.
CAMERA_BY_ANGLE: Dict[str, str] = {v: k for k, v in ANGLE_BY_CAMERA.items()}

#: Sub-tree inside a wagon's evidence folder.
DIR_NAME = "wagon_frames"


def local_frame_range(wagon, camera_id: str, local_fps: float,
                      local_total_frames: int,
                      time_offset: float = 0.0) -> Tuple[int, int]:
    """Inclusive `[start, end]` local frame indices for one wagon on one camera.

    THIS MUST MATCH THE MATERIALIZER. Stage 2 wrote the cache using
    `materializer.wagon_cache_builder._wagon_local_range`; a different window
    names frames that were never cached, and every position would then be
    dropped as missing. The same two sources in the same order:

      1. `wagon.local_range(camera_id)` -- the window a counting engine ALREADY
         aligned for this camera, reversed timelines included, which no single
         clock delta can express. Preferred whenever present.
      2. the master-time projection `(start_time - offset) * local_fps`, exact
         for an engine that emits a shared clock with per-camera offsets.

    `(0, -1)` when the wagon does not overlap this camera's footage at all, so
    the caller publishes nothing rather than naming a frame of another wagon.
    `tests/test_wagon_frame_positions.py` asserts agreement with the materializer.
    """
    if local_total_frames <= 0:
        return (0, -1)

    explicit = None
    if camera_id is not None and hasattr(wagon, "local_range"):
        explicit = wagon.local_range(camera_id)
    if explicit is not None:
        sf, ef = explicit
        if ef < 0 or sf > local_total_frames - 1:
            return (0, -1)
        sf = max(0, min(local_total_frames - 1, sf))
        ef = max(0, min(local_total_frames - 1, ef))
        return (sf, max(sf, ef))

    if local_fps <= 0:
        return (0, -1)
    sf = int(round((float(wagon.start_time) - time_offset) * local_fps))
    ef = int(round((float(wagon.end_time) - time_offset) * local_fps)) - 1
    if ef < 0 or sf > local_total_frames - 1:
        return (0, -1)
    sf = max(0, min(local_total_frames - 1, sf))
    ef = max(0, min(local_total_frames - 1, ef))
    if ef < sf:
        ef = sf
    return (sf, ef)


def frame_filename(wagon_index: int, frame_idx: int) -> str:
    """`w39_frame_002957.jpg` -- the wagon's ordinal and its camera-local index.

    Carrying both means a reader can match a filename to a wagon and to a frame
    in that camera's clip without consulting anything else.
    """
    return "w%d_frame_%06d.jpg" % (int(wagon_index), int(frame_idx))


def evidence_rel_path(gw_id: str, angle: str, filename: str) -> str:
    """Path relative to `evidence/`, which is exactly what Stage 6 mirrors."""
    return "%s/%s/%s/%s" % (gw_id, DIR_NAME, angle, filename)


def cache_frame_path(cache_root: str, gw_id: str, camera_id: str,
                     frame_idx: int) -> str:
    """Where Stage 2 wrote that frame."""
    folder = C.CAMERA_FOLDER.get(camera_id, str(camera_id).lower())
    return os.path.join(cache_root, gw_id, folder,
                        "frame_%06d.jpg" % int(frame_idx))


def plan_for_camera(
    *,
    cache_root: Optional[str],
    wagon,
    camera_id: str,
    local_fps: float,
    local_total_frames: int,
    time_offset: float = 0.0,
) -> List[Dict[str, Any]]:
    """The positions that ACTUALLY have a cached frame, for one (wagon, camera).

    Each entry: `{position, frame_idx, source, rel_path}`. A position whose
    frame is absent from the cache is dropped rather than carried as a hole, so
    no caller has to guard for a missing file and no URL is ever published for a
    picture that does not exist.

    Duplicate indices collapse, earliest position winning: a wagon spanning only
    a few frames can sample the same frame twice, and publishing one picture
    under two positions would misrepresent it.
    """
    gw_id = getattr(wagon, "global_id", "")
    angle = ANGLE_BY_CAMERA.get(camera_id)
    if not cache_root or not gw_id or not angle:
        return []
    sf, ef = local_frame_range(wagon, camera_id, local_fps,
                               local_total_frames, time_offset)
    if ef <= sf:
        return []
    span = ef - sf
    wagon_index = int(getattr(wagon, "wagon_index", 0) or 0)
    out: List[Dict[str, Any]] = []
    seen: set = set()
    for position, frac in zip(POSITIONS, FRACTIONS):
        idx = max(sf, min(ef, sf + int(round(frac * span))))
        if idx in seen:
            continue
        source = cache_frame_path(cache_root, gw_id, camera_id, idx)
        if not os.path.isfile(source):
            continue
        seen.add(idx)
        out.append({
            "position":  position,
            "frame_idx": idx,
            "source":    source,
            "rel_path":  evidence_rel_path(
                gw_id, angle, frame_filename(wagon_index, idx)),
        })
    return out


def materialize(
    *,
    state,
    cache_root: Optional[str],
    evidence_root: Optional[str],
    per_camera_meta: Dict[str, Dict[str, Any]],
    camera_offsets: Optional[Dict[str, Any]] = None,
    cameras=C.ALL_CAMERAS,
    verbose: bool = True,
) -> Dict[str, Dict[str, List[Dict[str, Any]]]]:
    """Copy the four frames per wagon per camera into the evidence tree.

    Returns `{gw_id: {angle: [{position, rel_path, frame_idx}, ...]}}` -- only
    what was actually written, so the report publishes URLs for objects Stage 6
    will genuinely upload.

    A copy, not a move: the cache is still needed by the PDF's own quartile
    lookup, which runs after this.
    """
    out: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
    if not cache_root or not evidence_root or not state:
        return out
    copied = skipped = 0
    for wagon in getattr(state, "wagons", ()) or ():
        gw_id = wagon.global_id
        per_angle: Dict[str, List[Dict[str, Any]]] = {}
        for camera_id in cameras:
            meta = per_camera_meta.get(camera_id) or {}
            offset = 0.0
            if camera_offsets:
                entry = camera_offsets.get(camera_id)
                offset = float(getattr(entry, "delta_seconds", 0.0) or 0.0) \
                    if not isinstance(entry, dict) \
                    else float(entry.get("delta_seconds") or 0.0)
            plan = plan_for_camera(
                cache_root=cache_root, wagon=wagon, camera_id=camera_id,
                local_fps=float(meta.get("fps") or 0.0),
                local_total_frames=int(meta.get("total_frames") or 0),
                time_offset=offset)
            written: List[Dict[str, Any]] = []
            for item in plan:
                dest = os.path.join(evidence_root, *item["rel_path"].split("/"))
                try:
                    os.makedirs(os.path.dirname(dest), exist_ok=True)
                    if not os.path.isfile(dest):
                        shutil.copyfile(item["source"], dest)
                    copied += 1
                except OSError:
                    skipped += 1
                    continue
                written.append({"position": item["position"],
                                "rel_path": item["rel_path"],
                                "frame_idx": item["frame_idx"]})
            if written:
                per_angle[ANGLE_BY_CAMERA[camera_id]] = written
        if per_angle:
            out[gw_id] = per_angle
    if verbose:
        print("[STAGE5] wagon frames: %d copied into evidence/ across %d wagon(s)"
              "%s" % (copied, len(out),
                      "  (%d failed)" % skipped if skipped else ""))
    return out


def published(
    manifest: Dict[str, Dict[str, List[Dict[str, Any]]]], gw_id: str,
    evidence_url_base: Optional[str],
) -> Dict[str, List[Dict[str, str]]]:
    """One wagon's manifest as the wire shape: `{angle: [{position, s3_url}]}`.

    The URL is constructed, not observed -- Stage 5b publishes before Stage 6
    uploads. `evidence_url_base` is the caller's already-assembled
    `https://<bucket>.s3.<region>.amazonaws.com/<prefix>/<batch_key>/evidence`,
    the same base the door snapshots use. No base means the run is not
    uploading, so nothing is published.
    """
    if not evidence_url_base or not manifest:
        return {}
    per_angle = manifest.get(gw_id) or {}
    base = evidence_url_base.rstrip("/")
    out: Dict[str, List[Dict[str, str]]] = {}
    for angle in ANGLE_BY_CAMERA.values():          # deterministic key order
        frames = per_angle.get(angle) or []
        if frames:
            out[angle] = [{"position": f["position"],
                           "s3_url": "%s/%s" % (base, f["rel_path"])}
                          for f in frames]
    return out
