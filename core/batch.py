"""TrainBatch + CameraVideo records.

Copied (and trimmed) from the legacy `train_batch_manager.py` so that
`wagon_eye_v4/` stays self-contained.  Polling logic + S3 state code
that actually talks to S3 lives in `orchestrator/master_runner.py`.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

from . import constants as C

#: The site's zone.  Filename timestamps are IST wall-clock (see `age_seconds`).
#: Defined locally rather than imported from `core.config` so this module stays
#: importable without the config layer, as it has always been.
_IST = timezone(timedelta(hours=5, minutes=30))


# -----------------------------------------------------------------------------
# CameraVideo
# -----------------------------------------------------------------------------

@dataclass
class CameraVideo:
    """One downloaded / locatable video file for one camera in a batch."""
    camera_id: str
    bucket: str                 # S3 bucket name OR sentinel '__local__'
    s3_key: str                 # for local mode this is the local filesystem path
    filename: str
    s3_url: str
    train_timestamp: str        # YYYYMMDD_HHMMSS
    file_size: int = 0
    last_modified: Optional[datetime] = None
    etag: Optional[str] = None  # S3 ETag (source version); None in local mode


# -----------------------------------------------------------------------------
# TrainBatch
# -----------------------------------------------------------------------------

@dataclass
class TrainBatch:
    batch_key: str
    train_timestamp: str
    videos: Dict[str, CameraVideo] = field(default_factory=dict)

    def present_cameras(self) -> List[str]:
        return [cam for cam in C.ALL_CAMERAS if cam in self.videos]

    def missing_cameras(self) -> List[str]:
        return [cam for cam in C.ALL_CAMERAS if cam not in self.videos]

    def is_complete(self) -> bool:
        return not self.missing_cameras()

    def age_seconds(self) -> float:
        """Seconds since the batch's train_timestamp.

        The filename digits are **IST wall-clock**, not UTC: the upstream
        trimmer writes them that way and names each trimmed clip after its raw
        basename.  Labelling them UTC makes every batch look 5h30m in the
        FUTURE, so this returns about -19,500 s and
        `train_batch_manager.select_runnable_batch`'s
        `age_seconds() >= partial_wait` gate can never be satisfied -- a 2- or
        3-camera train is held back for ~5.4 hours instead of the intended 30
        minutes.  There was no consumer of this method before S3 discovery
        existed, so correcting the zone changes nothing that previously ran.
        """
        try:
            t = datetime.strptime(self.train_timestamp, "%Y%m%d_%H%M%S")
            t = t.replace(tzinfo=_IST)
        except ValueError:
            return 0.0
        return (datetime.now(timezone.utc) - t).total_seconds()


# -----------------------------------------------------------------------------
# Filename → train_timestamp parser
# -----------------------------------------------------------------------------

# Matches  ..._YYYYMMDD_HHMMSS...   (the convention used by the upstream
# trimmer service).
_TS_RE = re.compile(r"(\d{8}_\d{6})")


def parse_train_timestamp(filename: str) -> Optional[str]:
    m = _TS_RE.search(os.path.basename(filename))
    return m.group(1) if m else None


# -----------------------------------------------------------------------------
# Local batch helper
# -----------------------------------------------------------------------------

def build_local_batch(
    video_paths: Dict[str, str],
    batch_key: Optional[str] = None,
) -> TrainBatch:
    """Wrap a {camera_id -> local_path} mapping as a TrainBatch."""
    if not batch_key:
        batch_key = datetime.now().strftime("%Y%m%d_%H%M%S")
    videos: Dict[str, CameraVideo] = {}
    for cam, path in video_paths.items():
        videos[cam] = CameraVideo(
            camera_id=cam,
            bucket="__local__",
            s3_key=path,
            filename=os.path.basename(path),
            s3_url=f"file://{path}",
            train_timestamp=batch_key,
            file_size=os.path.getsize(path) if os.path.exists(path) else 0,
            last_modified=datetime.now(timezone.utc),
        )
    return TrainBatch(batch_key=batch_key,
                      train_timestamp=batch_key,
                      videos=videos)


# -----------------------------------------------------------------------------
# Scan a local folder for one video per camera (for --local-only)
# -----------------------------------------------------------------------------

def scan_local_video_dir(local_dir: str) -> Dict[str, str]:
    """Find one video per camera by filename substring (case-insensitive).

    Each camera accepts several spellings (`C.CAMERA_FILENAME_ALIASES`) because
    the CCTV exporter names the top cameras RIGHT_TOP / LEFT_TOP rather than
    RIGHT_UP_TOP / LEFT_UP_TOP.  Aliases are tried LONGEST FIRST, so
    RIGHT_UP_TOP is never captured by the shorter, ambiguous RIGHT_UP.
    """
    import glob

    candidates: List[str] = []
    for ext in ("*.mp4", "*.MP4", "*.avi", "*.AVI", "*.mov", "*.MOV"):
        candidates.extend(glob.glob(os.path.join(local_dir, "**", ext), recursive=True))
    candidates = sorted(set(candidates))

    # (alias, camera) pairs, longest alias first; ties keep ALL_CAMERAS order.
    pairs: List[Tuple[str, str]] = [
        (alias, cam)
        for cam in C.ALL_CAMERAS
        for alias in C.CAMERA_FILENAME_ALIASES.get(cam, (cam,))
    ]
    pairs.sort(key=lambda item: len(item[0]), reverse=True)

    found: Dict[str, str] = {}
    for alias, cam in pairs:
        if cam in found:
            continue
        alias_l = alias.lower()
        for path in candidates:
            if alias_l in os.path.basename(path).lower() and path not in found.values():
                found[cam] = path
                break
    return found
