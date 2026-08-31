"""The camera evidence contract: what one camera knows, and nothing more.

`CAMERA = EVIDENCE, GLOBAL = MEANING.`

A camera records what it saw -- gap detections, door/damage/load observations,
its own timing and provenance -- and nothing about the train as a whole. In
particular a camera NEVER records a canonical global wagon id: `GW_1..GW_N`
belong exclusively to Global Assembly, which is the single place the global
interpretation is created. `assert_no_canonical_ids()` enforces that, and the
architecture tests call it on real evidence.

Layout under the batch workspace:

    camera_evidence/<CAMERA>/evidence.json     observations + timing + provenance
    camera_evidence/<CAMERA>/sealed.json       the seal (written LAST)
    camera_reports/<CAMERA>/<CAMERA>_report.json
    camera_reports/<CAMERA>/<CAMERA>_report.pdf

`sealed.json` is written only after every other artifact is on disk, so its
presence means "this camera is complete and reusable". It carries the
fingerprints that decide reuse: change the video, the weights, the relevant
config or the schema, and the seal no longer matches, so the camera is
reprocessed instead of silently reusing stale evidence.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Sequence

from core import constants as C

SCHEMA_VERSION = "wagon_eye.camera_evidence.v2"
SEAL_SCHEMA_VERSION = "wagon_eye.camera_seal.v2"

EVIDENCE_DIRNAME = "camera_evidence"
CAMERA_REPORTS_DIRNAME = "camera_reports"
COMBINED_DIRNAME = "combined"

EVIDENCE_FILENAME = "evidence.json"
SEAL_FILENAME = "sealed.json"

STATUS_SEALED = "SEALED"
STATUS_FAILED = "FAILED"
STATUS_NO_REGION = "NO_WAGON_REGION"

# A camera-local segment label. Deliberately NOT "GW_n": a camera does not know
# the canonical roster, and a reader of a single-camera report must never
# mistake a local segment for a global wagon.
SEGMENT_ID_FORMAT = "%s_SEG_%d"

# Any canonical id leaking into camera evidence is a design violation.
_CANONICAL_ID = re.compile(r"\bGW_\d+\b")


class EvidenceError(RuntimeError):
    pass


# -----------------------------------------------------------------------------
# Fingerprints
# -----------------------------------------------------------------------------

def file_fingerprint(path: Optional[str]) -> Dict[str, Any]:
    """Cheap, deterministic identity for a large binary (video or weights).

    Name + size + mtime rather than a content hash: hashing multi-hundred-MB
    videos and weights on every run would cost more than the staleness check is
    worth. It detects a replaced or re-exported file, which is what resume
    safety needs.
    """
    if not path or not os.path.isfile(path):
        return {"path": path, "present": False}
    stat = os.stat(path)
    digest = hashlib.sha256(
        ("%s|%d|%d" % (os.path.basename(path), stat.st_size,
                       stat.st_mtime_ns)).encode("utf-8")).hexdigest()
    return {
        "path": os.path.abspath(path),
        "name": os.path.basename(path),
        "present": True,
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "fingerprint": digest[:32],
    }


def digest_of(payload: Any) -> str:
    """Stable digest of any JSON-serialisable structure."""
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"),
                      default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:32]


def config_fingerprint(*, features: Sequence[str], door_stride: int,
                       damage_stride: int, load_stride: int,
                       extra: Optional[Dict[str, Any]] = None) -> str:
    """Identity of the configuration that produced a camera's evidence.

    Only values that change the OBSERVATIONS belong here: the feature set, the
    sampling strides and the schema. A change to any of them invalidates reuse.
    """
    payload = {
        "schema": SCHEMA_VERSION,
        "features": sorted(features),
        "door_stride": int(door_stride),
        "damage_stride": int(damage_stride),
        "load_stride": int(load_stride),
    }
    if extra:
        payload["extra"] = extra
    return digest_of(payload)


# -----------------------------------------------------------------------------
# Observations -- plain records, no global meaning
# -----------------------------------------------------------------------------

@dataclass
class GapObservation:
    """One tracked, CONFIRMED unique gap on this camera's own timeline.

    Frames are this camera's ORIGINAL video indices. `normalized_position` is
    the engine's 0-1000 position within this camera's confirmed wagon region --
    a camera-local coordinate, not a global one.
    """
    local_gap_id: str
    confirmation_frame: int
    first_frame: int
    last_frame: int
    normalized_position: float
    max_confidence: float
    average_confidence: float = 0.0
    frame_count: int = 0


@dataclass
class FeatureObservation:
    """One raw detection, on one frame, from one feature detector.

    Raw on purpose: aggregation into a per-wagon verdict is Global Assembly's
    job, because only it knows which wagon a frame belongs to. Storing verdicts
    here would bake a camera-local guess into the evidence.
    """
    feature: str                    # door | damage | load
    frame_idx: int                  # ORIGINAL video frame index
    timestamp: float
    state: str                      # canonical feature state / class name
    confidence: float
    bbox: Optional[List[float]] = None
    raw_class: str = ""
    score: float = 0.0              # snapshot quality score, when computed
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CameraTiming:
    fps: float = 0.0
    total_frames: int = 0
    decoded_frames: int = 0
    wagon_region_start_frame: int = 0
    wagon_region_end_frame: int = 0
    wagon_region_frames: int = 0
    duration_seconds: float = 0.0


@dataclass
class CameraEvidence:
    """Everything one camera contributes, and nothing it cannot know."""
    camera_id: str
    schema_version: str = SCHEMA_VERSION
    status: str = STATUS_SEALED
    timing: CameraTiming = field(default_factory=CameraTiming)

    gaps: List[GapObservation] = field(default_factory=list)
    observations: List[FeatureObservation] = field(default_factory=list)

    # Per-frame classification the trimming stage produced, kept because Global
    # Assembly needs it for the WAGON-active interval and for wagon
    # classification without re-running the classifier.
    classification_timeline: List[Dict[str, Any]] = field(default_factory=list)
    # Camera-local segments between consecutive local gaps. NOT canonical.
    segments: List[Dict[str, Any]] = field(default_factory=list)

    provenance: Dict[str, Any] = field(default_factory=dict)
    feature_config: Dict[str, Any] = field(default_factory=dict)
    diagnostics: Dict[str, Any] = field(default_factory=dict)
    snapshots: Dict[str, str] = field(default_factory=dict)

    def to_document(self) -> Dict[str, Any]:
        return {
            "camera_id": self.camera_id,
            "schema_version": self.schema_version,
            "status": self.status,
            "timing": asdict(self.timing),
            "gaps": [asdict(g) for g in self.gaps],
            "observations": [asdict(o) for o in self.observations],
            "classification_timeline": self.classification_timeline,
            "segments": self.segments,
            "provenance": self.provenance,
            "feature_config": self.feature_config,
            "diagnostics": self.diagnostics,
            "snapshots": self.snapshots,
        }

    # ---- convenience -------------------------------------------------
    def observations_for(self, feature: str) -> List[FeatureObservation]:
        return [o for o in self.observations if o.feature == feature]

    @property
    def unique_gap_count(self) -> int:
        return len(self.gaps)


def evidence_from_document(document: Dict[str, Any]) -> CameraEvidence:
    timing = CameraTiming(**{k: v for k, v in
                            (document.get("timing") or {}).items()
                            if k in CameraTiming.__dataclass_fields__})
    return CameraEvidence(
        camera_id=document["camera_id"],
        schema_version=document.get("schema_version", ""),
        status=document.get("status", ""),
        timing=timing,
        gaps=[GapObservation(**g) for g in (document.get("gaps") or [])],
        observations=[FeatureObservation(**o)
                      for o in (document.get("observations") or [])],
        classification_timeline=list(document.get("classification_timeline") or []),
        segments=list(document.get("segments") or []),
        provenance=dict(document.get("provenance") or {}),
        feature_config=dict(document.get("feature_config") or {}),
        diagnostics=dict(document.get("diagnostics") or {}),
        snapshots=dict(document.get("snapshots") or {}),
    )


# -----------------------------------------------------------------------------
# The canonical-id guard
# -----------------------------------------------------------------------------

def assert_no_canonical_ids(document: Any, *, where: str) -> None:
    """Raise if a canonical global wagon id appears in camera-local data.

    Camera evidence that already names `GW_3` has made a global decision, which
    is exactly the confusion this architecture exists to prevent.
    """
    text = json.dumps(document, default=str)
    found = _CANONICAL_ID.findall(text)
    if found:
        raise EvidenceError(
            "camera-local data in %s contains canonical global wagon id(s) %s. "
            "Canonical GW_n ids are created ONLY by Global Assembly; a camera "
            "may only emit local segment labels (%s)."
            % (where, sorted(set(found))[:5], SEGMENT_ID_FORMAT % ("RIGHT_UP", 1)))


# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------

def camera_evidence_dir(workspace: str, camera_id: str) -> str:
    return os.path.join(workspace, EVIDENCE_DIRNAME, camera_id)


def camera_report_dir(workspace: str, camera_id: str) -> str:
    return os.path.join(workspace, CAMERA_REPORTS_DIRNAME, camera_id)


def combined_dir(workspace: str) -> str:
    return os.path.join(workspace, COMBINED_DIRNAME)


def evidence_path(workspace: str, camera_id: str) -> str:
    return os.path.join(camera_evidence_dir(workspace, camera_id),
                        EVIDENCE_FILENAME)


def seal_path(workspace: str, camera_id: str) -> str:
    return os.path.join(camera_evidence_dir(workspace, camera_id), SEAL_FILENAME)


# -----------------------------------------------------------------------------
# Write / read
# -----------------------------------------------------------------------------

def _write_json(path: str, document: Any) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temporary = path + ".partial"
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(document, handle, indent=2, default=str)
    os.replace(temporary, path)          # atomic: no half-written evidence
    return path


def write_evidence(workspace: str, evidence: CameraEvidence) -> str:
    """Persist one camera's evidence, refusing canonical ids."""
    document = evidence.to_document()
    assert_no_canonical_ids(document, where="camera evidence for %s"
                            % evidence.camera_id)
    return _write_json(evidence_path(workspace, evidence.camera_id), document)


def load_evidence(workspace: str, camera_id: str) -> CameraEvidence:
    path = evidence_path(workspace, camera_id)
    if not os.path.isfile(path):
        raise EvidenceError("no evidence for %s at %s" % (camera_id, path))
    with open(path, "r", encoding="utf-8") as handle:
        return evidence_from_document(json.load(handle))


def write_seal(workspace: str, *, camera_id: str, status: str,
               timing: CameraTiming, video_fingerprint: Dict[str, Any],
               model_fingerprints: Dict[str, Any], config_digest: str,
               feature_config: Dict[str, Any], processing_seconds: float,
               report_paths: Dict[str, Optional[str]],
               unique_gap_count: int, observation_count: int,
               notes: Optional[List[str]] = None) -> str:
    """Write `sealed.json`. Called LAST, so its presence means completeness."""
    document = {
        "schema_version": SEAL_SCHEMA_VERSION,
        "evidence_schema_version": SCHEMA_VERSION,
        "camera_id": camera_id,
        "status": status,
        "sealed_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "processing_seconds": round(float(processing_seconds), 3),
        "frame_count": timing.decoded_frames,
        "fps": timing.fps,
        "timing": asdict(timing),
        "unique_gap_count": int(unique_gap_count),
        "observation_count": int(observation_count),
        "video_fingerprint": video_fingerprint,
        "model_fingerprints": model_fingerprints,
        "config_fingerprint": config_digest,
        "feature_config": feature_config,
        "evidence_dir": camera_evidence_dir(workspace, camera_id),
        "evidence_path": evidence_path(workspace, camera_id),
        "reports": report_paths,
        "notes": notes or [],
    }
    assert_no_canonical_ids(document, where="seal for %s" % camera_id)
    return _write_json(seal_path(workspace, camera_id), document)


def load_seal(workspace: str, camera_id: str) -> Optional[Dict[str, Any]]:
    path = seal_path(workspace, camera_id)
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None


# -----------------------------------------------------------------------------
# Resume
# -----------------------------------------------------------------------------

@dataclass
class ResumeDecision:
    reuse: bool
    reason: str
    seal: Optional[Dict[str, Any]] = None


def evaluate_resume(workspace: str, camera_id: str, *,
                    video_fingerprint: Dict[str, Any],
                    model_fingerprints: Dict[str, Any],
                    config_digest: str) -> ResumeDecision:
    """Decide whether a sealed camera may be reused instead of reprocessed.

    Reuse requires the seal to match on ALL of: schema version, status, video
    fingerprint, every model fingerprint and the configuration digest. Anything
    else is stale and is reprocessed -- and the reason is returned so the
    decision can be logged rather than being silent.
    """
    seal = load_seal(workspace, camera_id)
    if seal is None:
        return ResumeDecision(False, "no seal found")
    if seal.get("schema_version") != SEAL_SCHEMA_VERSION:
        return ResumeDecision(
            False, "seal schema %r != %r" % (seal.get("schema_version"),
                                             SEAL_SCHEMA_VERSION), seal)
    if seal.get("evidence_schema_version") != SCHEMA_VERSION:
        return ResumeDecision(
            False, "evidence schema %r != %r"
            % (seal.get("evidence_schema_version"), SCHEMA_VERSION), seal)
    if seal.get("status") != STATUS_SEALED:
        return ResumeDecision(False, "sealed status is %r"
                              % seal.get("status"), seal)
    if not os.path.isfile(evidence_path(workspace, camera_id)):
        return ResumeDecision(False, "seal present but evidence file missing", seal)

    sealed_video = (seal.get("video_fingerprint") or {}).get("fingerprint")
    if sealed_video != video_fingerprint.get("fingerprint"):
        return ResumeDecision(False, "input video changed", seal)

    sealed_models = seal.get("model_fingerprints") or {}
    for slot, fingerprint in model_fingerprints.items():
        wanted = (fingerprint or {}).get("fingerprint")
        if (sealed_models.get(slot) or {}).get("fingerprint") != wanted:
            return ResumeDecision(False, "model %r changed" % slot, seal)

    if seal.get("config_fingerprint") != config_digest:
        return ResumeDecision(False, "configuration changed", seal)

    return ResumeDecision(True, "sealed evidence matches", seal)


def sealed_cameras(workspace: str,
                   cameras: Sequence[str] = C.ALL_CAMERAS) -> List[str]:
    """Cameras with a complete seal, in the given deterministic order."""
    out: List[str] = []
    for camera_id in cameras:
        seal = load_seal(workspace, camera_id)
        if seal and seal.get("status") == STATUS_SEALED:
            out.append(camera_id)
    return out
