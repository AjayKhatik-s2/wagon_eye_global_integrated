"""Canonical constants shared across the wagon_eye_v4 pipeline."""

from __future__ import annotations

import os


# -----------------------------------------------------------------------------
# Environment overrides.
#
# Every operational value below (bucket, endpoint, region, recipient list) is
# declared as `_env("WAGONEYE_<NAME>", <default>)`, so a deployment retargets the
# pipeline with an env file and NO source edit.  Before this layer existed the
# buckets were literals pointing at a previous AWS account, which meant a
# correctly-credentialed box still uploaded into a bucket it could not see.
# -----------------------------------------------------------------------------

def _env(name: str, default: str) -> str:
    """Read a WAGONEYE_* override, falling back to the built-in default."""
    val = os.getenv(name)
    return val if val else default


def _env_list(name: str, default: list) -> list:
    """Comma/semicolon separated env override for a list-valued constant."""
    raw = os.getenv(name)
    if not raw:
        return default
    return [x.strip() for x in raw.replace(";", ",").split(",") if x.strip()]


# -----------------------------------------------------------------------------
# Cameras
# -----------------------------------------------------------------------------

CAMERA_RIGHT_UP     = "RIGHT_UP"
CAMERA_LEFT_UP      = "LEFT_UP"
CAMERA_RIGHT_UP_TOP = "RIGHT_UP_TOP"
CAMERA_LEFT_UP_TOP  = "LEFT_UP_TOP"

ALL_CAMERAS = (
    CAMERA_RIGHT_UP, CAMERA_LEFT_UP,
    CAMERA_RIGHT_UP_TOP, CAMERA_LEFT_UP_TOP,
)
SIDE_CAMERAS = (CAMERA_RIGHT_UP, CAMERA_LEFT_UP)
TOP_CAMERAS  = (CAMERA_RIGHT_UP_TOP, CAMERA_LEFT_UP_TOP)
MASTER_CAMERA = CAMERA_RIGHT_UP

# Canonical lowercase cache folder name per camera
CAMERA_FOLDER = {
    CAMERA_RIGHT_UP:     "right_up",
    CAMERA_LEFT_UP:      "left_up",
    CAMERA_RIGHT_UP_TOP: "right_up_top",
    CAMERA_LEFT_UP_TOP:  "left_up_top",
}

# Reverse lookup
CAMERA_FROM_FOLDER = {v: k for k, v in CAMERA_FOLDER.items()}

# Filename spellings accepted when scanning an input folder for the 4 videos.
# The CCTV exporter writes the two TOP cameras as RIGHT_TOP / LEFT_TOP, so both
# that spelling and the canonical one must resolve to the same camera.
# scan_local_video_dir matches the LONGEST alias first, so RIGHT_UP_TOP is
# never captured by the shorter RIGHT_UP.
CAMERA_FILENAME_ALIASES = {
    CAMERA_RIGHT_UP:     (CAMERA_RIGHT_UP,),
    CAMERA_LEFT_UP:      (CAMERA_LEFT_UP,),
    CAMERA_RIGHT_UP_TOP: (CAMERA_RIGHT_UP_TOP, "RIGHT_TOP"),
    CAMERA_LEFT_UP_TOP:  (CAMERA_LEFT_UP_TOP,  "LEFT_TOP"),
}


# -----------------------------------------------------------------------------
# Status sentinel values
# -----------------------------------------------------------------------------

NO_DATA       = "NO_DATA"
STATUS_OK     = "OK"
STATUS_FAILED = "FAILED"
STATUS_NO_FRAMES = "NO_FRAMES"
STATUS_DISABLED  = "DISABLED_BY_USER"   # feature-JSON status when a user toggled it OFF

# Display string carried in UnifiedWagonState fields owned by a disabled
# feature, and rendered verbatim in reports in place of NO_DATA / OK.
DISABLED_DISPLAY = "DISABLED BY USER"

# Batch outcome statuses persisted in processed_batches.json
BATCH_COMPLETED          = "completed"
BATCH_COMPLETED_PARTIAL  = "completed_partial"
BATCH_REPORT_FAILED      = "report_failed"
BATCH_FAILED_NO_GLOBAL   = "failed_no_global_state"
BATCH_FAILED             = "failed"


# -----------------------------------------------------------------------------
# Classification labels (matching wagon_count.global_train_state.SegmentClass)
# -----------------------------------------------------------------------------

CLASS_ENGINE    = "ENGINE"
CLASS_WAGON     = "WAGON"
CLASS_BRAKE_VAN = "BRAKE_VAN"
CLASS_UNKNOWN   = "UNKNOWN"


# -----------------------------------------------------------------------------
# Reconstruction model filenames (in models/reconstruction/)
# -----------------------------------------------------------------------------

# Short names (preferred); the wagon_count package now also accepts these.
# DECLARATIVE ONLY -- documentation of the Stage-1 contract for operators.
# The counting engine (wagon_count/run_global_count.py) resolves these names
# itself under --recon-models-dir; nothing in wagon_eye_v4 reads the constants
# below to load a model.  They are kept in sync with the engine so the expected
# filenames are discoverable from one place.
MODEL_RIGHT_UP_GAP        = "right_up_wagon_gap.pt"
MODEL_LEFT_UP_GAP         = "left_up_wagon_gap.pt"
MODEL_TOP_GAP             = "top_gap.pt"
MODEL_SIDE_CLASSIFICATION = "side_classification.pt"

# OPTIONAL.  Classifies the two TOP cameras so engine / brake-van regions stay
# out of wagon synchronization.  Never a counting authority -- RIGHT_UP alone
# decides the count -- so a missing file degrades capability, never the count.
MODEL_TOP_CLASSIFICATION  = "top_classification.pt"


# -----------------------------------------------------------------------------
# Feature model filenames (in models/features/)
# -----------------------------------------------------------------------------

MODEL_DOOR_STATE        = "door_state.pt"
MODEL_LOADED            = "loaded.pt"
MODEL_DAMAGE            = "damage.pt"
MODEL_WAGON_ID_COUNTING = "wagon_id_counting.pt"


# -----------------------------------------------------------------------------
# Door state vocabulary (from the trained door_state.pt model)
# -----------------------------------------------------------------------------

DOOR_CLOSED  = "CLOSED"
DOOR_OPEN    = "OPEN"
DOOR_PARTIAL = "PARTIAL"
DOOR_DAMAGED = "DAMAGED"

# Map raw YOLO class names to canonical door states. Anything not in the
# dict is preserved verbatim (uppercased) so downstream can still see it.
DOOR_LABEL_TO_STATE = {
    "open":               DOOR_OPEN,
    "open_door":          DOOR_OPEN,
    "closed":             DOOR_CLOSED,
    "closed_door":        DOOR_CLOSED,
    "closed_with_wire":   DOOR_PARTIAL,
    "partial_closed":     DOOR_PARTIAL,
    "partially_closed":   DOOR_PARTIAL,
    "partial":            DOOR_PARTIAL,
    "damage":             DOOR_DAMAGED,
}


# -----------------------------------------------------------------------------
# Load status vocabulary
# -----------------------------------------------------------------------------

LOAD_LOADED = "LOADED"
LOAD_EMPTY  = "EMPTY"

LOAD_LABEL_TO_STATE = {
    "loaded": LOAD_LOADED,
    "load":   LOAD_LOADED,
    "full":   LOAD_LOADED,
    "empty":  LOAD_EMPTY,
    "unload": LOAD_EMPTY,
}


# -----------------------------------------------------------------------------
# Damage vocabulary (top cameras)
# -----------------------------------------------------------------------------

DAMAGE_PRESENT = "DAMAGE"
DAMAGE_OK      = "OK"

# Top-camera damage classes we COUNT as damage.  Outer-wall damage is
# skipped on top cameras because it is the side cameras' responsibility.
DAMAGE_CLASSES_TOP = {"floor_damage", "inner_wall_damage"}
DAMAGE_CLASSES_NEGATIVE = {"no_damage"}

# PROBABLE (not confirmed) top damage.  The dashboard reports these separately
# and must NOT count them as confirmed damage.  The double underscore in
# `floor__probable_damage` is the trained model's real class name, not a typo --
# matching it exactly is what makes probable damage reportable instead of
# silently unmapped.
DAMAGE_CLASSES_PROBABLE = {"floor__probable_damage", "floor_probable_damage",
                           "floor_dmg_probable"}


def is_probable_damage(class_name: str) -> bool:
    """True for a PROBABLE (not confirmed) top-damage class."""
    return str(class_name or "").strip().lower() in DAMAGE_CLASSES_PROBABLE


# -----------------------------------------------------------------------------
# S3 + email -- preserved from the legacy master_runner constants so the
# new package can drop in without operational changes.
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# Per-camera S3 layout.
#
# CANONICAL per-camera S3 folder (== the site's own `camera_id`).  SINGLE source
# of truth: the dashboard feed resolves through here, so a rig rename is a
# one-line edit.  Note the site names its TOP rigs RIGHT_TOP / LEFT_TOP even
# though the canonical camera ids are RIGHT_UP_TOP / LEFT_UP_TOP.
# -----------------------------------------------------------------------------

CAMERA_S3_FOLDER = {
    CAMERA_RIGHT_UP:     "camera_CCTV_HZBN_DHN_2_RIGHT_UP",
    CAMERA_LEFT_UP:      "camera_CCTV_HZBN_DHN_1_LEFT_UP",
    CAMERA_RIGHT_UP_TOP: "camera_CCTV_HZBN_DHN_5_RIGHT_TOP",
    CAMERA_LEFT_UP_TOP:  "camera_CCTV_HZBN_DHN_6_LEFT_TOP",
}

#: Reverse lookup: S3 folder -> camera id.
S3_FOLDER_TO_CAMERA = {v: k for k, v in CAMERA_S3_FOLDER.items()}


# -----------------------------------------------------------------------------
# S3 + email.
#
# THE DEPLOYED ACCOUNT IS `biputri-*`.  Every default names a bucket in THAT
# account, so a box with no env file still reads and writes inside the account it
# is credentialed for.  The previous account's `biro-*` names are deliberately
# absent: leaving one as a default is how a misconfigured box silently uploads
# into an account nobody is watching, which is exactly what happened before.
# -----------------------------------------------------------------------------

S3_REGION = _env("WAGONEYE_S3_REGION", "ap-south-1")
S3_OUTPUT_BUCKET = _env("WAGONEYE_S3_OUTPUT_BUCKET", "biputri-wagoneye-report")
S3_TRAIN_BATCH_PREFIX = _env("WAGONEYE_S3_TRAIN_BATCH_PREFIX", "train_batch")
S3_STATE_KEY = _env("WAGONEYE_S3_STATE_KEY",
                    "master_runner/processed_batches.json")

# Where the per-camera inspection JSON is uploaded.  The dashboard ingest API is
# handed an `s3://` URI into this bucket and FETCHES the document from there.
#
# It DERIVES from S3_OUTPUT_BUCKET rather than naming a bucket of its own: a
# standalone default is the one value an operator has no reason to think about,
# so it survives an account migration untouched and the feed keeps addressing the
# old account.  Derived, it moves with the account by construction.  The RECEIVER
# must have read access to whichever bucket this resolves to.
S3_ARTIFACT_BUCKET = _env("WAGONEYE_ARTIFACT_BUCKET", S3_OUTPUT_BUCKET)

# -----------------------------------------------------------------------------
# Dashboard ingest endpoints.  Both hosts serve the SAME path -- the `version`
# field inside the document, not the URL, selects the dashboard tab.
# -----------------------------------------------------------------------------

INGEST_API_URL_PROD = _env(
    "WAGONEYE_INGEST_API_URL_PROD",
    "https://ms-pnr-location-notification-api.suvidhaen.com/"
    "cctv-receiver/inspections/ingest",
)

INGEST_API_URL_UAT = _env(
    "WAGONEYE_INGEST_API_URL_UAT",
    "https://cctv-wagon-uat-api.suvidhaen.com/inspections/ingest",
)

# The `version` carried in each per-camera document; the dashboard chooses which
# tab renders the report from this ("v1" -> the V1 tab).
INSPECTION_VERSION = _env("WAGONEYE_INSPECTION_VERSION", "v1")

UPLOAD_API_URL = _env("WAGONEYE_UPLOAD_API_URL",
                      "https://reports-api.suvidhaen.com/api/upload-pdf")
EMAIL_API_URL = _env(
    "WAGONEYE_EMAIL_API_URL",
    "https://ms-pnr-location-notification-api.suvidhaen.com/"
    "notification_microservice/send-email",
)
PRODUCT_NAME = _env("WAGONEYE_PRODUCT_NAME", "CCTV-WagonEye-CombinedReports")

EMAIL_RECEIVER = _env_list("WAGONEYE_EMAIL_RECEIVER",
                           ["atul.nitt.cse@gmail.com"])
EMAIL_RECEIVER_CC = _env_list("WAGONEYE_EMAIL_RECEIVER_CC", [
    "Shivank.kumar.s2.s2@gmail.com",
    "rithish.sheru.s2@gmail.com",
    "omarbil01.s2@gmail.com",
    "kumarankitiitps2@gmail.com",
    "ajaykhatik6367s2@gmail.com",
    "priyankagp51.s2@gmail.com",
    "aman.freelancer.s2@gmail.com",
    "rajchaudhary01.official@gmail.com",
    "shyambabugupt.s2@gmail.com",
    "contact@suvidhaen.com",
])


# -----------------------------------------------------------------------------
# Misc tunables
# -----------------------------------------------------------------------------

# Confidence floors (inference)
CONF_DOOR    = 0.40
CONF_DAMAGE  = 0.55
CONF_OCR_BOX = 0.40

# JPEG quality for materializer
JPEG_QUALITY = 90

# OCR
WAGON_NUMBER_LENGTH = 11
