"""Runtime configuration for the S3-facing modes (`--auto`, `--historical`).

Single source of truth for the handful of paths and runtime knobs the S3
discovery layer needs.  Every value is env-override-capable and defaults to
exactly the path/behaviour this package already used, so a deployment that sets
no environment variables behaves identically to a hand-driven `--local-only`
run.

Design rules:
    * No module hardcodes an absolute path -- import from here.
    * PROJECT_ROOT is discovered dynamically from this file's location, so the
      project works no matter where it is cloned on the host.
    * Nothing here loads a model, reads a frame, or touches GlobalTrainState.
      It is pure configuration (mirrors core/feature_config.py's discipline).

Deliberately NOT declared here
------------------------------
The MODEL directories.  `orchestrator.master_runner` already resolves those
through its own `_dir_default(env_var, fallback)` helper
(`WAGONEYE_RECON_MODELS_DIR` / `WAGONEYE_FEAT_MODELS_DIR`), and it stays the
authority.  Restating them here would create a second place that answers "where
are the weights", which is exactly the failure this module exists to avoid.

Environment variables (all optional):
    WAGONEYE_WORKSPACE_ROOT             output root (default <root>/batch_outputs)
    WAGONEYE_LOG_DIR                    log directory (default <root>/logs)
    WAGONEYE_LOG_LEVEL                  root log level (default INFO)
    WAGONEYE_OPERATIONAL_DAY_START_HOUR_IST  discovery anchor hour (default 5)
    WAGONEYE_PROCESSOR_START_UTC        one-time backlog-skip anchor (ISO 8601)
"""

from __future__ import annotations

import os
from datetime import (datetime as _datetime, timedelta as _timedelta,
                      timezone as _timezone)

# -----------------------------------------------------------------------------
# Project root -- discovered from this file, never hardcoded.
# core/config.py  ->  <PROJECT_ROOT>/core/config.py, so PROJECT_ROOT is two
# levels up.  Works regardless of where the repo is cloned.
# -----------------------------------------------------------------------------

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _env_path(var: str, default: str) -> str:
    """Return an absolute path from an env var, falling back to `default`.

    A relative env value is resolved against PROJECT_ROOT so the project still
    works no matter what the process working directory is.
    """
    raw = os.getenv(var)
    if not raw:
        return default
    return raw if os.path.isabs(raw) else os.path.join(PROJECT_ROOT, raw)


def _env_str(var: str, default: str) -> str:
    val = os.getenv(var)
    return val if val else default


def _env_float(var: str, default: float) -> float:
    raw = os.getenv(var)
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


# -----------------------------------------------------------------------------
# Filesystem paths (all overridable; all default to this package's own layout)
# -----------------------------------------------------------------------------

WORKSPACE_ROOT = _env_path("WAGONEYE_WORKSPACE_ROOT",
                           os.path.join(PROJECT_ROOT, "batch_outputs"))
LOG_DIR = _env_path("WAGONEYE_LOG_DIR", os.path.join(PROJECT_ROOT, "logs"))
LOG_LEVEL = _env_str("WAGONEYE_LOG_LEVEL", "INFO")


# -----------------------------------------------------------------------------
# Per-batch output subfolder names.  These MIRROR the literals
# `orchestrator.master_runner.process_batch` already builds its batch tree from;
# they are named here so the historical runner can address `downloads/` and
# `wagon_cache/` without restating a string the orchestrator owns.
# -----------------------------------------------------------------------------

DIR_DOWNLOADS        = "downloads"
DIR_GLOBAL_STATE     = "global_state"
DIR_WAGON_CACHE      = "wagon_cache"
DIR_WAGON_STATES     = "wagon_states"
DIR_EVIDENCE         = "evidence"
DIR_PROCESSED_VIDEOS = "processed_videos"
DIR_REPORTS          = "reports"
DIR_ARCHIVE          = "archive"


# -----------------------------------------------------------------------------
# OPERATIONAL-DAY DISCOVERY ANCHOR  (the production rule, adopted verbatim)
#
# Everything discovery finds is bounded by the START OF THE CURRENT OPERATIONAL
# DAY: 05:00 IST, rolling back a day when the clock is before 05:00.
#
# Why an anchor beats a sliding "last N minutes" window:
#   * A restart at any hour still sees the whole operational day, so stopping
#     overnight and starting at 05:30 loses nothing -- a sliding 10-minute
#     window would skip every train uploaded while the service was down.
#   * It is inherently bounded to ONE day, so it can never reach back into
#     months of archive and queue thousands of batches.
#   * It matches the 05:00 boundary the dashboard already uses for its date
#     folders (delivery.dashboard_ingest.date_folder), so a train and its report
#     always agree about which day they belong to.
#
# WAGONEYE_PROCESSOR_START_UTC (ISO 8601) raises the anchor for a one-time
# backlog skip -- never below the 05:00 anchor, and it self-expires at the next
# day's anchor.
# -----------------------------------------------------------------------------

IST = _timezone(_timedelta(hours=5, minutes=30))

OPERATIONAL_DAY_START_HOUR_IST = int(
    _env_float("WAGONEYE_OPERATIONAL_DAY_START_HOUR_IST", 5))


def operational_day_start_utc(now=None):
    """UTC datetime of the current operational day's start (05:00 IST default)."""
    now_utc = now or _datetime.now(_timezone.utc)
    if getattr(now_utc, "tzinfo", None) is None:
        now_utc = now_utc.replace(tzinfo=_timezone.utc)
    now_ist = now_utc.astimezone(IST)
    start_ist = now_ist.replace(hour=OPERATIONAL_DAY_START_HOUR_IST,
                                minute=0, second=0, microsecond=0)
    if now_ist.hour < OPERATIONAL_DAY_START_HOUR_IST:
        start_ist = start_ist - _timedelta(days=1)
    return start_ist.astimezone(_timezone.utc)


def discovery_cutoff_utc(now=None):
    """The effective "ignore anything older than this" instant for discovery.

    The operational-day anchor, raised by WAGONEYE_PROCESSOR_START_UTC when that
    is set and later.  Returns a tz-aware UTC datetime.
    """
    anchor = operational_day_start_utc(now)
    raw = os.getenv("WAGONEYE_PROCESSOR_START_UTC")
    if raw:
        try:
            ov = _datetime.fromisoformat(raw.strip())
            if ov.tzinfo is None:
                ov = ov.replace(tzinfo=_timezone.utc)
            return max(anchor, ov.astimezone(_timezone.utc))
        except ValueError:
            pass
    return anchor
