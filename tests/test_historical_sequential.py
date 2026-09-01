"""Historical mode feeding the SEQUENTIAL pipeline.

Historical mode is an input-SELECTION layer: it decides which already-trimmed S3
clips belong to which train, stages them, and hands each train to the pipeline.
These tests pin the four things that layer can get wrong in ways no downstream
stage would notice:

  * the four cameras are grouped into the right trains despite a real ~2 min
    inter-camera skew (tolerance),
  * each camera's OWN source URL reaches the report, so each camera's dashboard
    document carries its own capture time rather than one shared batch time,
  * `--historical` runs SEQUENTIAL by default but honours an explicit choice,
  * a partial batch is reported PARTIAL rather than silently assembled.

`run_sequential` is stubbed throughout: this file is about orchestration, and
real inference here would make it a multi-hour test.
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone

import pytest

from core import constants as C
from core.batch import CameraVideo, TrainBatch
from orchestrator import historical_runner as HR

IST = timezone(timedelta(hours=5, minutes=30))

#: The measured skew at this site: RIGHT_UP is stamped ~121-127 s BEFORE the
#: other three cameras for the same train.
REAL_SKEW_SEC = 125


# -----------------------------------------------------------------------------
# Fake S3
# -----------------------------------------------------------------------------

def _objects(times=("07:15:00", "08:12:00"), skew=REAL_SKEW_SEC, cameras=None):
    cams = cameras or list(C.ALL_CAMERAS)
    objs = []
    for base in times:
        h, m, s = (int(x) for x in base.split(":"))
        t0 = datetime(2026, 7, 24, h, m, s, tzinfo=IST)
        for cam in cams:
            t = t0 - timedelta(seconds=skew) if cam == C.CAMERA_RIGHT_UP else t0
            folder = C.CAMERA_S3_FOLDER[cam]
            objs.append({
                "Key": "%s/CCTV_%s_%s_train.mp4"
                       % (folder, folder, t.strftime("%Y%m%d_%H%M%S")),
                "LastModified": t.astimezone(timezone.utc),
                "ETag": '"etag-%s"' % cam,
                "Size": 500_000_000,
            })
    return objs


class FakeS3:
    def __init__(self, objs):
        self.objs = objs
        self.downloads = []

    def list_objects_v2(self, **kw):
        pre = kw.get("Prefix", "")
        items = [o for o in self.objs if o["Key"].startswith(pre)]
        return {"Contents": items, "IsTruncated": False, "KeyCount": len(items)}

    def get_paginator(self, _op):
        outer = self

        class _P:
            def paginate(self, **kw):
                yield outer.list_objects_v2(**kw)

        return _P()

    def download_file(self, bucket, key, dest):
        self.downloads.append((bucket, key, dest))
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        with open(dest, "wb") as fh:
            fh.write(b"\0" * 1024)


def _window():
    return HR.resolve_window(date="2026-07-24", start_time="07:00",
                             end_time="10:00", timezone_name=None,
                             start_iso=None, end_iso=None)


# -----------------------------------------------------------------------------
# Stub sequential pipeline
# -----------------------------------------------------------------------------

class _Assembly:
    def __init__(self, ready=True, reason=""):
        self.ready = ready
        self.reason = reason
        self.report_pdf_path = "/tmp/combined.pdf" if ready else None
        self.report_json_path = "/tmp/combined.json" if ready else None


class _Outcome:
    def __init__(self, ready=True, sealed=None):
        self.assembly = _Assembly(ready)
        self.sealed_cameras = (list(C.ALL_CAMERAS) if sealed is None else sealed)
        self.cameras = []


@pytest.fixture
def stub_sequential(monkeypatch):
    """Replace `run_sequential` with a recorder; yields the captured kwargs."""
    calls = []

    def _fake(**kw):
        calls.append(kw)
        return _Outcome()

    from sequential import runner as seq_runner
    monkeypatch.setattr(seq_runner, "run_sequential", _fake)
    return calls


# -----------------------------------------------------------------------------
# Grouping / tolerance
# -----------------------------------------------------------------------------

def test_default_tolerance_splits_trains_at_this_site():
    """120 s is too tight HERE: it splits every train into 1 + 3 cameras.

    Pinned deliberately.  The default is not a bug in the clustering -- it is the
    live value and correct for a site whose cameras are stamped together.  The
    test exists so that anyone who "fixes" the default by widening it sees that
    the split behaviour was known and measured, and so the 300 s recommendation
    in the CLI help has something backing it.
    """
    res = HR.select_objects(s3_client=FakeS3(_objects()), window=_window(),
                            pad_minutes=15.0, tolerance_sec=120)
    assert len(res.batches) == 4, [b.batch_key for b in res.batches]
    assert not any(b.is_complete() for b in res.batches)
    assert sorted(len(b.present_cameras()) for b in res.batches) == [1, 1, 3, 3]


def test_widened_tolerance_recovers_whole_trains():
    res = HR.select_objects(s3_client=FakeS3(_objects()), window=_window(),
                            pad_minutes=15.0, tolerance_sec=300)
    assert len(res.batches) == 2
    assert all(b.is_complete() for b in res.batches)
    for b in res.batches:
        assert sorted(b.present_cameras()) == sorted(C.ALL_CAMERAS)


def test_top_cameras_are_not_dropped():
    """The site writes RIGHT_TOP/LEFT_TOP; the canonical ids are *_UP_TOP.

    Matching only the canonical id resolves both top clips to no camera, so every
    batch silently forms with two side cameras and the combined report never
    builds.  This is the regression that guards the alias/folder resolution.
    """
    res = HR.select_objects(s3_client=FakeS3(_objects(times=("07:15:00",))),
                            window=_window(), pad_minutes=15.0,
                            tolerance_sec=300)
    assert len(res.batches) == 1
    present = res.batches[0].present_cameras()
    assert C.CAMERA_RIGHT_UP_TOP in present
    assert C.CAMERA_LEFT_UP_TOP in present


# -----------------------------------------------------------------------------
# Staging
# -----------------------------------------------------------------------------

def test_stage_clips_downloads_each_camera_once(tmp_path):
    batch = TrainBatch(batch_key="20260724_071255",
                       train_timestamp="20260724_071255")
    for cam in C.ALL_CAMERAS:
        batch.videos[cam] = CameraVideo(
            camera_id=cam, bucket="b", s3_key="k/%s.mp4" % cam,
            filename="%s.mp4" % cam, s3_url="https://x/%s.mp4" % cam,
            train_timestamp="20260724_071255", file_size=1024)
    s3 = FakeS3([])
    paths = HR.stage_clips(batch=batch, s3_client=s3,
                           batch_root=str(tmp_path), verbose=False)
    assert sorted(paths) == sorted(C.ALL_CAMERAS)
    assert len(s3.downloads) == 4
    assert all(os.path.exists(p) for p in paths.values())


def test_stage_clips_reuses_a_correctly_sized_clip(tmp_path):
    """A re-run must not re-download tens of GB of identical video.

    Size is the guard, so a TRUNCATED file is still re-fetched.
    """
    batch = TrainBatch(batch_key="k", train_timestamp="20260724_071255")
    batch.videos[C.CAMERA_RIGHT_UP] = CameraVideo(
        camera_id=C.CAMERA_RIGHT_UP, bucket="b", s3_key="k/a.mp4",
        filename="a.mp4", s3_url="https://x/a.mp4",
        train_timestamp="20260724_071255", file_size=1024)

    s3 = FakeS3([])
    HR.stage_clips(batch=batch, s3_client=s3, batch_root=str(tmp_path),
                   verbose=False)
    assert len(s3.downloads) == 1

    # Second pass: correct size on disk -> no download.
    HR.stage_clips(batch=batch, s3_client=s3, batch_root=str(tmp_path),
                   verbose=False)
    assert len(s3.downloads) == 1, "cached clip was re-downloaded"

    # Truncate it -> downloaded again.
    dest = s3.downloads[0][2]
    with open(dest, "wb") as fh:
        fh.write(b"\0" * 10)
    HR.stage_clips(batch=batch, s3_client=s3, batch_root=str(tmp_path),
                   verbose=False)
    assert len(s3.downloads) == 2, "truncated clip was NOT re-downloaded"


# -----------------------------------------------------------------------------
# Per-camera timestamps -- the point of source_video_urls
# -----------------------------------------------------------------------------

def test_each_camera_keeps_its_own_source_url(tmp_path, stub_sequential):
    """RIGHT_UP's clip is stamped ~2 min before the others' for the same train.

    `dashboard_ingest` derives each camera's `upload_timestamp` from the raw video
    NAME it finds in `train_metadata.source_video_urls`.  If historical mode does
    not pass per-camera URLs, every camera falls back to a batch-key-derived name
    and all four documents get one identical timestamp -- which is the bug that
    filed global runs under the reprocess date.
    """
    s3 = FakeS3(_objects(times=("07:15:00",)))
    rc = HR.run(s3_client=s3, window=_window(), workspace_root=str(tmp_path),
                recon_models_dir="/m/r", feat_models_dir="/m/f",
                pad_minutes=15.0, tolerance_sec=300, dry_run=False,
                keep_inputs=True, deliver=False, send_email=False,
                mode="sequential", verbose=False)
    assert rc == 0
    assert len(stub_sequential) == 1
    urls = stub_sequential[0]["source_video_urls"]

    assert sorted(urls) == sorted(C.ALL_CAMERAS)
    stamps = {cam: os.path.basename(u).split("_")[-3:-1] for cam, u in urls.items()}
    right_up = "_".join(stamps[C.CAMERA_RIGHT_UP])
    others = {"_".join(v) for k, v in stamps.items() if k != C.CAMERA_RIGHT_UP}
    assert len(others) == 1, "the three synced cameras disagree: %s" % others
    assert right_up != others.pop(), \
        "RIGHT_UP's skew was flattened -- per-camera timestamps are lost"


def test_source_video_urls_reaches_run_sequential_signature():
    """The parameter must exist on the real function, not just the stub."""
    import inspect
    from sequential import runner as seq_runner
    from sequential import global_assembly
    assert "source_video_urls" in inspect.signature(
        seq_runner.run_sequential).parameters
    assert "source_video_urls" in inspect.signature(
        global_assembly.assemble).parameters


# -----------------------------------------------------------------------------
# Delivery flags
# -----------------------------------------------------------------------------

@pytest.mark.parametrize("deliver,send_email,exp_upload,exp_email", [
    (False, False, True,  True),
    (False, True,  True,  True),   # no delivery at all -> email suppressed too
    (True,  False, False, True),   # deliver, but --skip-email
    (True,  True,  False, False),  # deliver everything
])
def test_delivery_flags(tmp_path, stub_sequential, deliver, send_email,
                        exp_upload, exp_email):
    HR.run(s3_client=FakeS3(_objects(times=("07:15:00",))), window=_window(),
           workspace_root=str(tmp_path), recon_models_dir="/m/r",
           feat_models_dir="/m/f", pad_minutes=15.0, tolerance_sec=300,
           dry_run=False, keep_inputs=True, deliver=deliver,
           send_email=send_email, mode="sequential", verbose=False)
    kw = stub_sequential[0]
    assert kw["skip_upload"] is exp_upload
    assert kw["skip_email"] is exp_email


def test_dry_run_downloads_nothing_and_runs_no_inference(tmp_path,
                                                        stub_sequential):
    s3 = FakeS3(_objects())
    rc = HR.run(s3_client=s3, window=_window(), workspace_root=str(tmp_path),
                recon_models_dir="/m/r", feat_models_dir="/m/f",
                pad_minutes=15.0, tolerance_sec=300, dry_run=True,
                keep_inputs=True, deliver=True, send_email=True,
                mode="sequential", verbose=False)
    assert rc == 0
    assert s3.downloads == []
    assert stub_sequential == []


def test_no_matching_video_is_exit_2(tmp_path, stub_sequential):
    rc = HR.run(s3_client=FakeS3([]), window=_window(),
                workspace_root=str(tmp_path), recon_models_dir="/m/r",
                feat_models_dir="/m/f", pad_minutes=15.0, tolerance_sec=300,
                dry_run=False, keep_inputs=True, deliver=False,
                send_email=False, mode="sequential", verbose=False)
    assert rc == 2
    assert stub_sequential == []


# -----------------------------------------------------------------------------
# Mode selection
# -----------------------------------------------------------------------------

def test_unknown_mode_is_refused(tmp_path):
    rc = HR.run(s3_client=FakeS3([]), window=_window(),
                workspace_root=str(tmp_path), recon_models_dir="/m/r",
                feat_models_dir="/m/f", mode="turbo", verbose=False)
    assert rc == 2


def test_historical_defaults_to_sequential(monkeypatch):
    """`--historical` with no `--mode` must run SEQUENTIAL.

    `DEFAULT_MODE` is batch for the live paths, so reusing it here would silently
    give every historical run the pipeline the operator did not ask for.
    """
    from orchestrator import master_runner as MR
    monkeypatch.delenv(MR.MODE_ENV_VAR, raising=False)
    captured = {}
    monkeypatch.setattr(MR, "run_historical",
                        lambda args, **kw: captured.update(kw) or 0)
    rc = MR.main(["--historical", "--date", "2026-07-24",
                  "--start-time", "07:00", "--end-time", "08:00",
                  "--no-interactive", "--features", "door"])
    assert rc == 0
    assert captured["mode"] == MR.MODE_SEQUENTIAL


def test_explicit_mode_batch_is_honoured(monkeypatch):
    from orchestrator import master_runner as MR
    monkeypatch.delenv(MR.MODE_ENV_VAR, raising=False)
    captured = {}
    monkeypatch.setattr(MR, "run_historical",
                        lambda args, **kw: captured.update(kw) or 0)
    MR.main(["--historical", "--date", "2026-07-24", "--mode", "batch",
             "--no-interactive", "--features", "door"])
    assert captured["mode"] == MR.MODE_BATCH


def test_env_mode_is_honoured(monkeypatch):
    from orchestrator import master_runner as MR
    monkeypatch.setenv(MR.MODE_ENV_VAR, "batch")
    captured = {}
    monkeypatch.setattr(MR, "run_historical",
                        lambda args, **kw: captured.update(kw) or 0)
    MR.main(["--historical", "--date", "2026-07-24",
             "--no-interactive", "--features", "door"])
    assert captured["mode"] == MR.MODE_BATCH


# -----------------------------------------------------------------------------
# Partial batches
# -----------------------------------------------------------------------------

def test_partial_batch_is_partial_not_silently_assembled(tmp_path, monkeypatch):
    """Sealed cameras but no combined report -> PARTIAL, and inputs are kept.

    The sequential architecture requires all four cameras before assembly, because
    the master camera is whichever has the most unique gaps.  A missing camera
    therefore changes WHICH train gets built, not merely how well.  Historical
    mode must surface that instead of treating it as success.
    """
    from sequential import runner as seq_runner
    monkeypatch.setattr(seq_runner, "run_sequential",
                        lambda **kw: _Outcome(ready=False,
                                              sealed=[C.CAMERA_RIGHT_UP]))
    rc = HR.run(s3_client=FakeS3(_objects(times=("07:15:00",))),
                window=_window(), workspace_root=str(tmp_path),
                recon_models_dir="/m/r", feat_models_dir="/m/f",
                pad_minutes=15.0, tolerance_sec=300, dry_run=False,
                keep_inputs=True, deliver=False, send_email=False,
                mode="sequential", verbose=False)
    # COMPLETED_PARTIAL counts as OK for the loop, so exit is 0, but no PDF.
    assert rc == 0


def test_failed_batch_keeps_its_inputs(tmp_path, monkeypatch):
    from sequential import runner as seq_runner
    monkeypatch.setattr(seq_runner, "run_sequential",
                        lambda **kw: _Outcome(ready=False, sealed=[]))
    rc = HR.run(s3_client=FakeS3(_objects(times=("07:15:00",))),
                window=_window(), workspace_root=str(tmp_path),
                recon_models_dir="/m/r", feat_models_dir="/m/f",
                pad_minutes=15.0, tolerance_sec=300, dry_run=False,
                keep_inputs=False, deliver=False, send_email=False,
                mode="sequential", verbose=False)
    assert rc == 3
    downloads = tmp_path / HR.HISTORICAL_SUBDIR / "20260724_071255" / "downloads"
    assert downloads.is_dir(), "a FAILED batch's inputs must survive for diagnosis"


# -----------------------------------------------------------------------------
# IST anchoring
# -----------------------------------------------------------------------------

def test_batch_age_is_anchored_to_ist():
    """Filename digits are IST wall clock, not UTC.

    Reading them as UTC makes every batch look 5h30m younger, so an age-based
    gate misjudges every clip on a UTC box.
    """
    now_ist = datetime.now(IST).strftime("%Y%m%d_%H%M%S")
    b = TrainBatch(batch_key=now_ist, train_timestamp=now_ist)
    assert -5 < b.age_seconds() < 120


def test_historical_does_not_print_the_generic_mode_line(monkeypatch, capsys):
    """`Execution mode: BATCH` must not appear above a SEQUENTIAL historical run.

    The generic line reports `resolve_mode()`, which historical mode does not
    use.  Printing both is how a log comes to say BATCH directly above a run that
    was sequential.
    """
    from orchestrator import master_runner as MR
    monkeypatch.delenv(MR.MODE_ENV_VAR, raising=False)
    monkeypatch.setattr(MR, "run_historical", lambda args, **kw: 0)
    MR.main(["--historical", "--date", "2026-07-24",
             "--no-interactive", "--features", "door"])
    out = capsys.readouterr().out
    assert "Execution mode:" not in out
    assert "Historical pipeline mode: SEQUENTIAL" in out
