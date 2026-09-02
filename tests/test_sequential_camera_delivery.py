"""A camera is delivered when IT seals, not when the slowest camera does.

Sequential's reason to exist is that a camera's findings are reachable as soon as
that camera is done. Holding every upload until after global assembly threw that
away -- a report sat on disk for up to twenty minutes before anyone could reach
it.

These tests also pin what early delivery must NOT do: the per-camera DASHBOARD
document is built from the canonical roster and the fused unified states, so it
cannot be published before assembly. Publishing one early would mean a document
keyed to a camera's own local wagons -- a competing global wagon list.
"""

from __future__ import annotations

import os

import pytest

from core import constants as C
from sequential import evidence as ev
from sequential import runner as R


class _Result:
    def __init__(self, camera_id, sealed=True):
        self.camera_id = camera_id
        self.status = ev.STATUS_SEALED if sealed else ev.STATUS_FAILED
        self.sealed = sealed


@pytest.fixture
def workspace(tmp_path):
    """A workspace with one camera's report and evidence already written."""
    cam = C.CAMERA_RIGHT_UP
    d = tmp_path / ev.CAMERA_REPORTS_DIRNAME / cam
    d.mkdir(parents=True)
    (d / ("%s_report.json" % cam)).write_text("{}")
    (d / ("%s_report.pdf" % cam)).write_bytes(b"%PDF-1.4")
    e = tmp_path / "evidence" / cam
    e.mkdir(parents=True)
    (e / "frame.jpg").write_bytes(b"\xff\xd8\xff")
    return str(tmp_path)


@pytest.fixture
def spy(monkeypatch):
    """Record what upload_tree was asked to upload."""
    calls = []

    def _upload_tree(s3, local, batch_key, sub_prefix=None, **kw):
        calls.append({"local": local, "batch_key": batch_key,
                      "sub_prefix": sub_prefix})
        return sum(len(f) for _r, _d, f in os.walk(local))

    from delivery import s3_upload
    monkeypatch.setattr(s3_upload, "upload_tree", _upload_tree)

    import boto3
    monkeypatch.setattr(boto3, "client", lambda *a, **k: object())
    return calls


def test_a_sealed_camera_is_uploaded_immediately(workspace, spy):
    ok = R.deliver_camera(_Result(C.CAMERA_RIGHT_UP), workspace=workspace,
                          batch_key="20260724_081227", skip_upload=False,
                          verbose=False)
    assert ok
    prefixes = sorted(c["sub_prefix"] for c in spy)
    assert prefixes == ["camera_reports/RIGHT_UP", "evidence/RIGHT_UP"]
    assert all(c["batch_key"] == "20260724_081227" for c in spy)


def test_only_this_camera_is_uploaded(workspace, spy):
    """A camera must not carry another camera's artifacts up with it."""
    R.deliver_camera(_Result(C.CAMERA_RIGHT_UP), workspace=workspace,
                     batch_key="k", skip_upload=False, verbose=False)
    for c in spy:
        assert C.CAMERA_RIGHT_UP in c["sub_prefix"]
        for other in (C.CAMERA_LEFT_UP, C.CAMERA_RIGHT_UP_TOP,
                      C.CAMERA_LEFT_UP_TOP):
            assert other not in c["sub_prefix"]


def test_an_unsealed_camera_delivers_nothing(workspace, spy):
    """A failed camera's partial output must not be published."""
    ok = R.deliver_camera(_Result(C.CAMERA_RIGHT_UP, sealed=False),
                          workspace=workspace, batch_key="k",
                          skip_upload=False, verbose=False)
    assert ok is False
    assert spy == []


def test_skip_upload_delivers_nothing(workspace, spy):
    ok = R.deliver_camera(_Result(C.CAMERA_RIGHT_UP), workspace=workspace,
                          batch_key="k", skip_upload=True, verbose=False)
    assert ok is False
    assert spy == []


def test_a_camera_with_no_report_on_disk_is_not_an_error(tmp_path, spy):
    ok = R.deliver_camera(_Result(C.CAMERA_LEFT_UP), workspace=str(tmp_path),
                          batch_key="k", skip_upload=False, verbose=False)
    assert ok is False
    assert spy == []


def test_an_upload_failure_does_not_raise(workspace, monkeypatch):
    """A receiver outage costs an early copy, not the camera or the run."""
    from delivery import s3_upload

    def _boom(*a, **k):
        raise RuntimeError("bucket unreachable")

    monkeypatch.setattr(s3_upload, "upload_tree", _boom)
    import boto3
    monkeypatch.setattr(boto3, "client", lambda *a, **k: object())

    ok = R.deliver_camera(_Result(C.CAMERA_RIGHT_UP), workspace=workspace,
                          batch_key="k", skip_upload=False, verbose=False)
    assert ok is False


def test_no_s3_client_does_not_raise(workspace, monkeypatch):
    import boto3

    def _boom(*a, **k):
        raise RuntimeError("no credentials")

    monkeypatch.setattr(boto3, "client", _boom)
    assert R.deliver_camera(_Result(C.CAMERA_RIGHT_UP), workspace=workspace,
                            batch_key="k", skip_upload=False,
                            verbose=False) is False


def test_early_delivery_does_not_touch_the_dashboard(workspace, spy,
                                                     monkeypatch):
    """The per-camera dashboard document needs the canonical roster.

    It is derived from `report_doc["wagons"]` and the fused unified states so
    that all four camera documents describe the same wagon sequence. Publishing
    one before assembly would key it to this camera's own local wagons -- a
    competing global wagon list, which the architecture forbids.
    """
    from delivery import dashboard_ingest

    def _must_not_run(**kw):
        raise AssertionError("dashboard ingest ran before global assembly")

    monkeypatch.setattr(dashboard_ingest, "run", _must_not_run)
    R.deliver_camera(_Result(C.CAMERA_RIGHT_UP), workspace=workspace,
                     batch_key="k", skip_upload=False, verbose=False)


def test_the_runner_calls_it_inside_the_camera_loop():
    """Not after the loop -- that would be the behaviour it replaces."""
    import inspect
    src = inspect.getsource(R.run_sequential)
    loop = src.index("for camera_id in order:")
    gate = src.index("required = global_assembly.required_cameras()")
    call = src.index("deliver_camera(")
    assert loop < call < gate, "early delivery is not inside the camera loop"
