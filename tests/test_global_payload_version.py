"""The global POST carries the same dashboard `version` as the per-camera POSTs.

`version` selects the dashboard TAB ("v1" -> the V1 tab). The per-camera payload
has always carried it; the global one did not, so nothing told the receiver which
tab the global document belonged to while its four siblings were filed under v1.

This is a different axis from `global_train_data["schema"]`
("wagon_eye.combined_report.v4"), which names the report FORMAT. A V4-format
report displayed in the V1 tab is correct.
"""

from __future__ import annotations

from delivery import global_train_webhook as W
from delivery import dashboard_ingest as D


def test_version_matches_the_per_camera_accessor(monkeypatch):
    """One accessor: the two POSTs can never name different tabs."""
    monkeypatch.setenv("WAGONEYE_INSPECTION_VERSION", "v1")
    assert D._version() == "v1"
    monkeypatch.setenv("WAGONEYE_INSPECTION_VERSION", "v7")
    assert D._version() == "v7", "the global payload must follow this accessor"


def test_schema_is_not_the_tab():
    """`schema` names the report format and must not be read as a tab.

    Conflating them would file the global document under a V4 tab while its four
    per-camera siblings sit under V1.
    """
    from reporting import combined_train_report as R
    assert R._REPORT_SCHEMA == "wagon_eye.combined_report.v4"
    assert R._REPORT_SCHEMA != D._version()


def test_the_body_builder_sets_version():
    import inspect
    src = inspect.getsource(W)
    assert 'body["version"]' in src
    assert "_tab_version()" in src
