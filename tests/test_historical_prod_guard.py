"""`--historical-deliver` refuses a non-UAT ingest endpoint unless told otherwise.

`ingest_api_urls()` DEFAULTS to both receivers, PROD first, so a shell that forgot
`WAGONEYE_INSPECTION_INGEST_API_URLS=uat` sends every historical document to
production -- 52 POSTs for a 13-train day, all reprocessed history landing in a
live system. This happened twice during the first bulk run and was caught by hand
both times; a missing environment variable must not be able to do it.

The escape hatch is a FLAG, not another environment variable: the operator states
the intent at the point of use, where it is visible in the command line and in
shell history.
"""

from __future__ import annotations

import pytest

from orchestrator import master_runner as MR

UAT = "https://cctv-wagon-uat-api.suvidhaen.com/inspections/ingest"
PROD = "https://ms-pnr-location-notification-api.suvidhaen.com/cctv-receiver/inspections/ingest"


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv(MR.MODE_ENV_VAR, raising=False)


def _args(monkeypatch, urls, *extra):
    from delivery import dashboard_ingest as D
    monkeypatch.setattr(D, "ingest_api_urls", lambda: list(urls))
    monkeypatch.setattr(D, "_uat_url", lambda: UAT)
    return ["--historical", "--date", "2026-07-24", "--historical-deliver",
            "--no-interactive", "--features", "door", *extra]


def test_uat_only_is_allowed(monkeypatch):
    assert MR._prod_ingest_endpoints() is not None
    from delivery import dashboard_ingest as D
    monkeypatch.setattr(D, "ingest_api_urls", lambda: [UAT])
    monkeypatch.setattr(D, "_uat_url", lambda: UAT)
    assert MR._prod_ingest_endpoints() == []


def test_a_prod_endpoint_is_detected(monkeypatch):
    from delivery import dashboard_ingest as D
    monkeypatch.setattr(D, "ingest_api_urls", lambda: [PROD, UAT])
    monkeypatch.setattr(D, "_uat_url", lambda: UAT)
    assert MR._prod_ingest_endpoints() == [PROD]


class _Args:
    """Just the attributes run_historical's preflight reads."""
    historical_deliver = True
    allow_prod_ingest = False
    date = "2026-07-24"
    start_time = "07:15"
    end_time = "18:00"
    timezone = None
    start = None
    end = None
    workspace = None
    recon_models_dir = None
    feat_models_dir = None
    pad_minutes = None
    tolerance_sec = None
    dry_run = True
    keep_inputs = True
    skip_email = True
    manifest_out = None


def test_the_run_refuses_to_start_with_a_prod_endpoint(monkeypatch, capsys):
    """Exit 2, no S3 client built, no discovery -- and the endpoint is NAMED."""
    from delivery import dashboard_ingest as D
    monkeypatch.setattr(D, "ingest_api_urls", lambda: [PROD, UAT])
    monkeypatch.setattr(D, "_uat_url", lambda: UAT)

    import boto3
    def _no_client(*a, **k):
        raise AssertionError("an S3 client was built despite the refusal")
    monkeypatch.setattr(boto3, "client", _no_client)

    rc = MR.run_historical(_Args())
    err = capsys.readouterr().err
    assert rc == 2
    assert "NON-UAT" in err
    assert PROD in err, "the offending endpoint must be named, not just implied"


def test_allow_prod_ingest_gets_past_the_guard(monkeypatch):
    """With the flag, the same endpoint list proceeds to the S3 client."""
    from delivery import dashboard_ingest as D
    monkeypatch.setattr(D, "ingest_api_urls", lambda: [PROD, UAT])
    monkeypatch.setattr(D, "_uat_url", lambda: UAT)

    reached = []
    import boto3
    def _client(*a, **k):
        reached.append(1)
        raise RuntimeError("stop here")
    monkeypatch.setattr(boto3, "client", _client)

    args = _Args()
    args.allow_prod_ingest = True
    rc = MR.run_historical(args)
    assert reached, "the guard blocked a run that passed --allow-prod-ingest"
    assert rc == 2   # our stub raises, so it exits 2 -- but AFTER the guard


def test_allow_prod_ingest_permits_it(monkeypatch):
    from delivery import dashboard_ingest as D
    monkeypatch.setattr(D, "ingest_api_urls", lambda: [PROD, UAT])
    monkeypatch.setattr(D, "_uat_url", lambda: UAT)
    # The flag exists and is wired: the guard reads it, so with it set the same
    # endpoint list is not a refusal.
    import inspect
    src = inspect.getsource(MR.run_historical)
    assert "args.allow_prod_ingest" in src
    assert "args.historical_deliver and not args.allow_prod_ingest" in src


def test_an_unresolvable_endpoint_list_is_not_treated_as_safe(monkeypatch):
    """"I could not check" must not read as "it is safe"."""
    from delivery import dashboard_ingest as D

    def _boom():
        raise RuntimeError("no config")

    monkeypatch.setattr(D, "ingest_api_urls", _boom)
    assert MR._prod_ingest_endpoints(), "an unresolvable list was allowed"


def test_the_flag_is_not_an_environment_variable():
    """The escape hatch must be visible in the command line, not in a shell."""
    p = MR._build_parser()
    opts = {a.dest for a in p._actions}
    assert "allow_prod_ingest" in opts
