"""A failed global post logs the receiver's REASON, not just the status.

A 12-train bulk run produced twelve identical `FAILED: HTTP 409` lines that said
the request was refused and nothing about why. `_post` had captured the response
body all along; it simply was not logged, so the reason had to be re-obtained by
hand with a manual POST after the fact.
"""

from __future__ import annotations

import logging

from delivery import global_train_webhook as W


class _Resp:
    def __init__(self, code, text):
        self.status_code = code
        self.text = text

    def json(self):
        raise ValueError("not json")


class _Requests:
    def __init__(self, resp):
        self._resp = resp

    def post(self, url, json=None, headers=None, timeout=None):
        return self._resp


def test_the_body_is_logged_on_failure(caplog):
    reason = "global run already exists for rake 2 on 2026-07-24"
    r = W._post("https://x/ingest-global", {"a": 1},
                requests_mod=_Requests(_Resp(409, reason)))
    assert r["ok"] is False
    assert reason in str(r.get("body", "")), "the body was not captured"


def test_the_log_line_carries_the_detail(caplog):
    """The warning must contain the reason, so a bulk run is diagnosable."""
    import inspect
    src = inspect.getsource(W)
    assert 'detail = r.get("body")' in src
    assert "FAILED: %s -- %s" in src


def test_a_bodyless_failure_still_logs_cleanly():
    """No body (a timeout, a connection error) must not crash or print 'None'."""
    class _Boom:
        def post(self, *a, **k):
            raise RuntimeError("connection reset")

    r = W._post("https://x/ingest-global", {"a": 1}, requests_mod=_Boom())
    assert r["ok"] is False
    assert "connection reset" in r["error"]
    assert not r.get("body")
