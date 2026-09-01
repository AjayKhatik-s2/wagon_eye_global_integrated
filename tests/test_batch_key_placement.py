"""`batch_key` appears BOTH at the top level and inside `inspection_data`.

The backend reads `inspection_data`, so a sibling of it is out of reach without a
second lookup. Both copies are written from one variable, so they cannot disagree,
and the top-level key is kept because it is already the published contract.

The value is the SHARED TrainBatch key -- never re-derived per camera. The four
cameras of one train are stamped up to several minutes apart, so a per-camera
derivation would hand the backend four different keys for one train, which is
exactly the correlation this field exists to provide.
"""

from __future__ import annotations

import inspect

from delivery import dashboard_ingest as D


def test_both_placements_are_written_from_the_same_variable():
    src = inspect.getsource(D)
    assert 'doc["batch_key"] = batch_key' in src
    assert 'doc["inspection_data"]["batch_key"] = batch_key' in src


def test_both_are_guarded_by_the_same_truthiness_check():
    """An empty key must add NEITHER, so the two can never diverge."""
    src = inspect.getsource(D)
    i = src.index('doc["batch_key"] = batch_key')
    j = src.index('doc["inspection_data"]["batch_key"] = batch_key')
    guard = src.rindex("if batch_key:", 0, i)
    # No intervening dedent/branch between the guard and the second write.
    between = src[guard:j]
    assert between.count("if ") == 1, "the two writes are on different branches"


def test_batch_key_is_not_rederived_from_the_camera_clip():
    """It comes from the report, which carries the shared TrainBatch key."""
    src = inspect.getsource(D)
    assert 'batch_key = report_doc.get("batch_key", "")' in src
