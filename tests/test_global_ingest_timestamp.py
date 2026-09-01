"""The GLOBAL_FUSED run must be filed under the TRAIN's instant, not now().

`global_ingest.py` falls back to `datetime.utcnow()` when `upload_timestamp` is
absent from the request, so a historical reprocess used to file the fused run
under the day it was reprocessed. The per-camera feed never had the bug -- it
puts the timestamp INSIDE the document, which the receiver reads -- so the four
camera runs and the fused run for one train disagreed about which day it was.

These tests pin the value, the format, and the timezone convention.
"""

import json
import os
import sys
import tempfile
import unittest
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from delivery import dashboard_ingest as DI
from delivery import global_train_webhook as W
from delivery import inspection_json as IJ

BATCH = "20260724_072511"
EXPECTED = "2026-07-24T07:25:11"


class _Resp:
    status_code = 200
    text = "{}"

    def json(self):
        return {"run_id": 6813, "segments_count": 59, "already_existed": False}


class _Requests:
    """Captures what actually went on the wire."""

    def __init__(self):
        self.posts = []

    def post(self, url, json=None, headers=None, timeout=None):
        self.posts.append((url, json))
        return _Resp()


def _report(tmpdir, batch_key=BATCH, wagons=59):
    path = os.path.join(tmpdir, "combined_train_report.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump({"batch_key": batch_key,
                   "wagons": [{"global_id": "GW_%d" % i}
                              for i in range(1, wagons + 1)]}, fh)
    return path


class TrainUploadTimestamp(unittest.TestCase):

    # -- 1. the value --------------------------------------------------------

    def test_batch_key_yields_the_trains_own_instant(self):
        self.assertEqual(W.train_upload_timestamp(BATCH), EXPECTED)

    def test_a_second_batch_key(self):
        self.assertEqual(W.train_upload_timestamp("20260724_155656"),
                         "2026-07-24T15:56:56")

    # -- 3. NOT converted to UTC --------------------------------------------

    def test_is_ist_wall_clock_not_utc(self):
        """UTC would be 01:55:11 and would move a pre-05:30 train to the day
        before. The digits must survive unshifted."""
        got = W.train_upload_timestamp(BATCH)
        self.assertEqual(got, EXPECTED)
        self.assertNotEqual(got, "2026-07-24T01:55:11")
        self.assertNotIn("Z", got)
        self.assertNotIn("+", got)          # naive: no offset suffix
        self.assertEqual(datetime.fromisoformat(got).hour, 7)

    def test_a_pre_dawn_train_keeps_its_own_date(self):
        """02:10 IST is 20:40 the PREVIOUS day in UTC -- the case where a
        timezone conversion would silently change the dashboard date."""
        self.assertEqual(W.train_upload_timestamp("20260724_021000"),
                         "2026-07-24T02:10:00")

    # -- format parity with the per-camera feed ------------------------------

    def test_format_is_identical_to_the_per_camera_document(self):
        ts = DI.extract_train_timestamp(BATCH)
        self.assertEqual(W.train_upload_timestamp(BATCH),
                         ts.strftime("%Y-%m-%dT%H:%M:%S"))

    def test_derived_through_the_same_function_as_per_camera(self):
        """Not a second parser: both go through extract_train_timestamp."""
        self.assertEqual(
            W.train_upload_timestamp(BATCH),
            DI.extract_train_timestamp(BATCH).strftime("%Y-%m-%dT%H:%M:%S"))

    # -- 4. unparseable keys keep the receiver's optional-field fallback -----

    def test_no_timestamp_in_the_key_yields_none(self):
        self.assertIsNone(W.train_upload_timestamp("no-digits-here"))
        self.assertIsNone(W.train_upload_timestamp(""))


class GlobalFusedPayload(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.req = _Requests()

    def tearDown(self):
        self.tmp.cleanup()

    def _publish(self, batch_key=BATCH, **kw):
        path = _report(self.tmp.name, batch_key=batch_key)
        return W.publish(report_json_path=path, batch_key=batch_key,
                         requests_mod=self.req, verbose=False, **kw)

    # -- 2. the payload on the wire -----------------------------------------

    def test_payload_carries_the_trains_timestamp(self):
        res = self._publish()
        self.assertTrue(res.posted)
        self.assertEqual(len(self.req.posts), 1)
        _url, body = self.req.posts[0]
        self.assertEqual(body["upload_timestamp"], EXPECTED)

    def test_payload_still_carries_the_document_inline(self):
        """The pre-existing contract must be untouched."""
        self._publish()
        _url, body = self.req.posts[0]
        self.assertEqual(sorted(body),
                         ["camera_id", "global_train_data", "upload_timestamp"])
        self.assertEqual(body["camera_id"], "GLOBAL_FUSED")
        self.assertEqual(body["global_train_data"]["batch_key"], BATCH)
        self.assertEqual(len(body["global_train_data"]["wagons"]), 59)
        self.assertNotIn("inspection_s3_uri", body)   # inline, never a pointer

    def test_timestamp_is_not_today(self):
        """The bug: without the field the receiver stamped utcnow()."""
        self._publish()
        _url, body = self.req.posts[0]
        self.assertNotEqual(body["upload_timestamp"][:10],
                            datetime.now().strftime("%Y-%m-%d"))
        self.assertEqual(body["upload_timestamp"][:10], "2026-07-24")

    def test_result_records_what_was_sent(self):
        res = self._publish()
        self.assertEqual(res.upload_timestamp, EXPECTED)
        self.assertEqual(res.to_dict()["upload_timestamp"], EXPECTED)

    # -- 4. backward compatibility -----------------------------------------

    def test_unparseable_batch_key_omits_the_field_entirely(self):
        """Never guess. Omitting it leaves the receiver's documented
        `or datetime.utcnow()` fallback exactly as it was."""
        res = self._publish(batch_key="live-run")
        _url, body = self.req.posts[0]
        self.assertNotIn("upload_timestamp", body)
        self.assertEqual(res.upload_timestamp, "")
        self.assertTrue(res.posted)

    def test_the_key_inside_the_document_wins_over_the_argument(self):
        """`publish` already prefers doc['batch_key']; the timestamp follows it."""
        path = _report(self.tmp.name, batch_key="20260724_155656")
        res = W.publish(report_json_path=path, batch_key="20260101_000000",
                        requests_mod=self.req, verbose=False)
        _url, body = self.req.posts[0]
        self.assertEqual(res.batch_key, "20260724_155656")
        self.assertEqual(body["upload_timestamp"], "2026-07-24T15:56:56")

    def test_still_uat_only(self):
        self._publish()
        url, _body = self.req.posts[0]
        self.assertIn("cctv-wagon-uat-api", url)
        self.assertTrue(url.endswith("/inspections/ingest-global"))
        self.assertNotIn("ms-pnr-location-notification-api", url)


class PerCameraAndGlobalAgree(unittest.TestCase):
    """Acceptance criterion: the five runs for one train share one timestamp."""

    def test_same_string_both_paths(self):
        ts = DI.extract_train_timestamp(BATCH)
        per_camera = ts.strftime("%Y-%m-%dT%H:%M:%S")     # inspection_json:755
        self.assertEqual(per_camera, EXPECTED)
        self.assertEqual(W.train_upload_timestamp(BATCH), per_camera)

    def test_the_per_camera_serialization_is_what_we_mirror(self):
        """Guards the mirror: if inspection_json ever changes its format, this
        fails rather than letting the two drift apart silently."""
        import inspect
        src = inspect.getsource(IJ.build_inspection_json)
        self.assertIn('strftime("%Y-%m-%dT%H:%M:%S")', src)


if __name__ == "__main__":
    unittest.main()
