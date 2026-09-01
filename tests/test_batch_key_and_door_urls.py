"""Two additive report fields.

REQ 1  wagons[].doors[].s3_url  in the GLOBAL report -- the door's own best
       snapshot, as an absolute URL, never fabricated.
REQ 2  batch_key  in EVERY per-camera document -- the ORCHESTRATOR's
       TrainBatch.batch_key, identical across all four cameras, while each
       camera keeps its own upload_timestamp.
"""

import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core import constants as C
from core.global_state_loader import load_global_train_state
from delivery import dashboard_ingest as DI
from delivery import finalization
from fusion import wagon_state_builder
from global_counting import adapter
from reporting import combined_train_report as CR
from test_global_counting_integration import build_harvest

BATCH = "20260724_072011"
BUCKET = "biputri-wagoneye-report"
BASE = ("https://%s.s3.ap-south-1.amazonaws.com/train_batch/%s/evidence"
        % (BUCKET, BATCH))

#: Each camera's OWN clip stamp -- deliberately different from BATCH and from
#: each other, which is the whole point of REQ 2.
CLIP_TS = {
    C.CAMERA_RIGHT_UP:     "20260724_072011",
    C.CAMERA_LEFT_UP:      "20260724_072430",
    C.CAMERA_RIGHT_UP_TOP: "20260724_072705",
    C.CAMERA_LEFT_UP_TOP:  "20260724_072245",
}


def _batch_root(tmpdir, *, doors_for=(C.CAMERA_LEFT_UP, C.CAMERA_RIGHT_UP),
                write_snapshots=True):
    """A finished batch tree: state, fused states, evidence, report."""
    br = os.path.join(tmpdir, BATCH)
    for d in ("global_state", "wagon_states", "evidence", "reports"):
        os.makedirs(os.path.join(br, d), exist_ok=True)
    sp = os.path.join(br, "global_state", "global_train_state.json")
    with open(sp, "w", encoding="utf-8") as fh:
        json.dump(adapter.build_global_train_state_document(build_harvest()),
                  fh, default=str)
    state = load_global_train_state(sp)
    ws, ev = os.path.join(br, "wagon_states"), os.path.join(br, "evidence")
    for feat in ("door", "load", "damage", "ocr"):
        os.makedirs(os.path.join(ws, feat), exist_ok=True)

    for i, w in enumerate(state.wagons, start=1):
        gw = w.global_id
        for rel in ("ocr/best_frame.jpg", "load/best_frame.jpg",
                    "damage/track_1.jpg"):
            p = os.path.join(ev, gw, *rel.split("/"))
            os.makedirs(os.path.dirname(p), exist_ok=True)
            with open(p, "wb") as fh:
                fh.write(b"\xff\xd8\xff\xdb")
        if write_snapshots:
            for cam in doors_for:
                fn = "left_best.jpg" if cam == C.CAMERA_LEFT_UP else "right_best.jpg"
                p = os.path.join(ev, gw, "door", fn)
                os.makedirs(os.path.dirname(p), exist_ok=True)
                with open(p, "wb") as fh:
                    fh.write(b"\xff\xd8\xff\xdb")
        # one door per SIDE camera, each carrying its own camera_id
        doors = [{"camera_id": cam, "door_index": n, "side":
                  "left" if cam == C.CAMERA_LEFT_UP else "right",
                  "state": C.DOOR_CLOSED, "track_id": n, "total_hits": 9}
                 for n, cam in enumerate(
                     (C.CAMERA_LEFT_UP, C.CAMERA_RIGHT_UP), start=1)]
        for feat, extra in (
                ("door", {"supporting_cameras": list(C.SIDE_CAMERAS),
                          "left_door": C.DOOR_CLOSED,
                          "right_door": C.DOOR_CLOSED, "doors": doors}),
                ("load", {"supporting_cameras": [C.CAMERA_RIGHT_UP_TOP],
                          "load_status": C.LOAD_LOADED}),
                ("damage", {"supporting_cameras": list(C.TOP_CAMERAS),
                            "top_damage": C.DAMAGE_OK,
                            "top_damage_details": [], "per_camera": {}}),
                ("ocr", {"supporting_cameras": [C.CAMERA_RIGHT_UP],
                         "wagon_identifier": "3112345678%d" % (i % 10)})):
            with open(os.path.join(ws, feat, "%s.json" % gw), "w") as fh:
                json.dump(dict({"global_id": gw, "feature": feat,
                                "status": C.STATUS_OK}, **extra), fh)

    unified = wagon_state_builder.build(state=state, wagon_states_root=ws,
                                        verbose=False)
    return br, state, unified, ws, ev


def _report(br, state, unified, ws, ev, *, base=BASE):
    """Stage 5b, with the per-camera clip URLs the real orchestrator passes."""
    return CR.build(
        state=state, unified=unified,
        output_dir=os.path.join(br, "reports"), batch_key=BATCH,
        evidence_root=ev, evidence_url_base=base, wagon_states_root=ws,
        source_video_urls={
            cam: ("https://%s.s3.ap-south-1.amazonaws.com/%s/%s_%s_train.mp4"
                  % (BUCKET, C.CAMERA_S3_FOLDER[cam],
                     C.CAMERA_S3_FOLDER[cam], ts))
            for cam, ts in CLIP_TS.items()},
        verbose=False)


# ---------------------------------------------------------------------------
# REQ 1 -- doors[].s3_url
# ---------------------------------------------------------------------------

class DoorS3Urls(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.br, st, un, ws, self.ev = _batch_root(self.tmp.name)
        with open(_report(self.br, st, un, ws, self.ev)["json_path"]) as fh:
            self.doc = json.load(fh)
        self.wagon = self.doc["wagons"][0]
        self.gw = self.wagon["global_id"]

    def tearDown(self):
        self.tmp.cleanup()

    def _door(self, camera):
        return next(d for d in self.wagon["doors"] if d["camera_id"] == camera)

    # 1. LEFT_UP -> door_left_best
    def test_left_up_gets_left_best(self):
        self.assertEqual(
            self._door(C.CAMERA_LEFT_UP)["s3_url"],
            "%s/%s/door/left_best.jpg" % (BASE, self.gw))

    # 2. RIGHT_UP -> door_right_best
    def test_right_up_gets_right_best(self):
        self.assertEqual(
            self._door(C.CAMERA_RIGHT_UP)["s3_url"],
            "%s/%s/door/right_best.jpg" % (BASE, self.gw))

    def test_the_two_sides_never_share_a_url(self):
        self.assertNotEqual(self._door(C.CAMERA_LEFT_UP)["s3_url"],
                            self._door(C.CAMERA_RIGHT_UP)["s3_url"])

    # 3. bucket / region / prefix / batch_key / evidence path
    def test_url_components(self):
        url = self._door(C.CAMERA_RIGHT_UP)["s3_url"]
        for part in (BUCKET, "ap-south-1", C.S3_TRAIN_BATCH_PREFIX, BATCH,
                     "evidence", "%s/door/right_best.jpg" % self.gw):
            self.assertEqual(url.count(part), 1, "%r in %s" % (part, url))
        self.assertEqual(url.count("https://"), 1)
        self.assertNotIn("//train_batch", url)
        self.assertTrue(url.startswith(
            "https://%s.s3.ap-south-1.amazonaws.com/%s/%s/evidence/"
            % (BUCKET, C.S3_TRAIN_BATCH_PREFIX, BATCH)))

    def test_every_wagon_gets_both_doors(self):
        for wagon in self.doc["wagons"]:
            urls = {d["camera_id"]: d.get("s3_url") for d in wagon["doors"]}
            self.assertEqual(sorted(urls), sorted(C.SIDE_CAMERAS))
            for cam, url in urls.items():
                self.assertTrue(url and wagon["global_id"] in url)

    # 4. missing evidence -> NO url invented
    def test_missing_snapshot_yields_no_url(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        br, st, un, ws, ev = _batch_root(tmp.name, write_snapshots=False)
        with open(_report(br, st, un, ws, ev)["json_path"]) as fh:
            doc = json.load(fh)
        for wagon in doc["wagons"]:
            for d in wagon["doors"]:
                self.assertNotIn("s3_url", d)

    def test_only_the_present_side_gets_a_url(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        br, st, un, ws, ev = _batch_root(tmp.name,
                                         doors_for=(C.CAMERA_LEFT_UP,))
        with open(_report(br, st, un, ws, ev)["json_path"]) as fh:
            doc = json.load(fh)
        for wagon in doc["wagons"]:
            by_cam = {d["camera_id"]: d for d in wagon["doors"]}
            self.assertIn("s3_url", by_cam[C.CAMERA_LEFT_UP])
            self.assertNotIn("s3_url", by_cam[C.CAMERA_RIGHT_UP])

    def test_no_base_means_no_urls(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        br, st, un, ws, ev = _batch_root(tmp.name)
        with open(_report(br, st, un, ws, ev, base=None)["json_path"]) as fh:
            doc = json.load(fh)
        for wagon in doc["wagons"]:
            for d in wagon["doors"]:
                self.assertNotIn("s3_url", d)

    def test_unit_never_invents(self):
        self.assertIsNone(CR._door_s3_url("GW_1", C.CAMERA_LEFT_UP, self.ev, None))
        self.assertIsNone(CR._door_s3_url("GW_1", C.CAMERA_LEFT_UP, None, BASE))
        self.assertIsNone(CR._door_s3_url("GW_1", "", self.ev, BASE))
        # a TOP camera photographs no door
        self.assertIsNone(CR._door_s3_url("GW_1", C.CAMERA_RIGHT_UP_TOP,
                                          self.ev, BASE))

    # 5. evidence_pages unchanged, and no evidence_page_urls
    def test_evidence_pages_untouched(self):
        self.assertIn("evidence_pages", self.doc)
        for snaps in self.doc["evidence_pages"].values():
            for rel in snaps.values():
                self.assertFalse(rel.startswith("http"), rel)
        self.assertNotIn("evidence_page_urls", self.doc)

    # 6. pre-existing door + wagon fields preserved
    def test_existing_door_fields_preserved(self):
        d = self._door(C.CAMERA_RIGHT_UP)
        for k in ("camera_id", "door_index", "side", "state", "track_id",
                  "total_hits"):
            self.assertIn(k, d)
        self.assertEqual(d["state"], C.DOOR_CLOSED)
        self.assertEqual(d["side"], "right")

    def test_wagon_level_fields_preserved(self):
        for k in ("global_id", "wagon_index", "classification", "left_door",
                  "right_door", "load_status", "top_damage", "doors"):
            self.assertIn(k, self.wagon)

    # 5 (global). the report keeps its own batch_key
    def test_global_report_keeps_batch_key(self):
        self.assertEqual(self.doc["batch_key"], BATCH)


# ---------------------------------------------------------------------------
# REQ 2 -- batch_key in every per-camera document
# ---------------------------------------------------------------------------

class _Resp:
    status_code = 200
    text = "{}"

    def json(self):
        return {"run_id": 1}


class _S3:
    def upload_file(self, *a, **k):
        pass


class _R:
    @staticmethod
    def post(url, json=None, headers=None, timeout=None):
        return _Resp()


class PerCameraBatchKey(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.br, st, un, ws, ev = _batch_root(self.tmp.name)
        _report(self.br, st, un, ws, ev)
        finalization.write(self.br, {"batch_key": BATCH, "uploaded": True,
                                     "upload_urls": {"pdf": "https://x/c.pdf"}})
        DI.run(batch_root=self.br, s3_client=_S3(), requests_mod=_R)
        self.docs = {}
        d = os.path.join(self.br, "delivery", "dashboard")
        for fn in sorted(os.listdir(d)):
            with open(os.path.join(d, fn), encoding="utf-8") as fh:
                doc = json.load(fh)
            self.docs[fn.split("_camera")[0]] = doc

    def tearDown(self):
        self.tmp.cleanup()

    # 1. present in every document
    def test_batch_key_in_every_camera_document(self):
        self.assertEqual(sorted(self.docs), sorted(C.ALL_CAMERAS))
        for cam, doc in self.docs.items():
            self.assertIn("batch_key", doc, cam)
            self.assertEqual(doc["batch_key"], BATCH, cam)

    # 2. identical across cameras
    def test_all_cameras_share_one_batch_key(self):
        keys = {doc["batch_key"] for doc in self.docs.values()}
        self.assertEqual(keys, {BATCH})

    # 3. different clip timestamps do NOT change it
    def test_differing_timestamps_do_not_change_the_batch_key(self):
        stamps = {doc["inspection_data"]["upload_timestamp"]
                  for doc in self.docs.values()}
        self.assertGreater(len(stamps), 1, "fixture must have differing stamps")
        self.assertEqual({doc["batch_key"] for doc in self.docs.values()},
                         {BATCH})

    # 4. each camera keeps its OWN upload_timestamp
    def test_upload_timestamp_stays_per_camera(self):
        from datetime import datetime
        for cam, ts in CLIP_TS.items():
            expected = datetime.strptime(ts, "%Y%m%d_%H%M%S").strftime(
                "%Y-%m-%dT%H:%M:%S")
            self.assertEqual(
                self.docs[cam]["inspection_data"]["upload_timestamp"], expected,
                cam)

    def test_batch_key_is_top_level_beside_camera_id_and_version(self):
        doc = self.docs[C.CAMERA_RIGHT_UP]
        for k in ("camera_id", "version", "batch_key", "inspection_data"):
            self.assertIn(k, doc)

    def test_pre_existing_document_shape_kept(self):
        doc = self.docs[C.CAMERA_RIGHT_UP]
        self.assertEqual(doc["version"], DI._version())
        self.assertTrue(doc["camera_id"].startswith("camera_CCTV_"))
        self.assertIn("wagon_segments", doc["inspection_data"])
        self.assertIn("_adapter", doc["inspection_data"])


if __name__ == "__main__":
    unittest.main()
