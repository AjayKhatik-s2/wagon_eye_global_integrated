"""`wagons[].wagon_frames` -- four positional frames per camera ANGLE.

    wagon_cache/<GW>/<camera>/frame_NNNNNN.jpg
      -> evidence/<GW>/wagon_frames/<angle>/w<N>_frame_<IDX>.jpg
      -> https://<bucket>.s3.<region>.amazonaws.com/train_batch/<key>/evidence/...

The URL must name an object Stage 6 actually uploads, which is why the frames go
through `evidence/` (upload_tree mirrors it) and not `wagon_cache/` (never
uploaded).
"""

import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core import constants as C
from core import wagon_frames as WF
from core.global_state_loader import load_global_train_state
from fusion import wagon_state_builder
from global_counting import adapter
from reporting import combined_train_report as CR
from test_global_counting_integration import build_harvest

BATCH = "20260724_072511"
BUCKET = "biputri-wagoneye-report"
BASE = ("https://%s.s3.ap-south-1.amazonaws.com/train_batch/%s/evidence"
        % (BUCKET, BATCH))
FPS, TOTAL = 15.0, 4000
META = {cam: {"fps": FPS, "total_frames": TOTAL} for cam in C.ALL_CAMERAS}


# ---------------------------------------------------------------------------
# 2. angle -> physical camera
# ---------------------------------------------------------------------------

class AngleMapping(unittest.TestCase):

    def test_four_angles_exactly(self):
        self.assertEqual(list(WF.ANGLE_BY_CAMERA.values()),
                         ["left_up", "right_up", "right_top", "left_top"])

    def test_each_angle_maps_to_the_right_camera(self):
        self.assertEqual(WF.ANGLE_BY_CAMERA[C.CAMERA_LEFT_UP], "left_up")
        self.assertEqual(WF.ANGLE_BY_CAMERA[C.CAMERA_RIGHT_UP], "right_up")
        self.assertEqual(WF.ANGLE_BY_CAMERA[C.CAMERA_RIGHT_UP_TOP], "right_top")
        self.assertEqual(WF.ANGLE_BY_CAMERA[C.CAMERA_LEFT_UP_TOP], "left_top")

    def test_angles_agree_with_the_authoritative_rig_folders(self):
        """`right_top` must be the camera whose rig folder says RIGHT_TOP."""
        expected = {
            "left_up":   "camera_CCTV_HZBN_DHN_1_LEFT_UP",
            "right_up":  "camera_CCTV_HZBN_DHN_2_RIGHT_UP",
            "right_top": "camera_CCTV_HZBN_DHN_5_RIGHT_TOP",
            "left_top":  "camera_CCTV_HZBN_DHN_6_LEFT_TOP",
        }
        for angle, folder in expected.items():
            cam = WF.CAMERA_BY_ANGLE[angle]
            self.assertEqual(C.CAMERA_S3_FOLDER[cam], folder, angle)

    def test_no_angle_is_reused(self):
        vals = list(WF.ANGLE_BY_CAMERA.values())
        self.assertEqual(len(set(vals)), 4)


# ---------------------------------------------------------------------------
# frame-range parity with the materializer
# ---------------------------------------------------------------------------

class _W:
    """Minimal stand-in with the GlobalWagon surface these functions read."""
    def __init__(self, gw="GW_1", idx=1, st=0.0, et=4.0, ranges=None):
        self.global_id, self.wagon_index = gw, idx
        self.start_time, self.end_time = st, et
        self._r = ranges or {}

    def local_range(self, camera_id):
        e = self._r.get(camera_id)
        return (e["start_frame"], e["end_frame"]) if e else None


class MaterializerParity(unittest.TestCase):
    """The window must match the one that WROTE the cache, or every frame is
    reported missing."""

    def _mat(self, wagon, cam, fps, total, offset=0.0):
        from materializer.wagon_cache_builder import _wagon_local_range
        return _wagon_local_range(wagon, fps, total, offset, cam)

    def test_explicit_range_path(self):
        w = _W(ranges={C.CAMERA_RIGHT_UP: {"start_frame": 300, "end_frame": 360}})
        self.assertEqual(
            WF.local_frame_range(w, C.CAMERA_RIGHT_UP, FPS, TOTAL),
            self._mat(w, C.CAMERA_RIGHT_UP, FPS, TOTAL))

    def test_time_projection_path(self):
        w = _W(st=20.0, et=24.0)
        self.assertEqual(
            WF.local_frame_range(w, C.CAMERA_RIGHT_UP, FPS, TOTAL),
            self._mat(w, C.CAMERA_RIGHT_UP, FPS, TOTAL))

    def test_offset_path(self):
        w = _W(st=20.0, et=24.0)
        self.assertEqual(
            WF.local_frame_range(w, C.CAMERA_LEFT_UP, FPS, TOTAL, 1.5),
            self._mat(w, C.CAMERA_LEFT_UP, FPS, TOTAL, 1.5))

    def test_wagon_outside_the_footage_yields_nothing(self):
        w = _W(st=9000.0, et=9004.0)
        self.assertEqual(WF.local_frame_range(w, C.CAMERA_RIGHT_UP, FPS, TOTAL),
                         (0, -1))


# ---------------------------------------------------------------------------
# path + URL construction
# ---------------------------------------------------------------------------

class Paths(unittest.TestCase):

    def test_filename_carries_wagon_and_frame(self):
        self.assertEqual(WF.frame_filename(39, 2957), "w39_frame_002957.jpg")

    def test_rel_path_is_under_evidence(self):
        self.assertEqual(
            WF.evidence_rel_path("GW_39", "right_up", "w39_frame_002957.jpg"),
            "GW_39/wagon_frames/right_up/w39_frame_002957.jpg")

    # 5. not the backend's sample bucket
    def test_url_uses_our_bucket_not_the_backend_sample(self):
        man = {"GW_1": {"right_up": [{"position": "start", "frame_idx": 10,
                                      "rel_path": "GW_1/wagon_frames/right_up/w1_frame_000010.jpg"}]}}
        url = WF.published(man, "GW_1", BASE)["right_up"][0]["s3_url"]
        self.assertNotIn("test-inspection-artifacts-sarva", url)
        self.assertIn(BUCKET, url)
        self.assertEqual(
            url, "https://%s.s3.ap-south-1.amazonaws.com/train_batch/%s/"
                 "evidence/GW_1/wagon_frames/right_up/w1_frame_000010.jpg"
                 % (BUCKET, BATCH))

    def test_no_base_publishes_nothing(self):
        man = {"GW_1": {"right_up": [{"position": "start", "frame_idx": 1,
                                      "rel_path": "x"}]}}
        self.assertEqual(WF.published(man, "GW_1", None), {})


# ---------------------------------------------------------------------------
# end to end through the real report
# ---------------------------------------------------------------------------

def _fixture(tmpdir, *, cameras=C.ALL_CAMERAS, cache_frac=1.0):
    br = os.path.join(tmpdir, BATCH)
    for d in ("global_state", "wagon_states", "evidence", "reports",
              "wagon_cache"):
        os.makedirs(os.path.join(br, d), exist_ok=True)
    sp = os.path.join(br, "global_state", "global_train_state.json")
    with open(sp, "w", encoding="utf-8") as fh:
        json.dump(adapter.build_global_train_state_document(build_harvest()),
                  fh, default=str)
    state = load_global_train_state(sp)
    ws = os.path.join(br, "wagon_states")
    ev = os.path.join(br, "evidence")
    cache = os.path.join(br, "wagon_cache")
    for feat in ("door", "load", "damage", "ocr"):
        os.makedirs(os.path.join(ws, feat), exist_ok=True)

    for i, w in enumerate(state.wagons, start=1):
        gw = w.global_id
        # door snapshots, so doors[].s3_url keeps working
        for fn in ("left_best.jpg", "right_best.jpg"):
            p = os.path.join(ev, gw, "door", fn)
            os.makedirs(os.path.dirname(p), exist_ok=True)
            open(p, "wb").write(b"\xff\xd8\xff\xdb")
        p = os.path.join(ev, gw, "ocr", "best_frame.jpg")
        os.makedirs(os.path.dirname(p), exist_ok=True)
        open(p, "wb").write(b"\xff\xd8\xff\xdb")
        # a cache of contiguous frames per camera, over the wagon's own window
        for cam in cameras:
            sf, ef = WF.local_frame_range(w, cam, FPS, TOTAL)
            if ef <= sf:
                continue
            folder = C.CAMERA_FOLDER[cam]
            d = os.path.join(cache, gw, folder)
            os.makedirs(d, exist_ok=True)
            # Cache the wagon's WHOLE range, as Stage 2 does. `cache_frac`
            # truncates it, to exercise the missing-frame path.
            last = sf + int((ef - sf) * cache_frac)
            for idx in range(sf, min(ef, last) + 1):
                open(os.path.join(d, "frame_%06d.jpg" % idx), "wb").write(b"\xff\xd8")
        doors = [{"camera_id": cam, "door_index": n,
                  "side": "left" if cam == C.CAMERA_LEFT_UP else "right",
                  "state": C.DOOR_CLOSED, "track_id": n}
                 for n, cam in enumerate((C.CAMERA_LEFT_UP, C.CAMERA_RIGHT_UP), 1)]
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
                         "wagon_identifier": "311234567%02d" % i})):
            with open(os.path.join(ws, feat, "%s.json" % gw), "w") as fh:
                json.dump(dict({"global_id": gw, "feature": feat,
                                "status": C.STATUS_OK}, **extra), fh)
    unified = wagon_state_builder.build(state=state, wagon_states_root=ws,
                                        verbose=False)
    return br, state, unified, ws, ev, cache


class EndToEnd(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        (self.br, self.state, self.unified,
         self.ws, self.ev, self.cache) = _fixture(self.tmp.name)
        self.man = WF.materialize(
            state=self.state, cache_root=self.cache, evidence_root=self.ev,
            per_camera_meta=META, verbose=False)
        with open(self._report()["json_path"], encoding="utf-8") as fh:
            self.doc = json.load(fh)
        self.wagon = self.doc["wagons"][0]

    def tearDown(self):
        self.tmp.cleanup()

    def _report(self, **kw):
        return CR.build(state=self.state, unified=self.unified,
                        output_dir=os.path.join(self.br, "reports"),
                        batch_key=BATCH, evidence_root=self.ev,
                        evidence_url_base=BASE, wagon_frames=self.man,
                        wagon_states_root=self.ws, cache_root=self.cache,
                        verbose=False, **kw)

    # 1 + 3. all four keys, each with the four positions
    def test_all_four_angles_present_with_four_positions(self):
        wf = self.wagon["wagon_frames"]
        self.assertEqual(sorted(wf), sorted(["left_up", "right_up",
                                             "right_top", "left_top"]))
        for angle, frames in wf.items():
            self.assertEqual([f["position"] for f in frames],
                             ["start", "mid1", "mid2", "end"], angle)

    def test_key_order_is_deterministic(self):
        self.assertEqual(list(self.wagon["wagon_frames"]),
                         ["left_up", "right_up", "right_top", "left_top"])

    # 2. each angle's URL sits under that angle's own folder
    def test_each_angle_url_is_in_its_own_folder(self):
        for angle, frames in self.wagon["wagon_frames"].items():
            for f in frames:
                self.assertIn("/wagon_frames/%s/" % angle, f["s3_url"])

    def test_angles_do_not_share_urls(self):
        seen = {}
        for angle, frames in self.wagon["wagon_frames"].items():
            for f in frames:
                self.assertNotIn(f["s3_url"], seen,
                                 "%s reused %s's URL" % (angle, seen.get(f["s3_url"])))
                seen[f["s3_url"]] = angle

    # 4. correct batch_key and S3 path
    def test_urls_carry_the_right_batch_key_and_prefix(self):
        for frames in self.wagon["wagon_frames"].values():
            for f in frames:
                self.assertTrue(f["s3_url"].startswith(
                    "https://%s.s3.ap-south-1.amazonaws.com/%s/%s/evidence/"
                    % (BUCKET, C.S3_TRAIN_BATCH_PREFIX, BATCH)), f["s3_url"])
                self.assertEqual(f["s3_url"].count(BATCH), 1)

    def test_url_names_a_file_that_exists_and_will_upload(self):
        """Stage 6 mirrors evidence/ verbatim, so on-disk == will-be-uploaded."""
        for frames in self.wagon["wagon_frames"].values():
            for f in frames:
                rel = f["s3_url"].split("/evidence/", 1)[1]
                self.assertTrue(os.path.isfile(os.path.join(self.ev, *rel.split("/"))),
                                rel)

    # 5. never the backend's sample values
    def test_no_backend_sample_values_anywhere(self):
        blob = json.dumps(self.doc)
        for bad in ("test-inspection-artifacts-sarva", "2026-07-26_05-26-05"):
            self.assertNotIn(bad, blob)

    # 6. missing local frame -> no fake URL
    def test_camera_with_no_cache_is_absent(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        br, st, un, ws, ev, cache = _fixture(
            tmp.name, cameras=(C.CAMERA_RIGHT_UP, C.CAMERA_LEFT_UP))
        man = WF.materialize(state=st, cache_root=cache, evidence_root=ev,
                             per_camera_meta=META, verbose=False)
        with open(CR.build(state=st, unified=un,
                           output_dir=os.path.join(br, "reports"),
                           batch_key=BATCH, evidence_root=ev,
                           evidence_url_base=BASE, wagon_frames=man,
                           wagon_states_root=ws, cache_root=cache,
                           verbose=False)["json_path"], encoding="utf-8") as fh:
            doc = json.load(fh)
        for wagon in doc["wagons"]:
            wf = wagon.get("wagon_frames", {})
            self.assertEqual(sorted(wf), ["left_up", "right_up"])
            self.assertNotIn("right_top", wf)
            self.assertNotIn("left_top", wf)

    def test_truncated_cache_drops_the_later_positions(self):
        """A camera whose cache stops half way through the wagon publishes only
        the positions that exist -- never a URL for a frame never written."""
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        br, st, un, ws, ev, cache = _fixture(tmp.name, cache_frac=0.4)
        man = WF.materialize(state=st, cache_root=cache, evidence_root=ev,
                             per_camera_meta=META, verbose=False)
        with open(CR.build(state=st, unified=un,
                           output_dir=os.path.join(br, "reports"),
                           batch_key=BATCH, evidence_root=ev,
                           evidence_url_base=BASE, wagon_frames=man,
                           wagon_states_root=ws, cache_root=cache,
                           verbose=False)["json_path"], encoding="utf-8") as fh:
            doc = json.load(fh)
        wf = doc["wagons"][0]["wagon_frames"]
        for angle, frames in wf.items():
            got = [f["position"] for f in frames]
            self.assertEqual(got, ["start", "mid1"], angle)
            for f in frames:
                rel = f["s3_url"].split("/evidence/", 1)[1]
                self.assertTrue(os.path.isfile(os.path.join(ev, *rel.split("/"))))

    def test_no_manifest_means_no_key(self):
        self.man = {}
        with open(self._report()["json_path"], encoding="utf-8") as fh:
            doc = json.load(fh)
        for wagon in doc["wagons"]:
            self.assertNotIn("wagon_frames", wagon)

    # 7. doors[].s3_url still works
    def test_door_urls_still_present(self):
        by_cam = {d["camera_id"]: d for d in self.wagon["doors"]}
        gw = self.wagon["global_id"]
        self.assertEqual(by_cam[C.CAMERA_LEFT_UP]["s3_url"],
                         "%s/%s/door/left_best.jpg" % (BASE, gw))
        self.assertEqual(by_cam[C.CAMERA_RIGHT_UP]["s3_url"],
                         "%s/%s/door/right_best.jpg" % (BASE, gw))

    # 8. evidence_pages unchanged
    def test_evidence_pages_still_relative(self):
        self.assertIn("evidence_pages", self.doc)
        for snaps in self.doc["evidence_pages"].values():
            for rel in snaps.values():
                self.assertFalse(rel.startswith("http"), rel)

    def test_batch_key_unchanged(self):
        self.assertEqual(self.doc["batch_key"], BATCH)

    def test_existing_wagon_fields_preserved(self):
        for k in ("global_id", "wagon_index", "classification", "left_door",
                  "right_door", "load_status", "top_damage", "doors"):
            self.assertIn(k, self.wagon)


if __name__ == "__main__":
    unittest.main()
