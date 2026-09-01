"""`wagons[].problem_frames` -- one entry per damage snapshot, in the GLOBAL report.

The snapshots are stored camera-AMBIGUOUSLY on disk (`track_1.jpg`, not
`track_1__RIGHT_UP_TOP.jpg`). `evidence/<GW>/damage/metadata.json`'s
`tracks[].camera_id` is the authority for who took each one, and these tests pin
that: the two top cameras photograph the same roof, so an assumed owner shows
one camera's damage under the other's name.
"""

import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core import constants as C
from reporting import combined_train_report as CR

BATCH = "20260724_072511"
BUCKET = "biputri-wagoneye-report"
BASE = ("https://%s.s3.ap-south-1.amazonaws.com/train_batch/%s/evidence"
        % (BUCKET, BATCH))
RT, LT = C.CAMERA_RIGHT_UP_TOP, C.CAMERA_LEFT_UP_TOP

TRACKS = [
    {"track_idx": 1, "camera_id": RT, "track_id": 1, "class_name": "floor_damage",
     "confidence": 0.78, "best_confidence": 0.81, "best_frame_idx": 2971,
     "bbox": [520.0, 214.0, 812.0, 498.0]},
    {"track_idx": 2, "camera_id": LT, "track_id": 2,
     "class_name": "inner_wall_damage", "confidence": 0.66,
     "best_confidence": 0.72, "best_frame_idx": 3004,
     "bbox": [180.0, 260.0, 430.0, 540.0]},
]


class _Tree:
    """An evidence tree with damage, as features/damage/processor.py writes it."""

    def __init__(self, tracks=TRACKS, *, write_jpegs=(1, 2), metadata=True):
        self.tmp = tempfile.TemporaryDirectory()
        self.ev = os.path.join(self.tmp.name, "evidence")
        d = os.path.join(self.ev, "GW_2", "damage")
        os.makedirs(d, exist_ok=True)
        for i in write_jpegs:
            with open(os.path.join(d, "track_%d.jpg" % i), "wb") as fh:
                fh.write(b"\xff\xd8\xff\xdb")
        if metadata:
            with open(os.path.join(d, "metadata.json"), "w") as fh:
                json.dump({"global_id": "GW_2", "feature": "damage",
                           "top_damage": C.DAMAGE_PRESENT, "tracks": tracks}, fh)

    def frames(self, base=BASE, wagon_index=2):
        return CR._damage_problem_frames("GW_2", wagon_index, self.ev, base)

    def close(self):
        self.tmp.cleanup()


class ProblemFrames(unittest.TestCase):

    def setUp(self):
        self.t = _Tree()
        self.addCleanup(self.t.close)
        self.frames = self.t.frames()

    # -- one entry per snapshot -------------------------------------------

    def test_one_entry_per_track(self):
        self.assertEqual(len(self.frames), 2)
        self.assertEqual([f["filename"] for f in self.frames],
                         ["track_1.jpg", "track_2.jpg"])

    # -- the owning camera comes from metadata, not a guess ---------------

    def test_camera_id_read_from_metadata(self):
        self.assertEqual(self.frames[0]["camera_id"], RT)
        self.assertEqual(self.frames[1]["camera_id"], LT)

    def test_the_two_top_cameras_are_not_conflated(self):
        cams = {f["camera_id"] for f in self.frames}
        self.assertEqual(cams, {RT, LT})

    def test_reordered_metadata_still_pairs_correctly(self):
        """Pairing is by track_idx, not list position."""
        t = _Tree(tracks=list(reversed(TRACKS)))
        self.addCleanup(t.close)
        by_file = {f["filename"]: f for f in t.frames()}
        self.assertEqual(by_file["track_1.jpg"]["camera_id"], RT)
        self.assertEqual(by_file["track_2.jpg"]["camera_id"], LT)

    # -- URL ---------------------------------------------------------------

    def test_url_is_the_uploaded_evidence_path(self):
        self.assertEqual(
            self.frames[0]["s3_url"],
            "https://%s.s3.ap-south-1.amazonaws.com/%s/%s/evidence/"
            "GW_2/damage/track_1.jpg"
            % (BUCKET, C.S3_TRAIN_BATCH_PREFIX, BATCH))

    def test_annotated_url_matches_s3_url(self):
        for f in self.frames:
            self.assertEqual(f["annotated_image_url"], f["s3_url"])
            self.assertTrue(f["is_annotated"])

    def test_not_the_backend_sample_bucket(self):
        for f in self.frames:
            self.assertNotIn("test-inspection-artifacts-sarva", f["s3_url"])
            self.assertIn(BUCKET, f["s3_url"])

    # -- shape parity with the per-camera documents ------------------------

    def test_shape_mirrors_the_per_camera_problem_frame(self):
        from delivery import inspection_json as IJ
        per_camera = IJ._problem_frame(
            wagon_count=2, segment_type="wagon", segment_number=2,
            problem_type="floor_dmg", frame_number=2971,
            url="https://x/y.jpg", bbox=[1, 2, 3, 4], extra={},
            confidence=0.81, class_name="floor_damage")
        shared = {"wagon_count", "segment_type", "problem_type", "frame_number",
                  "filename", "s3_url", "is_annotated", "annotated_image_url",
                  "bounding_box"}
        self.assertTrue(shared <= set(per_camera))
        self.assertTrue(shared <= set(self.frames[0]))

    def test_global_only_fields_present(self):
        """The global report must say WHICH camera and WHICH wagon."""
        for f in self.frames:
            self.assertEqual(f["global_id"], "GW_2")
            self.assertEqual(f["wagon_count"], 2)
            self.assertIn("camera_id", f)

    def test_problem_type_uses_the_dashboard_vocabulary(self):
        self.assertEqual(self.frames[0]["problem_type"], "floor_dmg")
        self.assertEqual(self.frames[1]["problem_type"], "inner_wall_dmg")

    def test_bounding_box_carries_coords_confidence_class(self):
        bb = self.frames[0]["bounding_box"]
        self.assertEqual(bb["coordinates"], [520.0, 214.0, 812.0, 498.0])
        self.assertEqual(bb["confidence"], 0.81)
        self.assertEqual(bb["class_name"], "floor_damage")

    # -- never a fabricated URL -------------------------------------------

    def test_missing_jpeg_is_skipped(self):
        t = _Tree(write_jpegs=(1,))          # track_2.jpg absent
        self.addCleanup(t.close)
        got = t.frames()
        self.assertEqual([f["filename"] for f in got], ["track_1.jpg"])

    def test_no_metadata_yields_nothing(self):
        t = _Tree(metadata=False)
        self.addCleanup(t.close)
        self.assertEqual(t.frames(), [])

    def test_corrupt_metadata_yields_nothing(self):
        t = _Tree()
        self.addCleanup(t.close)
        with open(os.path.join(t.ev, "GW_2", "damage", "metadata.json"), "w") as fh:
            fh.write("{ broken")
        self.assertEqual(t.frames(), [])

    def test_unparseable_track_idx_is_skipped(self):
        t = _Tree(tracks=[dict(TRACKS[0], track_idx=None), TRACKS[1]])
        self.addCleanup(t.close)
        self.assertEqual([f["filename"] for f in t.frames()], ["track_2.jpg"])

    def test_no_url_base_yields_nothing(self):
        self.assertEqual(self.t.frames(base=None), [])

    def test_no_evidence_root_yields_nothing(self):
        self.assertEqual(
            CR._damage_problem_frames("GW_2", 2, None, BASE), [])

    def test_a_clean_wagon_yields_nothing(self):
        """GW_1 has no damage folder at all."""
        self.assertEqual(
            CR._damage_problem_frames("GW_1", 1, self.t.ev, BASE), [])


if __name__ == "__main__":
    unittest.main()
