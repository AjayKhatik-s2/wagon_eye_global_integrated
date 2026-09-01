"""Sequential-mode dashboard delivery.

    global_assembly.assemble()   -> report carries doors/wagon_frames URLs
              |
    sequential.runner._deliver() -> S3 upload -> per-camera ingest -> global ingest

The ordering is the contract: nothing is POSTed until the object its URL names
has been uploaded. These tests pin that, the `reports_dir="combined"` bridge, and
the two independent skip flags.
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
from delivery import dashboard_ingest as DI
from sequential import evidence as ev
from sequential import runner as SR

BATCH = "20260724_072511"
BUCKET = "biputri-wagoneye-report"


# ---------------------------------------------------------------------------
# 1. sequential reports directory + reports_dir parameter
# ---------------------------------------------------------------------------

class ReportsDirBridge(unittest.TestCase):

    def test_sequential_layout_is_combined_and_unchanged(self):
        self.assertEqual(ev.COMBINED_DIRNAME, "combined")
        self.assertTrue(ev.combined_dir("/w").endswith("/w/combined"))

    def test_dashboard_ingest_defaults_to_batch_layout(self):
        self.assertEqual(DI.DEFAULT_REPORTS_DIR, "reports")
        import inspect
        self.assertEqual(
            inspect.signature(DI.run).parameters["reports_dir"].default,
            "reports")

    def test_reports_dir_is_honoured(self):
        """A report under combined/ is found only when the caller says so."""
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        os.makedirs(os.path.join(tmp.name, "combined"))
        with open(os.path.join(tmp.name, "combined",
                               "combined_train_report.json"), "w") as fh:
            json.dump({"batch_key": BATCH, "wagons": []}, fh)
        # default ("reports") cannot see it
        self.assertEqual(
            DI.run(batch_root=tmp.name, skip_upload=True).get("error"),
            "no_report")
        # naming combined/ does
        self.assertNotEqual(
            DI.run(batch_root=tmp.name, skip_upload=True,
                   reports_dir="combined").get("error"),
            "no_report")

    def test_the_runner_passes_combined(self):
        import inspect
        src = inspect.getsource(SR._deliver)
        self.assertIn("reports_dir = ev.COMBINED_DIRNAME", src)
        self.assertIn("reports_dir=reports_dir", src)


# ---------------------------------------------------------------------------
# 2. S3 URL construction -- must match upload_tree's own key layout
# ---------------------------------------------------------------------------

class UrlBase(unittest.TestCase):

    def _base(self, batch_key=BATCH):
        return ("https://%s.s3.%s.amazonaws.com/%s/%s/evidence"
                % (C.S3_OUTPUT_BUCKET, C.S3_REGION,
                   C.S3_TRAIN_BATCH_PREFIX, batch_key))

    def test_bucket_is_the_current_account(self):
        self.assertEqual(C.S3_OUTPUT_BUCKET, BUCKET)
        self.assertNotIn("biro", C.S3_OUTPUT_BUCKET)
        self.assertEqual(C.S3_REGION, "ap-south-1")

    def test_bucket_is_env_overridable(self):
        import importlib
        os.environ["WAGONEYE_S3_OUTPUT_BUCKET"] = "some-other-bucket"
        try:
            importlib.reload(C)
            self.assertEqual(C.S3_OUTPUT_BUCKET, "some-other-bucket")
        finally:
            del os.environ["WAGONEYE_S3_OUTPUT_BUCKET"]
            importlib.reload(C)
        self.assertEqual(C.S3_OUTPUT_BUCKET, BUCKET)

    def test_runner_base_mirrors_upload_tree(self):
        """The URL prefix and the upload key must be one layout."""
        import inspect
        from delivery import s3_upload
        up = " ".join(inspect.getsource(s3_upload.upload_tree).split())
        self.assertIn('base = f"{C.S3_TRAIN_BATCH_PREFIX}/{batch_key}"', up)
        run_src = " ".join(inspect.getsource(SR.run_sequential).split())
        self.assertIn("C.S3_TRAIN_BATCH_PREFIX, batch_key", run_src)
        self.assertIn("evidence", run_src)

    def test_no_base_when_not_uploading(self):
        import inspect
        src = inspect.getsource(SR.run_sequential)
        self.assertIn("None if skip_upload else", src)


# ---------------------------------------------------------------------------
# 3. wagon-frame materialization + URLs
# ---------------------------------------------------------------------------

class _W:
    def __init__(self, gw="GW_1", idx=1, st=0.0, et=4.0):
        self.global_id, self.wagon_index = gw, idx
        self.start_time, self.end_time = st, et

    def local_range(self, camera_id):
        return None


class WagonFrames(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.cache = os.path.join(self.tmp.name, "wagon_cache")
        self.ev = os.path.join(self.tmp.name, "evidence")
        os.makedirs(self.ev)
        self.w = _W()
        for cam in C.ALL_CAMERAS:
            d = os.path.join(self.cache, "GW_1", C.CAMERA_FOLDER[cam])
            os.makedirs(d, exist_ok=True)
            for i in range(0, 60):
                with open(os.path.join(d, "frame_%06d.jpg" % i), "wb") as fh:
                    fh.write(b"\xff\xd8\xff\xdb")

    def _state(self):
        class S:
            wagons = [self.w]
            def camera_time_offsets(inner):
                return {}
        return S()

    def test_four_angles_four_positions(self):
        man = WF.materialize(
            state=self._state(), cache_root=self.cache, evidence_root=self.ev,
            per_camera_meta={c: {"fps": 15.0, "total_frames": 60}
                             for c in C.ALL_CAMERAS}, verbose=False)
        self.assertEqual(sorted(man["GW_1"]),
                         sorted(["left_up", "right_up", "right_top", "left_top"]))
        for angle, frames in man["GW_1"].items():
            self.assertEqual([f["position"] for f in frames],
                             ["start", "mid1", "mid2", "end"], angle)

    def test_frames_land_under_evidence_so_stage6_uploads_them(self):
        man = WF.materialize(
            state=self._state(), cache_root=self.cache, evidence_root=self.ev,
            per_camera_meta={c: {"fps": 15.0, "total_frames": 60}
                             for c in C.ALL_CAMERAS}, verbose=False)
        for angle, frames in man["GW_1"].items():
            for f in frames:
                self.assertTrue(os.path.isfile(
                    os.path.join(self.ev, *f["rel_path"].split("/"))), f)
                self.assertTrue(f["rel_path"].startswith("GW_1/wagon_frames/"))

    def test_url_is_not_fabricated_without_a_base(self):
        man = WF.materialize(
            state=self._state(), cache_root=self.cache, evidence_root=self.ev,
            per_camera_meta={c: {"fps": 15.0, "total_frames": 60}
                             for c in C.ALL_CAMERAS}, verbose=False)
        self.assertEqual(WF.published(man, "GW_1", None), {})

    def test_no_cache_means_no_manifest(self):
        man = WF.materialize(
            state=self._state(), cache_root=None, evidence_root=self.ev,
            per_camera_meta={}, verbose=False)
        self.assertEqual(man, {})


# ---------------------------------------------------------------------------
# 4. assembly integration -- the manifest must exist BEFORE the report
# ---------------------------------------------------------------------------

class AssemblyIntegration(unittest.TestCase):

    def test_assemble_materializes_before_building(self):
        import inspect
        from sequential import global_assembly as GA
        src = inspect.getsource(GA.assemble)
        i_mat = src.index("wagon_frames.materialize(")
        i_build = src.index("combined_train_report.build(")
        self.assertLess(i_mat, i_build,
                        "frames must be materialized before the report is built")

    def test_assemble_forwards_both_to_build(self):
        import inspect
        from sequential import global_assembly as GA
        src = inspect.getsource(GA.assemble)
        self.assertIn("evidence_url_base=evidence_url_base", src)
        self.assertIn("wagon_frames=wagon_frame_manifest", src)

    def test_report_builder_accepts_them(self):
        import inspect
        from reporting import combined_train_report as CR
        sig = inspect.signature(CR.build).parameters
        self.assertIn("evidence_url_base", sig)
        self.assertIn("wagon_frames", sig)


# ---------------------------------------------------------------------------
# 5. delivery ordering + skip flags
# ---------------------------------------------------------------------------

class DeliveryOrdering(unittest.TestCase):

    def test_upload_precedes_both_ingests(self):
        import inspect
        src = inspect.getsource(SR._deliver)
        i_up = src.index("s3_upload.upload_tree(")
        i_cam = src.index("dashboard_ingest.run(")
        i_glob = src.index("global_train_webhook.publish(")
        self.assertLess(i_up, i_cam, "upload must precede per-camera ingest")
        self.assertLess(i_cam, i_glob, "per-camera must precede global")

    def test_skip_upload_suppresses_all_three(self):
        import inspect
        src = inspect.getsource(SR._deliver)
        head = src[:src.index("import boto3")]
        self.assertIn("if skip_upload:", head)
        self.assertIn("return", head)

    def test_skip_email_is_independent_of_ingest(self):
        """--skip-email must not disable the dashboard feed."""
        import inspect
        src = inspect.getsource(SR._deliver)
        self.assertIn("if not skip_email:", src)
        i_glob = src.index("global_train_webhook.publish(")
        i_mail = src.rindex("_send_email(")
        self.assertLess(i_glob, i_mail, "email comes after ingest, not instead")

    def test_hook_runs_after_assembly_and_only_when_ready(self):
        import inspect
        src = inspect.getsource(SR.run_sequential)
        i_asm = src.index("global_assembly.assemble(")
        i_del = src.index("_deliver(")
        self.assertLess(i_asm, i_del)
        self.assertIn("outcome.assembly.ready", src)

    def test_flags_are_threaded_from_the_cli(self):
        import inspect
        from orchestrator import master_runner as MR
        for fn in (MR.run_local, MR._run_sequential_local):
            p = inspect.signature(fn).parameters
            self.assertIn("skip_upload", p)
            self.assertIn("skip_email", p)
        src = inspect.getsource(MR.run_local)
        self.assertIn("skip_upload=skip_upload", src)
        self.assertNotIn("skip_upload=True, skip_email=True", src)

    def test_a_non_uploading_local_run_holds_no_live_client(self):
        import inspect
        src = inspect.getsource(SR.run_sequential)
        self.assertIn("None if skip_upload else", src)
        mr = inspect.getsource(
            __import__("orchestrator.master_runner", fromlist=["x"]).run_local)
        self.assertIn("_NoopS3() if skip_upload else", mr)


# ---------------------------------------------------------------------------
# 6. UAT only
# ---------------------------------------------------------------------------

class UatOnly(unittest.TestCase):

    def test_global_endpoint_is_uat(self):
        from delivery import global_train_webhook as W
        urls = W.global_ingest_urls()
        self.assertEqual(len(urls), 1)
        self.assertIn("cctv-wagon-uat-api", urls[0])
        self.assertTrue(urls[0].endswith("/inspections/ingest-global"))
        self.assertNotIn("ms-pnr-location-notification-api", urls[0])

    def test_per_camera_can_be_pinned_to_uat(self):
        os.environ["WAGONEYE_INSPECTION_INGEST_API_URLS"] = "uat"
        try:
            urls = DI.ingest_api_urls()
            self.assertEqual(len(urls), 1)
            self.assertIn("cctv-wagon-uat-api", urls[0])
        finally:
            del os.environ["WAGONEYE_INSPECTION_INGEST_API_URLS"]

    def test_global_camera_id(self):
        from delivery import global_train_webhook as W
        self.assertEqual(W.camera_id(), "GLOBAL_FUSED")


if __name__ == "__main__":
    unittest.main()
