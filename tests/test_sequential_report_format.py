"""Sequential's Phase-2 reports use Batch's builders with Batch's arguments.

`logo_path` defaults to None and the renderer honours that silently, so omitting
it produced a visibly different PDF from the SAME builder with no error anywhere
-- the one class of parity break the field-level comparator cannot see, because
parity_diff never opens a PDF.

Also pins that the Phase-1 camera-local report makes no GLOBAL claim. It used to
label RIGHT_UP the "canonical gap authority" from the static `C.MASTER_CAMERA`
constant, but the master is whichever camera has the most confirmed unique gaps
-- decided in Phase 2, and on real footage often not RIGHT_UP.
"""

from __future__ import annotations

import inspect
import os

from core import constants as C
from sequential import camera_report, global_assembly


# -----------------------------------------------------------------------------
# Phase 2: same builder, same arguments as Batch
# -----------------------------------------------------------------------------

def _batch_call(fn_name):
    """The argument names Batch passes to `fn_name`."""
    from orchestrator import master_runner
    src = inspect.getsource(master_runner.process_batch)
    start = src.index(fn_name + "(")
    depth, i = 0, start + len(fn_name)
    while True:
        if src[i] == "(":
            depth += 1
        elif src[i] == ")":
            depth -= 1
            if depth == 0:
                break
        i += 1
    import re
    return set(re.findall(r"(\w+)\s*=", src[start:i]))


def _seq_call(fn_name):
    src = inspect.getsource(global_assembly.assemble)
    start = src.index(fn_name + "(")
    depth, i = 0, start + len(fn_name)
    while True:
        if src[i] == "(":
            depth += 1
        elif src[i] == ")":
            depth -= 1
            if depth == 0:
                break
        i += 1
    import re
    return set(re.findall(r"(\w+)\s*=", src[start:i]))


def test_camera_reports_gets_logo_path_like_batch():
    seq = _seq_call("camera_reports.build_all")
    assert "logo_path" in seq, \
        "Sequential omits logo_path -> its camera PDFs render without the logo"


def test_combined_report_gets_logo_path_and_missing_cameras_like_batch():
    seq = _seq_call("combined_train_report.build")
    for arg in ("logo_path", "missing_cameras"):
        assert arg in seq, "Sequential omits %s" % arg


def test_sequential_passes_every_report_argument_batch_passes():
    """A future argument added to Batch's call must not be silently dropped.

    Excludes arguments Sequential legitimately cannot supply: it renders no
    annotated videos, so processed_video_urls is empty by design.
    """
    legitimate_gaps = {"processed_video_urls"}
    for fn in ("camera_reports.build_all", "combined_train_report.build"):
        batch = _batch_call(fn.split(".")[-1] if "." not in fn else fn)
        seq = _seq_call(fn)
        missing = (batch - seq) - legitimate_gaps
        assert not missing, "%s: Sequential drops %s" % (fn, sorted(missing))


def test_the_logo_resolves_to_a_real_file():
    """A wrong path would silently fall back to None -- the original bug."""
    from orchestrator import master_runner
    repo_root = os.path.dirname(os.path.dirname(
        os.path.abspath(master_runner.__file__)))
    logo = os.path.join(repo_root, "reporting", "assets", "Logo.jpeg")
    assert os.path.isfile(logo), "the logo Batch uses is not at %s" % logo


def test_sequential_does_not_reimplement_the_canonical_camera_report():
    """Phase 2 must call Batch's renderer, not a Sequential copy."""
    src = inspect.getsource(global_assembly.assemble)
    assert "camera_reports.build_all" in src
    assert "combined_train_report.build" in src


# -----------------------------------------------------------------------------
# Phase 1: camera-local, and no global claims
# -----------------------------------------------------------------------------

def _code_only(fn):
    """Source with comments stripped.

    The comments explaining a removal quote the very strings the removal was
    about, so asserting against raw source matches the explanation instead of
    the code and the test can never fail.
    """
    import re
    src = inspect.getsource(fn)
    return re.sub(r"^\s*#.*$", "", src, flags=re.M)


def test_phase1_report_is_marked_non_canonical():
    src = _code_only(camera_report.build_document)
    assert '"report_type": "single_camera"' in src
    assert '"canonical": False' in src


def test_phase1_report_names_no_master_camera():
    """The master is selected in Phase 2 from all four sealed evidences."""
    src = _code_only(camera_report.build_document)
    assert "C.MASTER_CAMERA" not in src, \
        "a static master-camera constant is a GLOBAL claim in a local report"
    assert "canonical gap authority (RIGHT_UP)" not in src


def test_phase1_report_does_not_demote_the_other_cameras():
    """In Phase 1 all four cameras are equal and independent."""
    src = _code_only(camera_report.build_document)
    assert "corroborating evidence only" not in src


def test_phase1_report_still_refuses_canonical_ids():
    """The existing GW_n guard must stay -- it is the stronger invariant."""
    src = _code_only(camera_report.build_document)
    assert "assert_no_canonical_ids" in src


def test_the_comment_stripper_actually_strips():
    """Guard the guard: a broken stripper makes three tests vacuous."""
    def _sample():
        # C.MASTER_CAMERA appears only in this comment
        return 1
    assert "C.MASTER_CAMERA" not in _code_only(_sample)
    assert "return 1" in _code_only(_sample)
