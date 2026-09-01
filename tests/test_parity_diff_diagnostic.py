"""The parity diagnostic must be trustworthy before its verdict means anything.

`scripts/parity_diff.py` is what decides whether a real Batch-vs-Sequential EC2
run agreed. Three properties make its answer usable, and all three are tested
here:

  1. it reports the UPSTREAM-MOST divergence, not a downstream symptom -- a
     wagon-count difference is usually a gap-count difference, which is usually
     a master-camera difference, so printing the wagon count first would send
     the reader to the wrong component;
  2. it distinguishes DATA from METADATA. Two runs of the same train differ in
     timestamps, run ids, output paths and durations by construction. Those are
     excused BY FIELD NAME -- never because their values happen to differ -- and
     every excused field is printed, so nothing is hidden by the policy;
  3. `--strict-metadata` removes the excuses, for when you want everything.

    python -m pytest tests/test_parity_diff_diagnostic.py -q
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import sys

import pytest

_TEST_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_TEST_DIR)
for _path in (_REPO_ROOT, os.path.join(_REPO_ROOT, "scripts")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import parity_diff


def _state(master="RIGHT_UP", wagons=4, generated="2026-08-31T10:00:00",
           batch_key="run-a", seconds=123.4):
    return {
        "global_counting_engine": "global_wagon_app",
        "master_camera": master,
        "generated_at": generated,
        "batch_key": batch_key,
        "processing_seconds": seconds,
        "global_gaps": [{"global_gap_id": "G%d" % index,
                         "normalized_position": index * 100.0}
                        for index in range(1, wagons + 2)],
        "wagons": [{"global_id": "GW_%d" % index, "wagon_index": index,
                    "classification": "WAGON",
                    "classification_confidence": 0.9,
                    "start_frame_master": index * 100,
                    "end_frame_master": index * 100 + 90,
                    "camera_frame_ranges": {}}
                   for index in range(1, wagons + 1)],
    }


def _write(root, name, document):
    directory = os.path.join(str(root), name, "global_state")
    os.makedirs(directory, exist_ok=True)
    with open(os.path.join(directory, "global_train_state.json"), "w",
              encoding="utf-8") as handle:
        json.dump(document, handle)
    return os.path.join(str(root), name)


def _run(batch_dir, sequential_dir, *extra):
    buffer = io.StringIO()
    argv = ["--batch", batch_dir, "--sequential", sequential_dir] + list(extra)
    with contextlib.redirect_stdout(buffer):
        code = parity_diff.main(argv)
    return code, buffer.getvalue()


def _first_field(output):
    for line in output.splitlines():
        if line.strip().startswith("field"):
            return line.split(":", 1)[1].strip()
    return None


def test_identical_runs_pass_cleanly(tmp_path):
    batch = _write(tmp_path, "b", _state())
    sequential = _write(tmp_path, "s", _state())
    code, output = _run(batch, sequential)
    assert code == 0, output
    assert "PARITY OK" in output


def test_metadata_only_differences_pass_and_are_reported(tmp_path):
    """Excused, but never silently: the operator sees each one and why."""
    batch = _write(tmp_path, "b", _state())
    sequential = _write(tmp_path, "s", _state(
        generated="2026-09-01T22:41:00", batch_key="run-b", seconds=71.2))

    code, output = _run(batch, sequential)
    assert code == 0, output
    assert "EXPECTED METADATA DIFFERENCES" in output
    for field in ("generated_at", "batch_key", "processing_seconds"):
        assert field in output, "%s was not reported as excused" % field
    assert "PARITY OK" in output


def test_strict_metadata_turns_provenance_into_a_divergence(tmp_path):
    batch = _write(tmp_path, "b", _state())
    sequential = _write(tmp_path, "s", _state(batch_key="run-b"))

    assert _run(batch, sequential)[0] == 0
    code, output = _run(batch, sequential, "--strict-metadata")
    assert code == 1, output


def test_metadata_is_excused_by_name_not_by_value():
    """A data field is never excused just because its values disagree."""
    assert parity_diff.metadata_reason("generated_at")
    assert parity_diff.metadata_reason("wagons[0].sealed_at")
    assert parity_diff.metadata_reason("master_camera") is None
    assert parity_diff.metadata_reason("global_gap_count") is None
    assert parity_diff.metadata_reason("total_wagons") is None
    assert parity_diff.metadata_reason("classification") is None


def test_master_camera_difference_is_reported(tmp_path):
    batch = _write(tmp_path, "b", _state(master="RIGHT_UP"))
    sequential = _write(tmp_path, "s", _state(master="LEFT_UP_TOP"))
    code, output = _run(batch, sequential)
    assert code == 1
    assert _first_field(output) == "master_camera"
    assert "select_master_camera" in output, (
        "the report must name the component that produced the field")


def test_the_cause_is_reported_before_its_downstream_symptom(tmp_path):
    """Master AND wagon count differ: the master must be printed first."""
    batch = _write(tmp_path, "b", _state(master="RIGHT_UP", wagons=4))
    sequential = _write(tmp_path, "s", _state(master="LEFT_UP", wagons=7))
    code, output = _run(batch, sequential)
    assert code == 1
    assert _first_field(output) == "master_camera", (
        "a downstream symptom was reported before its cause")


def test_a_pure_downstream_difference_is_still_caught(tmp_path):
    """With the same master, a roster difference must still fail the run."""
    batch = _write(tmp_path, "b", _state(wagons=4))
    sequential = _write(tmp_path, "s", _state(wagons=7))
    code, output = _run(batch, sequential)
    assert code == 1
    field = _first_field(output)
    assert field and "master_camera" not in field
    assert "gap" in field or "wagon" in field, field


def test_a_missing_state_document_is_not_comparable(tmp_path):
    batch = _write(tmp_path, "b", _state())
    empty = os.path.join(str(tmp_path), "empty")
    os.makedirs(empty, exist_ok=True)
    with pytest.raises(SystemExit):
        _run(batch, empty)
