"""What the CLI says is what Sequential uses -- models and features.

Regression for the first real EC2 Sequential run, which reported

    reconstruction models resolved to <repo>/models/reconstruction/
    [SEQ] features : door, ocr, load, damage

while the command supplied `--recon-models-dir ~/global_wagon_models` and
`--features door,load,damage`. Both symptoms are exactly what a run that OMITS
those flags produces, and neither could be reproduced from the CLI. These tests
pin the whole chain -- argv -> argparse -> main -> run_local -> run_sequential
-> process_camera -> resolve_models -- so if the value ever stops arriving, a
test fails here instead of a real train being processed with the wrong weights.

The model test does not merely check the happy path: it records every filesystem
probe and asserts the repository's `models/reconstruction` is never even looked
at when a directory is supplied.

    python -m pytest tests/test_sequential_cli_resolution.py -q
"""

from __future__ import annotations

import os
import subprocess
import sys

import pytest

_TEST_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_TEST_DIR)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from core import constants as C
from core.feature_config import FEATURE_KEYS
from global_counting import runner as gc_runner
from orchestrator import master_runner as mr
from sequential import camera_runner
from sequential import runner as sequential_runner

# The five weights the engine requires, in their EC2 spellings.
EC2_WEIGHTS = ("side_classification.pt", "top_classification.pt",
               "right_up_gap.pt", "left_up_gap.pt", "top_gap.pt")

REPO_RECON_DIR = os.path.join(_REPO_ROOT, "models", "reconstruction")


@pytest.fixture
def weights_dir(tmp_path):
    """A stand-in for ~/global_wagon_models, with the exact EC2 filenames."""
    directory = tmp_path / "global_wagon_models"
    directory.mkdir()
    for name in EC2_WEIGHTS:
        (directory / name).write_bytes(b"weights")
    return str(directory)


@pytest.fixture
def videos_dir(tmp_path):
    directory = tmp_path / "fresh_train"
    directory.mkdir()
    for camera in C.ALL_CAMERAS:
        (directory / ("%s.mp4" % camera)).write_bytes(b"video")
    return str(directory)


# =============================================================================
# 1. the five weights come from the supplied directory
# =============================================================================

def test_all_five_ec2_weights_resolve_from_the_supplied_directory(weights_dir):
    resolved = sequential_runner.resolve_reconstruction_models(
        weights_dir, _REPO_ROOT)

    assert set(resolved) == {"classification_side", "classification_top",
                             "gap_right", "gap_left", "gap_top"}
    assert sorted(os.path.basename(p) for p in resolved.values()) == \
        sorted(EC2_WEIGHTS)
    for path in resolved.values():
        assert os.path.dirname(path) == os.path.abspath(weights_dir)


def test_resolution_never_touches_the_repository_models_directory(weights_dir,
                                                                  monkeypatch):
    """The literal requirement: the repo's models/reconstruction is not searched."""
    probed = []
    real_isfile = os.path.isfile

    def _recording_isfile(path):
        probed.append(str(path))
        return real_isfile(path)

    monkeypatch.setattr(os.path, "isfile", _recording_isfile)
    sequential_runner.resolve_reconstruction_models(weights_dir, _REPO_ROOT)

    offenders = [path for path in probed
                 if os.path.join("models", "reconstruction") in path
                 or "models/reconstruction" in path]
    assert offenders == [], (
        "Sequential probed the repository model directory: %s" % offenders[:5])
    assert any(os.path.abspath(weights_dir) in path for path in probed), (
        "the supplied directory was never probed at all")


def test_tilde_and_env_vars_in_the_supplied_path_are_expanded(tmp_path,
                                                              monkeypatch):
    """An unexpanded ~ must not send resolution to the repo default."""
    directory = tmp_path / "models_home"
    directory.mkdir()
    for name in EC2_WEIGHTS:
        (directory / name).write_bytes(b"w")

    monkeypatch.setenv("WAGONEYE_TEST_WEIGHTS", str(directory))
    resolved = sequential_runner.resolve_reconstruction_models(
        "$WAGONEYE_TEST_WEIGHTS", _REPO_ROOT)
    assert len(resolved) == 5
    for path in resolved.values():
        assert os.path.dirname(path) == os.path.abspath(str(directory))


def test_sequential_uses_the_same_resolver_as_batch_stage1():
    """One contract, not two: the same function, on the same CLI value."""
    import inspect

    assert "gc_runner.resolve_models" in inspect.getsource(
        sequential_runner.resolve_reconstruction_models)
    assert "gc_runner.resolve_models" in inspect.getsource(
        camera_runner._model_fingerprints)
    assert "gc_runner.resolve_models" in inspect.getsource(
        camera_runner.process_camera)

    # ...and no Sequential module invents its own search path.
    for relative in ("sequential/runner.py", "sequential/camera_runner.py",
                     "sequential/global_assembly.py"):
        with open(os.path.join(_REPO_ROOT, relative), encoding="utf-8") as handle:
            source = handle.read()
        assert "MODEL_SLOTS" not in source, (
            "%s re-implements model slot resolution" % relative)
        assert "DEFAULT_RECON_MODELS_DIR" not in source, (
            "%s could override the CLI value with a default" % relative)


def test_missing_weights_error_names_the_flag_and_the_directory(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(camera_runner.CameraRunError) as excinfo:
        sequential_runner.resolve_reconstruction_models(str(empty), _REPO_ROOT)
    message = str(excinfo.value)
    assert str(empty) in message, "the searched directory must be named"
    assert "FIVE weights" in message
    # The "you forgot the flag" hint belongs ONLY to the in-repo default; for an
    # explicitly supplied directory it would be actively misleading.
    assert "no --recon-models-dir was supplied" not in message


def test_repo_default_failure_says_the_flag_was_missing():
    """The exact confusion from the EC2 log, turned into a self-explaining error."""
    if os.path.isdir(REPO_RECON_DIR) and any(
            name.endswith(".pt") for name in os.listdir(REPO_RECON_DIR)):
        pytest.skip("weights are present in the repo default directory")

    with pytest.raises(camera_runner.CameraRunError) as excinfo:
        sequential_runner.resolve_reconstruction_models(
            REPO_RECON_DIR, _REPO_ROOT)
    message = str(excinfo.value)
    assert "no --recon-models-dir was supplied" in message
    assert "--recon-models-dir ~/global_wagon_models" in message


# =============================================================================
# 2. the CLI value actually reaches run_sequential
# =============================================================================

def _capture_run_sequential(monkeypatch):
    seen = {}

    def _fake(**kwargs):
        seen.update(kwargs)

        class _Outcome:
            cameras: list = []
            assembly = None
            sealed_cameras = ["captured"]
            failed_cameras: list = []
        return _Outcome()

    monkeypatch.setattr(sequential_runner, "run_sequential", _fake)
    return seen


def test_cli_recon_models_dir_reaches_run_sequential(monkeypatch, weights_dir,
                                                    videos_dir, tmp_path):
    """argv -> argparse -> main -> run_local -> run_sequential, end to end."""
    seen = _capture_run_sequential(monkeypatch)

    code = mr.main([
        "--local-only", "--local-inputs", videos_dir,
        "--mode", "sequential", "--no-interactive", "--skip-assembly",
        "--recon-models-dir", weights_dir,
        "--features", "door,load,damage",
        "--workspace", str(tmp_path / "ws"),
    ])
    assert code == 0
    assert seen["recon_models_dir"] == weights_dir, (
        "the CLI directory did not reach Sequential; got %r"
        % seen.get("recon_models_dir"))
    assert os.path.abspath(seen["recon_models_dir"]) != \
        os.path.abspath(REPO_RECON_DIR)


def test_cli_features_reach_run_sequential_without_ocr(monkeypatch, weights_dir,
                                                       videos_dir, tmp_path):
    seen = _capture_run_sequential(monkeypatch)

    code = mr.main([
        "--local-only", "--local-inputs", videos_dir,
        "--mode", "sequential", "--no-interactive", "--skip-assembly",
        "--recon-models-dir", weights_dir,
        "--features", "door,load,damage",
        "--workspace", str(tmp_path / "ws"),
    ])
    assert code == 0
    assert list(seen["features"]) == ["door", "load", "damage"], (
        "Sequential received %r; OCR must not be added implicitly"
        % (list(seen["features"]),))
    assert "ocr" not in seen["features"]


def test_the_exact_production_command_forwards_both_values(monkeypatch,
                                                           weights_dir,
                                                           videos_dir,
                                                           tmp_path):
    """The command this repository is expected to run, verbatim."""
    seen = _capture_run_sequential(monkeypatch)

    code = mr.main([
        "--local-only",
        "--local-inputs", videos_dir,
        "--mode", "sequential",
        "--no-interactive",
        "--skip-assembly",
        "--recon-models-dir", weights_dir,
        "--features", "door,load,damage",
        "--workspace", str(tmp_path / "ws"),
    ])
    assert code == 0
    assert seen["recon_models_dir"] == weights_dir
    assert list(seen["features"]) == ["door", "load", "damage"]
    assert seen["skip_assembly"] is True
    assert seen["door_stride"] == 3
    assert seen["damage_stride"] == 3
    assert seen["load_stride"] == 2


# =============================================================================
# 3. feature resolution: exactly what was asked for
# =============================================================================

def test_door_load_damage_resolves_to_exactly_three_features():
    selected = mr.parse_features("door,load,damage")
    assert set(selected) == {"door", "load", "damage"}
    assert "ocr" not in selected

    config = mr.feature_config_from_selection(selected)
    assert config.enabled_keys() == ["door", "load", "damage"]
    assert config.disabled_keys() == ["ocr"]
    assert config.is_enabled("ocr") is False


def test_explicitly_requesting_ocr_still_enables_it():
    for spec in ("door,load,damage,ocr", "ocr", "all"):
        config = mr.feature_config_from_selection(mr.parse_features(spec))
        assert config.is_enabled("ocr") is True, spec
        assert "ocr" in config.enabled_keys()


def test_feature_resolution_reuses_the_batch_contract():
    """One parser, not two: Sequential has no feature parser of its own."""
    import inspect

    # The selection is built from Batch's own FeatureConfig + FEATURE_KEYS.
    source = inspect.getsource(mr.feature_config_from_selection)
    assert "FeatureConfig.from_disabled" in source
    assert "FEATURE_KEYS" in source

    for relative in ("sequential/runner.py", "sequential/camera_runner.py",
                     "sequential/global_assembly.py"):
        with open(os.path.join(_REPO_ROOT, relative), encoding="utf-8") as handle:
            text = handle.read()
        assert "def parse_features" not in text, (
            "%s defines a second feature parser" % relative)
        assert "FEATURES_ALL_KEYWORD" not in text


def test_sequential_never_widens_the_selection():
    """The selection reaching Global Assembly is exactly what was asked for.

    MIGRATED: `camera_runner.features_for_camera` no longer exists. Per-camera
    narrowing was part of camera-local feature inference, which moved to Global
    Assembly so Batch's own processors and wagon cache do the work. The
    selection is now carried whole from the CLI to `_run_features`, so that is
    where "never widened" is checked.
    """
    import inspect

    from sequential import global_assembly

    selected = mr.parse_features("door,load,damage")
    assert set(selected) == {"door", "load", "damage"}
    assert "ocr" not in selected

    # `_run_features` iterates Batch's four features and writes a DISABLED
    # sentinel for anything not selected -- it can never add one.
    source = inspect.getsource(global_assembly._run_features)
    assert 'for name in ("load", "door", "ocr", "damage")' in source
    assert "if name not in selected:" in source
    assert "_mark_disabled(name)" in source


def test_ocr_weight_is_not_even_fingerprinted_when_unselected(weights_dir,
                                                              tmp_path):
    """A disabled feature must not require or touch its model file."""
    feat_dir = tmp_path / "features"
    feat_dir.mkdir()
    for name in ("door_state.pt", "loaded.pt", "damage.pt"):
        (feat_dir / name).write_bytes(b"w")
    # deliberately NO wagon_id_counting.pt

    fingerprints = camera_runner._model_fingerprints(
        C.CAMERA_RIGHT_UP, recon_models_dir=weights_dir,
        feat_models_dir=str(feat_dir),
        features=("door", "load", "damage"))

    assert not any(key.endswith("_ocr") for key in fingerprints)
    assert "feature_door" in fingerprints

    # MIGRATED: all FIVE counting weights are fingerprinted now, not the two
    # for this camera. The engine's load_all_models validates the whole camera
    # mapping and build_class_maps reads every loaded model, so the result
    # depends on all five -- and a seal that ignored three of them would be
    # reused after one of those weights changed.
    for slot in sorted(gc_runner.MODEL_SLOTS):
        assert fingerprints[slot]["present"] is True, slot
        assert os.path.dirname(fingerprints[slot]["path"]) == \
            os.path.abspath(weights_dir), slot


def test_unselected_ocr_is_never_imported_in_a_fresh_process():
    """Process-isolated, so an earlier test's import cannot mask the answer."""
    code = (
        "import sys; sys.path.insert(0, %r);"
        "from orchestrator import master_runner as mr;"
        "from sequential import camera_runner;"
        "cfg = mr.feature_config_from_selection("
        "mr.parse_features('door,load,damage'));"
        "print('ENABLED=' + ','.join(cfg.enabled_keys()));"
        "print('OCR_MODULE=' + str('features.ocr.processor' in sys.modules));"
        "print('EASYOCR=' + str('easyocr' in sys.modules))" % _REPO_ROOT
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True,
                         text=True, cwd=_REPO_ROOT)
    assert out.returncode == 0, out.stderr
    assert "ENABLED=door,load,damage" in out.stdout, out.stdout
    assert "OCR_MODULE=False" in out.stdout, out.stdout
    assert "EASYOCR=False" in out.stdout, out.stdout


def test_sequential_log_states_the_resolved_features_and_directory(
        monkeypatch, weights_dir, videos_dir, tmp_path, capsys):
    """The log must make a defaulted run obvious instead of ambiguous."""
    monkeypatch.setattr(camera_runner, "process_camera",
                        lambda **kw: camera_runner.CameraRunResult(
                            camera_id=kw["camera_id"], status="SEALED"))

    sequential_runner.run_sequential(
        video_paths={C.CAMERA_RIGHT_UP: os.path.join(videos_dir,
                                                     "RIGHT_UP.mp4")},
        workspace=str(tmp_path / "ws"), repo_root=_REPO_ROOT,
        recon_models_dir=weights_dir, feat_models_dir=str(tmp_path),
        features=("door", "load", "damage"), batch_key="k",
        skip_assembly=True, verbose=True)

    output = capsys.readouterr().out
    assert "[SEQ] features        : door, load, damage" in output
    assert "[SEQ] features OFF    : ocr" in output
    assert "OCR             : DISABLED" in output
    assert os.path.abspath(weights_dir) in output
    # every resolved weight is named in the log
    for name in EC2_WEIGHTS:
        assert name in output, "%s missing from the log" % name
