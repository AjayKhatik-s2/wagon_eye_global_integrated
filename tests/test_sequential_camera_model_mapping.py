"""Sequential must satisfy the engine's model-registry contract, per camera.

Regression for the real EC2 failure:

    global_count_ec2/models.py::load_all_models
    RuntimeError: Classification model(s) required by the camera mapping are
                  not loaded: ['side']

Root cause, read from the engine rather than guessed: `load_all_models` checks
the WHOLE camera mapping, not the camera in hand --

    absent = [k for k in sorted(set(CAMERA_CLASSIFICATION_MODEL.values()))
              if k not in CLASSIFICATION_MODELS]          # needs side AND top
    absent = [k for k in sorted(set(CAMERA_GAP_MODEL.values()))
              if k not in GAP_MODELS]                     # needs left/right/top

-- so it is all-or-nothing. Sequential passed only the current camera's pair;
for LEFT_UP_TOP that was `{"top": ...}`, so `side` was absent. The KEY was
correct (`left_up_top` really does use `top` classification); the SET was not.

These tests drive the REAL `load_all_models` with fake loaders, so the engine's
own validation runs. The negative controls prove the tests would have caught the
regression: passing one camera's pair still raises, and a camera-id-keyed
registry still raises.

    python -m pytest tests/test_sequential_camera_model_mapping.py -q
"""

from __future__ import annotations

import os
import sys

import pytest

_TEST_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_TEST_DIR)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from core import constants as C
from global_counting import runner as gc_runner
from sequential import camera_runner


def _engine_dir():
    try:
        return gc_runner.locate_engine(_REPO_ROOT)
    except gc_runner.GlobalCountingError:
        return None


ENGINE_DIR = _engine_dir()
requires_engine = pytest.mark.skipif(
    ENGINE_DIR is None,
    reason="global_wagon_app engine not installed (set GLOBAL_WAGON_APP_DIR)")

# The five weights, exactly as they are named under ~/global_wagon_models.
WEIGHT_FILES = {
    "classification_side": "side_classification.pt",
    "classification_top": "top_classification.pt",
    "gap_right": "right_up_gap.pt",
    "gap_left": "left_up_gap.pt",
    "gap_top": "top_gap.pt",
}

# The mapping this test asserts, taken from the engine's camera_map and pinned
# here so a silent change to either side fails loudly.
EXPECTED_CLASSIFICATION = {
    C.CAMERA_RIGHT_UP:     "side",
    C.CAMERA_LEFT_UP:      "side",
    C.CAMERA_RIGHT_UP_TOP: "top",
    C.CAMERA_LEFT_UP_TOP:  "top",
}
EXPECTED_GAP = {
    C.CAMERA_RIGHT_UP:     "right",
    C.CAMERA_LEFT_UP:      "left",
    C.CAMERA_RIGHT_UP_TOP: "top",
    C.CAMERA_LEFT_UP_TOP:  "top",
}


@pytest.fixture
def weights(tmp_path):
    """Real (tiny) files, because the engine calls `.stat()` on every path."""
    directory = tmp_path / "global_wagon_models"
    directory.mkdir()
    for name in WEIGHT_FILES.values():
        (directory / name).write_bytes(b"weights")
    return gc_runner.resolve_models(str(directory))


# -----------------------------------------------------------------------------
# fake loaders, so the REAL load_all_models can run without weights
# -----------------------------------------------------------------------------

class _FakeGapModel:
    task = "detect"
    names = {0: "gap"}

    def predict(self, *args, **kwargs):
        return []


class _FakeClassificationModel:
    """`build_class_maps` reads `model.names`, so the fake must expose them."""
    names = {0: "empty_track", 1: "wagon"}

    def predict(self, *args, **kwargs):
        return []


def _install_fake_loaders(models_module, monkeypatch):
    def _fake_classification(path):
        from pathlib import Path
        return {"model": _FakeClassificationModel(), "task": "classify",
                "trained_imgsz": 224, "imgsz": 224, "half": False,
                "path": Path(path)}

    monkeypatch.setattr(models_module, "load_classification_model",
                        _fake_classification)
    monkeypatch.setattr(models_module, "load_gap_model",
                        lambda path, device: _FakeGapModel())


# =============================================================================
# 1. the engine's own mapping is what we think it is
# =============================================================================

@requires_engine
def test_engine_mapping_is_side_side_top_top():
    """Verified from the engine's code, not assumed."""
    with gc_runner.engine_session(ENGINE_DIR):
        import camera_map

        classification = dict(camera_map.CAMERA_CLASSIFICATION_MODEL)
        gap = dict(camera_map.CAMERA_GAP_MODEL)

    for camera_id, expected in EXPECTED_CLASSIFICATION.items():
        key = gc_runner.CAMERA_ID_TO_KEY[camera_id]
        assert classification[key] == expected, (
            "%s must use %r classification, engine says %r"
            % (camera_id, expected, classification[key]))
    for camera_id, expected in EXPECTED_GAP.items():
        key = gc_runner.CAMERA_ID_TO_KEY[camera_id]
        assert gap[key] == expected, (
            "%s must use %r gap model, engine says %r"
            % (camera_id, expected, gap[key]))

    # the specific worry from the field report
    assert classification[gc_runner.CAMERA_ID_TO_KEY[C.CAMERA_LEFT_UP_TOP]] \
        == "top", "LEFT_UP_TOP must use TOP classification, not side"
    # and there is no separate left-top gap model: both top cameras share one
    assert gap[gc_runner.CAMERA_ID_TO_KEY[C.CAMERA_LEFT_UP_TOP]] == \
        gap[gc_runner.CAMERA_ID_TO_KEY[C.CAMERA_RIGHT_UP_TOP]] == "top"


@requires_engine
def test_engine_requires_the_complete_registries():
    """Why one camera's pair is not enough -- the check that produced the bug."""
    with gc_runner.engine_session(ENGINE_DIR):
        import camera_map
        required_classification = set(camera_map.CAMERA_CLASSIFICATION_MODEL.values())
        required_gap = set(camera_map.CAMERA_GAP_MODEL.values())

    assert required_classification == {"side", "top"}
    assert required_gap == {"right", "left", "top"}
    # Sequential's registries must cover exactly those.
    assert set(camera_runner.ENGINE_CLASSIFICATION_KEYS.values()) == \
        required_classification
    assert set(camera_runner.ENGINE_GAP_KEYS.values()) == required_gap


# =============================================================================
# 2. every camera satisfies the REAL load_all_models
# =============================================================================

@requires_engine
@pytest.mark.parametrize("camera_id", list(C.ALL_CAMERAS))
def test_real_load_all_models_accepts_sequentials_registries(
        camera_id, weights, monkeypatch):
    """The regression: run the engine's own loader, for each camera in turn."""
    with gc_runner.engine_session(ENGINE_DIR):
        import camera_map
        import models as engine_models

        _install_fake_loaders(engine_models, monkeypatch)
        classification_paths, gap_paths = camera_runner.engine_model_registries(
            weights)

        # This raised RuntimeError("... not loaded: ['side']") before the fix.
        engine_models.load_all_models(classification_paths, gap_paths)
        engine_models.build_class_maps()

        camera_key = gc_runner.CAMERA_ID_TO_KEY[camera_id]
        classification_key = camera_map.CAMERA_CLASSIFICATION_MODEL[camera_key]
        gap_key = camera_map.CAMERA_GAP_MODEL[camera_key]

        # the camera's own keys resolve to loaded models and class maps
        assert classification_key in engine_models.CLASSIFICATION_MODELS
        assert gap_key in engine_models.GAP_MODELS
        assert classification_key in engine_models.CLASSIFICATION_CLASS_MAPS
        assert gap_key in engine_models.GAP_CLASS_MAPS

        assert classification_key == EXPECTED_CLASSIFICATION[camera_id]
        assert gap_key == EXPECTED_GAP[camera_id]

        engine_models.CLASSIFICATION_MODELS.clear()
        engine_models.GAP_MODELS.clear()


@requires_engine
def test_negative_control_one_cameras_pair_still_raises(weights, monkeypatch):
    """Proof this suite would have caught the bug: the old call still fails."""
    with gc_runner.engine_session(ENGINE_DIR):
        import models as engine_models

        _install_fake_loaders(engine_models, monkeypatch)
        # exactly what Sequential used to pass for LEFT_UP_TOP
        with pytest.raises(RuntimeError) as excinfo:
            engine_models.load_all_models(
                {"top": gc_runner._as_paths(
                    {"top": weights["classification_top"]})["top"]},
                {"top": gc_runner._as_paths(
                    {"top": weights["gap_top"]})["top"]})
        message = str(excinfo.value)
        engine_models.CLASSIFICATION_MODELS.clear()
        engine_models.GAP_MODELS.clear()

    assert "not loaded" in message
    assert "side" in message, (
        "expected the engine to name the missing 'side' key; got %r" % message)


@requires_engine
def test_negative_control_camera_id_keys_still_raise(weights, monkeypatch):
    """A camera-id-keyed registry must be rejected by the engine."""
    with gc_runner.engine_session(ENGINE_DIR):
        import models as engine_models

        _install_fake_loaders(engine_models, monkeypatch)
        camera_keyed = gc_runner._as_paths({
            "right_up": weights["classification_side"],
            "left_up": weights["classification_side"],
            "right_up_top": weights["classification_top"],
            "left_up_top": weights["classification_top"],
        })
        _classification, gap_paths = camera_runner.engine_model_registries(
            weights)
        with pytest.raises(RuntimeError) as excinfo:
            engine_models.load_all_models(camera_keyed, gap_paths)
        message = str(excinfo.value)
        engine_models.CLASSIFICATION_MODELS.clear()
        engine_models.GAP_MODELS.clear()

    assert "not loaded" in message
    assert "side" in message and "top" in message, (
        "camera-id keys must not satisfy the engine contract")


@requires_engine
def test_gap_paths_reach_the_engine_as_Path(weights, monkeypatch):
    """The engine calls `.stat()` on gap values -- str would crash (cf. 3dc848c)."""
    from pathlib import Path

    seen = {}
    with gc_runner.engine_session(ENGINE_DIR):
        import models as engine_models

        _install_fake_loaders(engine_models, monkeypatch)
        real_gap_loader = engine_models.load_gap_model

        def _recording(path, device):
            seen[str(path)] = type(path).__name__
            return real_gap_loader(path, device)

        monkeypatch.setattr(engine_models, "load_gap_model", _recording)
        classification_paths, gap_paths = camera_runner.engine_model_registries(
            weights)
        engine_models.load_all_models(classification_paths, gap_paths)
        engine_models.CLASSIFICATION_MODELS.clear()
        engine_models.GAP_MODELS.clear()

    assert seen, "no gap model was loaded"
    assert all("Path" in name for name in seen.values()), seen
    for paths in (classification_paths, gap_paths):
        assert all(isinstance(value, Path) for value in paths.values())


# =============================================================================
# 3. Sequential's mapping, for all four cameras
# =============================================================================

def test_sequential_registry_keys_are_engine_keys_never_camera_ids(weights):
    classification_paths, gap_paths = camera_runner.engine_model_registries(
        weights)

    assert set(classification_paths) == {"side", "top"}
    assert set(gap_paths) == {"right", "left", "top"}

    forbidden = {camera.lower() for camera in C.ALL_CAMERAS}
    forbidden |= set(C.ALL_CAMERAS)
    forbidden |= set(gc_runner.CAMERA_ID_TO_KEY.values())
    for key in list(classification_paths) + list(gap_paths):
        assert key not in forbidden, (
            "%r is a camera key, not an engine model key" % key)


def test_sequential_registry_holds_the_exact_five_weights(weights):
    classification_paths, gap_paths = camera_runner.engine_model_registries(
        weights)

    assert {key: path.name for key, path in classification_paths.items()} == {
        "side": "side_classification.pt", "top": "top_classification.pt"}
    assert {key: path.name for key, path in gap_paths.items()} == {
        "right": "right_up_gap.pt", "left": "left_up_gap.pt",
        "top": "top_gap.pt"}
    assert len(classification_paths) + len(gap_paths) == 5


@pytest.mark.parametrize("camera_id", list(C.ALL_CAMERAS))
def test_each_camera_maps_to_the_validated_keys_and_weights(camera_id, weights):
    """Camera -> engine key -> weight file, for all four cameras."""
    classification_paths, gap_paths = camera_runner.engine_model_registries(
        weights)

    classification_key = EXPECTED_CLASSIFICATION[camera_id]
    gap_key = EXPECTED_GAP[camera_id]

    expected_classification = ("top_classification.pt"
                               if classification_key == "top"
                               else "side_classification.pt")
    expected_gap = {"right": "right_up_gap.pt", "left": "left_up_gap.pt",
                    "top": "top_gap.pt"}[gap_key]

    assert classification_paths[classification_key].name == \
        expected_classification
    assert gap_paths[gap_key].name == expected_gap


def test_side_cameras_use_side_and_top_cameras_use_top():
    for camera_id in C.SIDE_CAMERAS:
        assert EXPECTED_CLASSIFICATION[camera_id] == "side"
    for camera_id in C.TOP_CAMERAS:
        assert EXPECTED_CLASSIFICATION[camera_id] == "top"
    # LEFT_UP_TOP is a TOP camera, so `top` -- the case that failed on EC2.
    assert C.CAMERA_LEFT_UP_TOP in C.TOP_CAMERAS
    assert EXPECTED_CLASSIFICATION[C.CAMERA_LEFT_UP_TOP] == "top"


def test_sequential_derives_the_key_from_the_engine_map_not_the_camera_id():
    """No camera-id-derived key may remain anywhere in the loading path.

    MIGRATED: the per-camera key lookup is no longer Sequential's to do. The
    camera stage hands the engine ALL five weights and calls the engine's own
    `camera_pipeline.process_camera`, which consults the engine's `camera_map`
    itself -- the same way Batch does. So the assertion moved from "Sequential
    reads camera_map correctly" to "Sequential does not choose at all", which
    is the stronger form of the same guarantee.
    """
    import inspect

    source = inspect.getsource(camera_runner.process_camera)

    # it delegates instead of selecting
    assert "camera_pipeline.process_camera" in source
    assert "models.load_all_models(" in source
    assert "engine_model_registries(" in source

    # and no camera-id-derived key survives anywhere in the module
    module_source = inspect.getsource(camera_runner)
    for banned in ("if camera_id in C.TOP_CAMERAS",
                   'CAMERA_RIGHT_UP_TOP: "gap_top"',
                   "CLASSIFICATION_MODEL[camera_key]",
                   "CAMERA_GAP_MODEL[camera_key]"):
        assert banned not in module_source, (
            "the camera stage is picking models itself again: %s" % banned)


def test_sequential_and_batch_build_the_same_registries(weights):
    """Both modes hand the engine the same five weights under the same keys."""
    batch_classification = gc_runner._as_paths({
        key[len("classification_"):]: path
        for key, path in weights.items() if key.startswith("classification_")})
    batch_gap = gc_runner._as_paths({
        key[len("gap_"):]: path
        for key, path in weights.items() if key.startswith("gap_")})

    sequential_classification, sequential_gap = \
        camera_runner.engine_model_registries(weights)

    assert sequential_classification == batch_classification
    assert sequential_gap == batch_gap


def test_the_engine_receives_every_key_its_camera_map_requires(weights):
    """All five weights go in, under the engine's own key names.

    OBSOLETE AS WRITTEN: this asserted against `camera_runner._decode_once`,
    which is gone -- the camera stage does not decode or infer any more. What
    still matters is that the registries handed to `load_all_models` cover the
    ENTIRE camera mapping, because the engine validates all of it at load time
    and raises otherwise. That EC2 failure was:
      RuntimeError: Classification model(s) required by the camera mapping are
      not loaded: ['side']
    """
    classification, gap = camera_runner.engine_model_registries(weights)

    # every key the engine's camera_map can ask for, for ANY camera
    assert set(classification) == {"side", "top"}
    assert set(gap) == {"left", "right", "top"}

    # and Sequential never singles one out
    import inspect

    module_source = inspect.getsource(camera_runner)
    for banned in ('CLASSIFICATION_MODELS["side"]',
                   'CLASSIFICATION_MODELS["top"]',
                   'GAP_MODELS["right"]', 'GAP_MODELS["top"]'):
        assert banned not in module_source, banned
