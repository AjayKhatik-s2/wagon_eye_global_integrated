"""The REAL engine's API, checked without loading a weight or decoding a frame.

`tests/test_global_counting_integration.py` proves the adapter and everything
downstream of it are correct, but it feeds them a synthesized harvest. That
leaves one gap: if the engine's own function signatures, config names or module
globals differ from what `global_counting/runner.py` calls, nothing fails until
several minutes into a real run.

This module closes that gap in about a second: it opens a real
`engine_session()` against the installed engine and introspects every symbol
the runner touches. It is the cheapest possible early warning on a fresh box.

Skipped cleanly when the engine is not installed, so it never blocks a
checkout that has not deployed it yet.

    python -m pytest tests/test_engine_contract.py -q
"""

from __future__ import annotations

import inspect
import os
import sys

import pytest

_TEST_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_TEST_DIR)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from global_counting import runner as gc_runner


def _engine_dir():
    try:
        return gc_runner.locate_engine(_REPO_ROOT)
    except gc_runner.GlobalCountingError:
        return None


ENGINE_DIR = _engine_dir()
requires_engine = pytest.mark.skipif(
    ENGINE_DIR is None,
    reason="global_wagon_app engine not installed (set GLOBAL_WAGON_APP_DIR)")


@pytest.fixture
def engine():
    """Everything the runner imports, inside one real isolated session."""
    with gc_runner.engine_session(ENGINE_DIR):
        import camera_map, camera_pipeline, config, io_paths
        import global_alignment, models, reporting, wagon_mapping
        yield {
            "camera_map": camera_map, "camera_pipeline": camera_pipeline,
            "config": config, "io_paths": io_paths,
            "global_alignment": global_alignment, "models": models,
            "reporting": reporting, "wagon_mapping": wagon_mapping,
        }


# -----------------------------------------------------------------------------

@requires_engine
def test_config_accepts_the_overrides_the_runner_applies(engine):
    """The runner must not be refused by the engine's own allow-list."""
    overridable = engine["config"]._OVERRIDABLE
    for name in ("GENERATE_TRIM_DEBUG_VIDEO", "GENERATE_GAP_ANNOTATED_VIDEO"):
        assert name in overridable, "%s is no longer overridable" % name


@requires_engine
def test_keyword_signatures_the_runner_relies_on(engine):
    expected = {
        engine["io_paths"].resolve_inputs: ["video_arguments", "model_arguments"],
        engine["io_paths"].prepare_output_dirs: ["output_dir"],
        engine["models"].load_all_models: ["classification_paths", "gap_paths"],
        engine["global_alignment"].build_normalized_timelines: ["camera_results"],
        engine["global_alignment"].collect_unmatched_extras: ["output_path"],
    }
    for function, leading in expected.items():
        parameters = list(inspect.signature(function).parameters)
        assert parameters[:len(leading)] == leading, (
            "%s signature changed: %s" % (function.__name__, parameters))


@requires_engine
def test_stage_functions_take_no_required_arguments(engine):
    """The runner calls these in the engine's own order, with no arguments."""
    ga = engine["global_alignment"]
    functions = (
        engine["camera_pipeline"].process_all_cameras,
        ga.select_master_camera, ga.validate_temporal_ordering,
        ga.set_master_camera, ga.match_all_cameras,
        ga.report_alignment_mappings, ga.recover_missing_gaps,
        ga.build_global_gap_timeline,
        engine["wagon_mapping"].build_global_wagon_timeline,
    )
    for function in functions:
        required = [p for p in inspect.signature(function).parameters.values()
                    if p.default is inspect.Parameter.empty
                    and p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD)]
        assert not required, "%s now requires %s" % (
            function.__name__, [p.name for p in required])


@requires_engine
def test_model_slot_names_match_the_engine(engine):
    """MODEL_SLOTS keys are split into the dicts load_all_models expects."""
    camera_map = engine["camera_map"]
    mine_classification = {key[len("classification_"):]
                           for key in gc_runner.MODEL_SLOTS
                           if key.startswith("classification_")}
    mine_gap = {key[len("gap_"):] for key in gc_runner.MODEL_SLOTS
                if key.startswith("gap_")}
    assert mine_classification == set(camera_map.CLASSIFICATION_MODEL_FILENAMES)
    assert mine_gap == set(camera_map.GAP_MODEL_FILENAMES)


@requires_engine
def test_camera_keys_match_the_engine(engine):
    from core import constants as C

    assert set(gc_runner.CAMERA_ID_TO_KEY) == set(C.ALL_CAMERAS)
    assert set(gc_runner.CAMERA_ID_TO_KEY.values()) == set(
        engine["camera_map"].CAMERAS)


@requires_engine
def test_module_globals_the_harvester_reads_exist(engine):
    for module_name, attribute in (
        ("global_alignment", "MASTER_CAMERA"),
        ("global_alignment", "GLOBAL_GAP_COUNT"),
        ("global_alignment", "CAMERA_ALIGNMENTS"),
        ("global_alignment", "GLOBAL_GAP_STATUS"),
        ("global_alignment", "MASTER_POSITIONS"),
        ("wagon_mapping", "GLOBAL_WAGONS"),
        ("wagon_mapping", "GLOBAL_WAGON_COUNT"),
        ("camera_pipeline", "CAMERA_RESULTS"),
        ("config", "NORMALIZED_TIMELINE_SCALE"),
    ):
        assert hasattr(engine[module_name], attribute), (
            "%s.%s is gone" % (module_name, attribute))


@requires_engine
def test_engine_csv_writers_exist(engine):
    for name in ("write_normalized_gap_timelines",
                 "write_camera_alignment_summary",
                 "write_global_gap_timeline"):
        assert callable(getattr(engine["reporting"], name, None))


@requires_engine
def test_output_path_keys_the_runner_uses_exist(tmp_path):
    with gc_runner.engine_session(ENGINE_DIR):
        import io_paths
        paths = io_paths.prepare_output_dirs(str(tmp_path / "engine_out"))
        for key in ("normalized_gap_timelines", "camera_alignment_summary",
                    "global_gap_timeline", "unmatched_extra_detections",
                    "global_wagon_timeline"):
            assert key in paths, "prepare_output_dirs lost %r" % key


@requires_engine
def test_wagon_count_rule_is_the_engine_s_own(engine):
    """GLOBAL_WAGON_COUNT = GLOBAL_GAP_COUNT - 1 must come from the engine."""
    source = inspect.getsource(engine["wagon_mapping"].build_global_wagon_timeline)
    assert "GLOBAL_WAGON_COUNT = max(0, gap_count - 1)" in source, (
        "the engine's wagon-count rule changed; the adapter's invariant "
        "check must be revisited")


@requires_engine
def test_session_does_not_leak_engine_modules_afterwards():
    """After the session, no engine module may remain importable-by-accident."""
    with gc_runner.engine_session(ENGINE_DIR):
        import camera_map                                      # noqa: F401
        assert "camera_map" in sys.modules
    assert "camera_map" not in sys.modules
    assert ENGINE_DIR not in sys.path


@requires_engine
def test_our_reporting_package_survives_the_session():
    """The collision that motivated engine_session, checked on the real engine."""
    import reporting as ours
    from reporting import combined_train_report                # noqa: F401

    before = sys.modules["reporting"]
    with gc_runner.engine_session(ENGINE_DIR):
        import reporting as theirs
        # The ENGINE's flat module must win inside the session...
        assert theirs is not before
        assert hasattr(theirs, "write_global_gap_timeline")
    # ...and ours must be back, same object, afterwards.
    assert sys.modules["reporting"] is before is ours
    assert hasattr(sys.modules["reporting"], "combined_train_report")
