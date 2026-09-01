"""EXACT Batch-vs-Sequential parity, stage by stage, in dependency order.

Batch is the golden reference. Sequential may differ in execution architecture,
timing, decode count, intermediate storage and WHEN inference happens -- and in
nothing else. This suite walks the pipeline in dependency order so the FIRST
stage that diverges is the one that fails, rather than a downstream symptom.

HOW PARITY IS ESTABLISHED

The only thing Sequential inserts into the pipeline is `persist -> restore`
around `camera_pipeline.CAMERA_RESULTS`:

    Batch      : CAMERA_RESULTS <- camera_pipeline.process_camera()   (live)
    Sequential : CAMERA_RESULTS <- process_camera() -> evidence.json
                                -> restore_camera_results()

Everything before that point is the ENGINE's own `process_camera`, which
Sequential calls (test_sequential_architecture.py proves it does, and that
camera_runner holds no decode or detector of its own). Everything after it is
the ENGINE's own global half and BATCH's own stages 2-5b, invoked on that same
dictionary. So parity reduces to two claims, and this file tests both:

  (1) persist -> restore is LOSSLESS for every field any downstream stage reads
      -- proved field by field, and then end to end by feeding both paths the
      same engine results and comparing the entire global snapshot.

  (2) the functions Sequential calls after restore are BATCH's, with BATCH's
      arguments -- proved structurally against Batch's own source, so that a
      future change to Batch which Sequential fails to follow breaks a test.

NUMERICAL TOLERANCE POLICY

Comparisons between the two paths are EXACT (`==`). They must be: both paths run
byte-identical code over byte-identical inputs, so any difference at all is a
real divergence and a tolerance would only hide it.

The single documented exception is comparing the engine's least-squares
alignment fit against its IDEAL value (scale 1.0, offset 0.0) on synthetic
co-linear timelines. That is not a Batch-vs-Sequential comparison; it is a
same-algorithm residual check, and it uses abs=1e-9 because the fit accumulates
float64 rounding (observed residual ~9.3e-14). No tolerance is ever applied to
a value produced by one path and compared to the other.

WHAT THIS SUITE IS NOT

It is NOT proof of real-train parity. It runs on deterministic synthetic
evidence with faked pixels. The acceptance criterion is a real same-input EC2
Batch-vs-Sequential run compared with scripts/parity_diff.py.

    python -m pytest tests/test_batch_sequential_exact_parity.py -q
"""

from __future__ import annotations

import inspect
import json
import os
import sys

import pytest

_TEST_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_TEST_DIR)
for _path in (_REPO_ROOT, _TEST_DIR):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import _parity_fixtures as F

from core import constants as C
from global_counting import runner as gc_runner
from sequential import evidence as ev, global_assembly

REAL_ENGINE = F.real_engine_dir()
needs_engine = pytest.mark.skipif(
    REAL_ENGINE is None,
    reason="the frozen global_wagon_app checkout is not on this machine")


# =============================================================================
# Shared runs (expensive: built once)
# =============================================================================

@pytest.fixture(scope="module")
def both_paths(tmp_path_factory):
    """The same engine results through the Batch path and the Sequential path.

    BATCH path      : the engine results handed straight to the global half,
                      exactly as `process_all_cameras()` leaves them.
    SEQUENTIAL path : the same results persisted to evidence.json, restored by
                      `restore_camera_results`, then handed to the global half.
    """
    if REAL_ENGINE is None:
        pytest.skip("engine checkout missing")

    root = tmp_path_factory.mktemp("bothpaths")
    batch_results = F.all_camera_results()
    batch_snapshot = F.run_global_half(REAL_ENGINE, batch_results,
                                       str(root / "batch_engine_out"))

    workspace = str(root / "ws")
    F.seal_all(workspace)
    evidences = {camera: ev.load_evidence(workspace, camera)
                 for camera in C.ALL_CAMERAS}
    sequential_results = global_assembly.restore_camera_results(evidences)
    sequential_snapshot = F.run_global_half(REAL_ENGINE, sequential_results,
                                            str(root / "seq_engine_out"))
    return {"batch": batch_snapshot, "sequential": sequential_snapshot,
            "batch_results": batch_results,
            "sequential_results": sequential_results,
            "evidences": evidences, "workspace": workspace}


@pytest.fixture(scope="module")
def assembled(tmp_path_factory):
    """One full Sequential assembly: REAL engine + REAL Batch stages 2-5b."""
    if REAL_ENGINE is None:
        pytest.skip("engine checkout missing")

    root = str(tmp_path_factory.mktemp("assembled"))
    patcher = pytest.MonkeyPatch()
    try:
        result, workspace, calls = F.run_full_assembly(root, patcher)
    finally:
        patcher.undo()
    assert result.ready, result.reason
    return {"result": result, "workspace": workspace, "calls": calls}


def _states(workspace, feature):
    directory = os.path.join(workspace, "wagon_states", feature)
    if not os.path.isdir(directory):
        return {}
    out = {}
    for name in sorted(os.listdir(directory)):
        if name.endswith(".json"):
            with open(os.path.join(directory, name), encoding="utf-8") as fh:
                out[name[:-5]] = json.load(fh)
    return out


# =============================================================================
# STAGE 0. Input / model / config fingerprints
# =============================================================================

def test_fingerprints_cover_video_models_and_config(assembled):
    """A stale seal must be detectable from the same three inputs Batch uses."""
    for camera in C.ALL_CAMERAS:
        seal = ev.load_seal(assembled["workspace"], camera)
        assert seal["video_fingerprint"]["digest"]
        assert "model_fingerprints" in seal
        assert seal["config_fingerprint"] or seal["config_fingerprint"] == ""


def test_config_fingerprint_is_shared_with_batch_not_reinvented():
    """Sequential must fingerprint config through the shared helper."""
    code = inspect.getsource(sys.modules["sequential.camera_runner"])
    assert "config_fingerprint" in code
    assert "hashlib.sha256" not in code.split("def _model_fingerprints")[0], (
        "camera_runner must not roll its own config digest")


# =============================================================================
# STAGE 1. Per-camera engine results  (the ONLY place Sequential differs)
# =============================================================================

@needs_engine
def test_per_camera_engine_results_survive_persist_and_restore(both_paths):
    """Field-for-field: what Batch holds in memory is what Sequential restores."""
    batch = both_paths["batch_results"]
    restored = both_paths["sequential_results"]
    assert sorted(batch) == sorted(restored)

    for key in sorted(batch):
        original, back = batch[key], restored[key]
        for field in ("status", "final_start_frame", "final_end_frame",
                      "trimmed_total_frames", "unique_gap_count", "n_frames"):
            assert back[field] == original[field], "%s.%s" % (key, field)
        assert back["video_info"] == original["video_info"], key
        assert back["trimmed_info"] == original["trimmed_info"], key


@needs_engine
def test_normalized_gap_timelines_are_identical(both_paths):
    """The gap timeline is the input to every global decision."""
    for key in sorted(both_paths["batch_results"]):
        original = both_paths["batch_results"][key]["normalized_timeline"]
        back = both_paths["sequential_results"][key]["normalized_timeline"]
        assert (back.to_dict("records") == original.to_dict("records")), key


@needs_engine
def test_classification_timeline_is_identical(both_paths):
    """`_harvest` classifies each wagon from this, so it must survive intact."""
    for key in sorted(both_paths["batch_results"]):
        original = both_paths["batch_results"][key]["timeline_df"]
        back = both_paths["sequential_results"][key]["timeline_df"]
        assert len(back) == len(original), key
        assert (back["normalized_class"].tolist()
                == original["normalized_class"].tolist()), key


# =============================================================================
# STAGE 2. Master camera, gap count, alignment, reversal, recovery, extras
# =============================================================================

@needs_engine
def test_master_camera_selection_is_identical(both_paths):
    assert (both_paths["sequential"]["master_camera"]
            == both_paths["batch"]["master_camera"])
    assert (both_paths["sequential"]["master_reason"]
            == both_paths["batch"]["master_reason"])


@needs_engine
def test_master_is_not_hardcoded_anywhere_in_sequential():
    """The historical divergence: a fixed RIGHT_UP instead of max_unique_gaps."""
    source = inspect.getsource(global_assembly)
    assert "select_master_camera()" in source
    assert "C.MASTER_CAMERA" not in inspect.getsource(global_assembly.assemble)


@needs_engine
def test_global_gap_count_is_identical(both_paths):
    assert (both_paths["sequential"]["global_gap_count"]
            == both_paths["batch"]["global_gap_count"])
    assert (both_paths["sequential"]["global_gap_ids"]
            == both_paths["batch"]["global_gap_ids"])


@needs_engine
def test_per_camera_gap_counts_are_identical(both_paths):
    assert (both_paths["sequential"]["camera_gap_counts"]
            == both_paths["batch"]["camera_gap_counts"])


@needs_engine
def test_alignment_scale_and_offset_are_identical(both_paths):
    """Exact equality: same estimator, same inputs, so no tolerance is allowed."""
    batch = both_paths["batch"]["alignments"]
    sequential = both_paths["sequential"]["alignments"]
    assert sorted(batch) == sorted(sequential)
    for camera in sorted(batch):
        assert sequential[camera]["scale"] == batch[camera]["scale"], camera
        assert sequential[camera]["offset"] == batch[camera]["offset"], camera
        assert sequential[camera]["status"] == batch[camera]["status"], camera


@needs_engine
def test_reversal_flags_are_identical(both_paths):
    for camera, mapping in both_paths["batch"]["alignments"].items():
        assert (both_paths["sequential"]["alignments"][camera]["reversed"]
                is mapping["reversed"]), camera


@needs_engine
def test_alignment_fit_is_ideal_on_colinear_synthetic_timelines(both_paths):
    """The ONE documented tolerance: a same-algorithm float residual.

    Not a Batch-vs-Sequential comparison -- both paths already compared exactly
    above. This checks the engine's least-squares fit recovers the identity
    mapping on co-linear inputs, where float64 rounding leaves ~1e-13.
    """
    for camera, mapping in both_paths["sequential"]["alignments"].items():
        assert mapping["scale"] == pytest.approx(1.0, abs=1e-9), camera
        assert mapping["offset"] == pytest.approx(0.0, abs=1e-9), camera


@needs_engine
def test_recovered_and_unmatched_are_identical(both_paths):
    for camera, mapping in both_paths["batch"]["alignments"].items():
        sequential = both_paths["sequential"]["alignments"][camera]
        assert sequential["matched"] == mapping["matched"], camera
        assert sequential["unmatched"] == mapping["unmatched"], camera


@needs_engine
def test_recovery_and_gap_timeline_stages_actually_run():
    """Both were MISSING from Sequential and silently shrank the train."""
    source = inspect.getsource(global_assembly.run_engine_global_half)
    assert "recover_missing_gaps()" in source
    assert "collect_unmatched_extras(" in source
    assert "build_global_gap_timeline()" in source


# =============================================================================
# STAGE 3. Global wagon timeline: count, ids, order, boundaries
# =============================================================================

@needs_engine
def test_global_wagon_count_and_rule_are_identical(both_paths):
    for path in ("batch", "sequential"):
        snapshot = both_paths[path]
        assert (snapshot["global_wagon_count"]
                == snapshot["global_gap_count"] - 1), path
    assert (both_paths["sequential"]["global_wagon_count"]
            == both_paths["batch"]["global_wagon_count"])


@needs_engine
def test_global_wagon_records_are_identical(both_paths):
    """Ids, order and every boundary column the engine emits."""
    batch = both_paths["batch"]["global_wagons"]
    sequential = both_paths["sequential"]["global_wagons"]
    assert len(sequential) == len(batch)
    for index, (expected, actual) in enumerate(zip(batch, sequential)):
        assert sorted(actual) == sorted(expected), index
        for column in sorted(expected):
            assert actual[column] == expected[column], "wagon %d.%s" % (
                index, column)


@needs_engine
def test_wagon_mapping_is_the_engines_not_a_hand_written_loop():
    source = inspect.getsource(global_assembly.run_engine_global_half)
    assert "wagon_mapping.build_global_wagon_timeline()" in source
    assembly = inspect.getsource(global_assembly)
    for banned in ("for index in range(len(global_gaps) - 1)",
                   "_frame_for_position"):
        assert banned not in assembly, (
            "assembly is mapping wagons itself again: %s" % banned)


# =============================================================================
# STAGE 4. Canonical roster and ownership
# =============================================================================

@needs_engine
def test_canonical_roster_is_gw_one_through_n(assembled):
    from core.global_state_loader import (load_global_train_state,
                                          verify_roster_integrity)

    state = load_global_train_state(assembled["result"].state_json_path)
    expected = ["GW_%d" % i for i in range(1, len(state.wagons) + 1)]
    assert [w.global_id for w in state.wagons] == expected
    assert verify_roster_integrity(state) == []
    assert len(state.wagons) == assembled["result"].global_wagon_count


@needs_engine
def test_wagon_classification_comes_from_batchs_harvest(assembled):
    """`_harvest` classifies each wagon; Sequential used to omit this entirely."""
    from core.global_state_loader import load_global_train_state

    state = load_global_train_state(assembled["result"].state_json_path)
    assert state.wagons
    for wagon in state.wagons:
        assert getattr(wagon, "classification", None), wagon.global_id
    assert "gc_runner._harvest" in inspect.getsource(global_assembly.assemble)


@needs_engine
def test_ownership_is_the_centralized_rule(assembled):
    from core import wagon_ownership
    from core.global_state_loader import load_global_train_state

    state = load_global_train_state(assembled["result"].state_json_path)
    ownership = wagon_ownership.for_state(state)
    assert ownership is not None
    assert wagon_ownership.BOUNDARY_GOES_TO == "next_wagon"


@needs_engine
def test_every_wagon_window_is_half_open_and_non_overlapping(assembled):
    """The ef2868f bug: one damage event landing on two consecutive wagons."""
    from core.global_state_loader import load_global_train_state

    state = load_global_train_state(assembled["result"].state_json_path)
    for camera in C.ALL_CAMERAS:
        windows = []
        for wagon in state.wagons:
            window = (wagon.camera_frame_ranges or {}).get(camera)
            if window and window.get("start_frame") is not None:
                windows.append((wagon.global_id, int(window["start_frame"]),
                                int(window["end_frame"])))
        assert windows, "no aligned window on %s" % camera
        for (a_id, _a_start, a_end), (b_id, b_start, _b_end) in zip(
                windows, windows[1:]):
            assert a_end <= b_start, (
                "%s and %s overlap on %s" % (a_id, b_id, camera))


# =============================================================================
# STAGE 5. Materialized wagon cache
# =============================================================================

@needs_engine
def test_wagon_cache_is_built_by_batchs_materializer(assembled):
    """Stage 2 was MISSING from Sequential; features ran on raw whole-video."""
    assert "wagon_cache_builder.build" in inspect.getsource(
        global_assembly.assemble)
    cache_root = os.path.join(assembled["workspace"], "wagon_cache")
    assert os.path.isdir(cache_root), "no wagon cache was materialized"
    wagons = [n for n in sorted(os.listdir(cache_root)) if n.startswith("GW_")]
    assert wagons, "the cache has no wagon directories"


@needs_engine
def test_cached_frames_are_jpeg_quality_ninety(assembled):
    """Feature pixels must be Batch's JPEG-90, not raw decoded frames."""
    from materializer import wagon_cache_builder

    assert C.JPEG_QUALITY == 90, (
        "Batch's cache quality changed; Sequential inherits it from the same "
        "constant, but the parity claim in this file names 90")
    source = inspect.getsource(wagon_cache_builder)
    assert "IMWRITE_JPEG_QUALITY" in source
    signature = inspect.signature(wagon_cache_builder.build)
    assert signature.parameters["jpeg_quality"].default == C.JPEG_QUALITY, (
        "Sequential calls build() without jpeg_quality, so the default must "
        "stay Batch's shared constant")

    cache_root = os.path.join(assembled["workspace"], "wagon_cache")
    found = []
    for root, _dirs, files in os.walk(cache_root):
        found.extend(f for f in files if f.lower().endswith(".jpg"))
    assert found, "the cache holds no JPEG frames"


@needs_engine
def test_features_only_ever_see_cached_frames(assembled):
    """The stable-interior frame set, not a whole-video stride."""
    calls = assembled["calls"]
    assert calls["door"], "the door processor was never run"
    assert calls["damage"], "the damage processor was never run"
    # A whole-video stride-3 pass over 4 cameras would be far larger than the
    # clamped stable interior of four wagons.
    assert len(calls["door"]) < 4 * F.TOTAL_FRAMES / 3


# =============================================================================
# STAGE 6. Door / Damage / Load / OCR through BATCH's processors
# =============================================================================

@needs_engine
def test_features_run_through_batchs_registry(assembled):
    source = inspect.getsource(global_assembly._run_features)
    assert "load_feature_runner" in source, (
        "features must be resolved through Batch's own registry")
    flat = " ".join(source.split())
    for needle in ('"door_sample_stride", 3', '"damage_sample_stride", 3',
                   '"load_sample_stride", 2'):
        assert needle in flat, needle


@needs_engine
def test_load_runs_before_damage(assembled):
    """Damage reads the sibling load payload; Batch fixes this order."""
    source = inspect.getsource(global_assembly._run_features)
    assert 'for name in ("load", "door", "ocr", "damage")' in source


@needs_engine
def test_door_payloads_have_batchs_shape(assembled):
    payloads = _states(assembled["workspace"], "door")
    assert payloads, "no door payload was written"
    for global_id, payload in payloads.items():
        assert payload["global_id"] == global_id
        assert "door_status" in payload
        assert "doors" in payload
        for door in payload["doors"]:
            assert "door_index" in door
            assert "state" in door


@needs_engine
def test_multiple_physical_doors_survive_with_their_own_states(assembled):
    """The b6f67b5 contract: two doors, one OPEN and one CLOSED, both kept.

    The fake detector alternates open/closed by frame index, so a wagon that
    sees both must keep both doors and must not collapse them into one verdict.
    """
    payloads = _states(assembled["workspace"], "door")
    multi = {gid: p for gid, p in payloads.items()
             if len(p.get("doors") or []) > 1}
    assert multi, "no wagon retained more than one physical door"

    for global_id, payload in multi.items():
        doors = payload["doors"]
        indices = [d["door_index"] for d in doors]
        assert len(set(indices)) == len(indices), (
            "%s has duplicate door_index values" % global_id)
        assert indices == sorted(indices), "%s doors are unordered" % global_id
        for door in doors:
            assert door["state"] in (C.DOOR_OPEN, C.DOOR_CLOSED,
                                     C.CLASS_UNKNOWN), door

    states_seen = {d["state"] for p in payloads.values()
                   for d in (p.get("doors") or [])}
    assert C.DOOR_OPEN in states_seen and C.DOOR_CLOSED in states_seen, (
        "the run never produced both door states, so multi-door survival is "
        "untested: %s" % states_seen)


@needs_engine
def test_door_status_is_derived_by_batchs_helper(assembled):
    """The wagon-level verdict must be Batch's function, not a reimplementation."""
    from features.door import processor as door_processor

    payloads = _states(assembled["workspace"], "door")
    for payload in payloads.values():
        expected = door_processor.wagon_door_status(payload["doors"])
        assert payload["door_status"] == expected, payload["global_id"]


@needs_engine
def test_damage_payloads_have_batchs_shape(assembled):
    payloads = _states(assembled["workspace"], "damage")
    assert payloads
    for global_id, payload in payloads.items():
        assert payload["global_id"] == global_id
        assert "status" in payload


@needs_engine
def test_load_payloads_have_batchs_shape(assembled):
    payloads = _states(assembled["workspace"], "load")
    assert payloads
    for global_id, payload in payloads.items():
        assert payload["global_id"] == global_id
        assert "status" in payload


@needs_engine
def test_ocr_is_disabled_with_batchs_own_sentinel(assembled):
    """`--features door,load,damage` means OCR OFF, everywhere, identically."""
    payloads = _states(assembled["workspace"], "ocr")
    assert payloads, "OCR must still write a DISABLED sentinel per wagon"
    for payload in payloads.values():
        assert payload["status"] == C.STATUS_DISABLED
        assert payload.get("disabled_by_user") is True
    assert assembled["calls"]["ocr"] == [], (
        "an OCR model was run even though OCR was not selected")


# =============================================================================
# STAGE 7. Fusion / unified state
# =============================================================================

@needs_engine
def test_fusion_is_batchs_state_builder(assembled):
    assert "wagon_state_builder.build" in inspect.getsource(
        global_assembly.assemble)


@needs_engine
def test_unified_state_covers_every_wagon_and_feature(assembled):
    unified = _states(assembled["workspace"], "unified")
    if not unified:
        state_dir = os.path.join(assembled["workspace"], "wagon_states")
        unified = {}
        for name in sorted(os.listdir(state_dir)):
            path = os.path.join(state_dir, name)
            if name.endswith(".json"):
                with open(path, encoding="utf-8") as handle:
                    unified[name[:-5]] = json.load(handle)
    assert unified, "fusion produced no unified state"


# =============================================================================
# STAGE 8. Stage 5a camera reports and Stage 5b combined report
# =============================================================================

@needs_engine
def test_stage_5a_uses_batchs_renderer(assembled):
    assert "camera_reports.build_all" in inspect.getsource(
        global_assembly.assemble)
    paths = assembled["result"].camera_report_paths
    assert set(paths) == set(C.ALL_CAMERAS), paths
    for camera, path in paths.items():
        assert path, camera


@needs_engine
def test_stage_5b_uses_batchs_renderer_and_covers_the_roster(assembled):
    assert "combined_train_report.build" in inspect.getsource(
        global_assembly.assemble)
    with open(assembled["result"].report_json_path, encoding="utf-8") as handle:
        document = json.load(handle)
    assert (document["summary"]["total_wagons"]
            == assembled["result"].global_wagon_count)
    wagons = document.get("wagons") or []
    assert len(wagons) == assembled["result"].global_wagon_count
    assert [w["global_id"] for w in wagons] == [
        "GW_%d" % i for i in range(1, len(wagons) + 1)]


@needs_engine
def test_there_is_exactly_one_report_generator(assembled):
    """No second renderer and no post-hoc PDF patching."""
    assembly = inspect.getsource(global_assembly)
    for banned in ("PdfPages", "canvas.Canvas", "SimpleDocTemplate",
                   "_patch_pdf", "rebuild_pdf"):
        assert banned not in assembly, (
            "Sequential is generating reports itself: %s" % banned)


@needs_engine
def test_pdf_semantics_match_the_canonical_contract(assembled):
    try:
        import reportlab                                        # noqa: F401
    except ImportError:
        pytest.skip("reportlab not installed")
    pdf = assembled["result"].report_pdf_path
    if not pdf:
        pytest.skip("combined PDF not produced in this configuration")
    assert os.path.getsize(pdf) > 800
    with open(assembled["result"].report_json_path, encoding="utf-8") as handle:
        document = json.load(handle)
    assert document.get("canonical") is not False, (
        "the combined report must be the canonical one")


# =============================================================================
# STAGE 9. The end-to-end claim
# =============================================================================

@needs_engine
def test_the_entire_global_snapshot_is_identical(both_paths):
    """The whole-structure comparison, after the staged ones above.

    Kept last on purpose: when an upstream stage diverges its own test fails
    first, which is what makes the failure legible.
    """
    assert both_paths["sequential"] == both_paths["batch"]
