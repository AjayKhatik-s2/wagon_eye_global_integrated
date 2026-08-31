# Dual-mode architecture map

`CAMERA = EVIDENCE, GLOBAL = MEANING`

Both modes are first-class. `orchestrator.master_runner` selects one with
`--mode {batch,sequential}` and neither may break the other.

```
master_runner
  ├── --mode batch      -> orchestrator.master_runner.process_batch   (UNCHANGED, validated)
  └── --mode sequential -> sequential.runner.run_sequential           (NEW)
```

## Batch (existing, validated — untouched)

```
4 videos
  Stage 1  reconstruction.runner.run          -> global_train_state.json  (engine drives all 4 cameras)
  Stage 2  materializer.wagon_cache_builder   -> wagon_cache/<GW_n>/<cam>/*.jpg
  Stage 3  features.{load then door,ocr,damage}.processor.run  (read cached JPEGs, per GW_n)
  Stage 4  fusion.wagon_state_builder
  Stage 4b rendering.feature_overlay_renderer
  Stage 5a reporting.camera_reports           (per-camera PDF, POST-global, uses GW ids)
  Stage 5b reporting.combined_train_report
  Stage 6  delivery.{s3_upload,notification}
```

Decodes per camera in Batch: engine classification pass, engine gap pass on the
trimmed clip, plus the Stage-2 materializer pass. Features then read JPEGs.

## Sequential (new)

```
for camera in (RIGHT_UP, LEFT_UP, RIGHT_UP_TOP, LEFT_UP_TOP):   # deterministic, C.ALL_CAMERAS
    sequential.camera_runner.process_camera(camera)
        ONE cv2.VideoCapture / ONE frame loop, feeding on the SAME decoded frame:
            every frame      -> engine gap_detection.detect_gaps_in_frame   (per-frame, stateless)
            every frame      -> classification batch (BATCH_SIZE)           (engine model + class map)
            stride 3         -> door   detector
            stride 3         -> damage detector
            stride 2         -> load   classifier
        after the loop (NO second decode, NO re-inference):
            trimming boundaries from the classification timeline
            replay the persisted gap detections for [final_start..final_end]
                through a fresh engine GapTracker  -> identical unique gaps
        persist camera evidence  (NO GW_ ids anywhere)
        camera-local JSON + PDF  (segments are <CAM>_SEG_n, explicitly NOT canonical)
        write sealed.json
        release VideoCapture + models + tensors
    -> next camera

sequential.global_assembly.assemble()            # exactly ONCE
        load sealed evidence only
        engine alignment on the persisted per-camera gap timelines
        ONE canonical gap sequence + ONE canonical roster GW_1..GW_N
        project canonical timeline into each camera (aligned frame ranges)
        assign persisted observations via core.wagon_ownership   (ef2868f rule)
        run the pure Door/Damage/Load aggregators on assigned evidence
        fusion.wagon_state_builder -> reporting.combined_train_report
```

## Component ownership

| Component | Batch | Sequential | Notes |
|---|---|---|---|
| `reconstruction/runner.py` | yes | no | Batch Stage 1 |
| `global_counting/{runner,adapter}.py` | yes | adapter reused | Sequential reuses the adapter's contract builder |
| `materializer/wagon_cache_builder.py` | yes | no | Sequential persists observations, not JPEG caches |
| `features/*/processor.py` | yes | no (their aggregators are reused) | processors need GW ids + JPEG caches |
| `features/evidence_aggregator.py` | yes | yes | PURE — reused by Global Assembly |
| `core/wagon_ownership.py` | yes | yes | one ownership rule for both |
| `fusion/wagon_state_builder.py` | yes | yes | unchanged semantics |
| `reporting/combined_train_report.py` | yes | yes | combined report only |
| `reporting/camera_reports.py` | yes | no | POST-global, needs GW ids |
| `sequential/*` | no | yes | new |

## Invariants

1. Camera-local evidence never contains `GW_`. Canonical ids exist only after
   Global Assembly.
2. Exactly one canonical gap sequence and one roster, created once.
3. RIGHT_UP is the canonical gap authority; support cameras corroborate only.
4. Global Assembly performs no `VideoCapture`, no YOLO predict, no model load.
5. One decode lifecycle per camera; GAP sees every decoded frame.
6. Door 3 / Damage 3 / Load 2 sampling unchanged; OCR off unless selected.
7. `global_count_ec2` is frozen and unmodified.

## Parity with Batch — how each decision is kept identical

| Decision | Shared mechanism |
|---|---|
| Door detection gate | `TrackerConfig().closed_confidence_threshold`, read from Batch's own config object |
| Damage detection gate | `features.damage.processor._filter_detections_for_top`, called directly |
| Door aggregation | `EvidenceAggregator` + `_door_evidence_from_groups` + `_pick_side_state` + `order_doors` + `wagon_door_status` |
| Door class mapping | `C.DOOR_LABEL_TO_STATE` / `door_processor._canonical` |
| Load verdict | `_LOADED_RATIO_THRESHOLD` and Batch's `used` denominator and branch order |
| Alignment + reversal | the engine's `estimate_alignment` (both directions tested) |
| Projection into a camera | the alignment's own `project_to_camera` (mirrors when reversed) |
| Wagon ownership | `core.wagon_ownership` (ef2868f) |
| Canonical contract | `global_counting.adapter`, the same builder Batch uses |

Nothing in that column is a copy: Sequential calls the same object Batch calls.
`tests/test_batch_sequential_parity.py` (30 tests) compares the two paths at
each point, and asserts a second implementation has not appeared.

## Classification: the one mirrored piece

The engine's only classification entry point takes a **video path** and owns its
own `VideoCapture`, so it cannot be reused without decoding the video a second
time — which is the invariant Sequential exists to uphold. The per-frame verdict
is therefore reproduced in `sequential/classification_adapter.py`, and nowhere
else. Everything it uses is the engine's: model_info, class maps, the
confidence threshold, `BATCH_SIZE`, `DEVICE_YOLO`, and the batch→per-frame
fallback.

`tests/test_classification_adapter_contract.py` runs the REAL
`classify_video_frames` and the adapter over the SAME frames and requires the
records to be identical field for field, and separately asserts the engine still
contains the record keys and threshold comparison being mirrored. It also fails
if the engine ever gains a frame-based entry point, so the mirror gets deleted
rather than forgotten.

## Known limitations (not yet resolved)

1. **No real end-to-end run has been performed** in Sequential mode. Every test
   uses a stub engine and a fake video capture. Batch remains the mode with a
   real-footage history.
2. The gap detector runs on every decoded frame, including outside the wagon
   region, which the engine avoids by detecting on the trimmed clip. Results are
   unchanged (the tracker sees the same slice); the cost is extra inference.
