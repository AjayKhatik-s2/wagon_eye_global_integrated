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

## Known limitations (not yet resolved)

1. **Door/Damage confidence gates.** The camera stage persists RAW detections.
   Batch's Door and Damage processors apply a per-model confidence gate
   (`min_conf`, `confidence_floor`) before aggregating; Global Assembly
   currently relies on `EvidenceAggregator`'s acceptance rules alone. Structure
   (gaps, roster, boundaries, ownership) and Load are unaffected, but Door and
   Damage verdicts may differ from Batch on marginal detections. Batch remains
   the authority for feature facts until a parity run on real footage passes.
2. **Classification batch logic is mirrored, not imported.** The engine's
   `_classify_batch` is a closure inside `classify_video_frames` and cannot be
   called on its own, so `camera_runner._flush_classification` reproduces it
   using the engine's model_info, class map and threshold. An opt-in parity
   check against `classify_video_frames` on real footage is the way to keep
   this honest if the engine ever changes.
3. **No real end-to-end run has been performed** for Sequential. Every test
   here uses a stub engine and a fake video capture.
4. **Reversed cameras.** The Sequential harvest sets `is_reversed=False`; the
   alignment path used here matches positions monotonically and does not yet
   detect a fully reversed support camera the way the engine's own
   `estimate_alignment` does. Batch handles reversal; Sequential would treat a
   reversed camera as unmatched rather than mis-assign it.
