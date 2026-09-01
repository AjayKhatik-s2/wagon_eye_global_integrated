# WagonEye — global_wagon_app counting + the f3d2d81 downstream pipeline

An integration of two validated projects:

| | Project | Role |
|---|---|---|
| 1 | **`global_wagon_app`** ([`global_count_ec2`](https://github.com/AjayKhatik-s2/global_count_ec2)) | Stage-1 counting: the authoritative global wagon timeline |
| 2 | **this repository** (from `final_version_wagon` @ `f3d2d81`) | everything downstream: Door / Load / Damage / OCR, fusion, rendering, reporting, delivery |

Single source of truth for wagon counting, numbering, and classification
is the **GlobalTrainState** produced by the external counting engine.
Everything downstream — frame extraction, feature inference, fusion,
reporting — consumes that state and never recounts wagons, never re-segments
video, never re-runs gap detection.

The engine is a **separate repository, deliberately not vendored here**, so the
validated counting algorithm has exactly one source of truth. See
[`global_counting/README.md`](global_counting/README.md) for the engine
contract, the import-isolation design, and rollback.

```
4 source videos
    │
    ▼
[Stage 1] reconstruction.runner  →  global_state/global_train_state.json
            SOLE counting authority: the external global_wagon_app engine,
            driven by global_counting/{runner,adapter}.py:
              classification → wagon-region trimming → gap detection
              → gap tracking → unique-gap confirmation
              → 0–1000 normalization
              → DYNAMIC master-camera selection (max confirmed unique gaps)
              → cross-camera alignment (scale + direction, reversal aware)
              → missing-gap recovery
              → GLOBAL GAP TIMELINE → GLOBAL WAGON TIMELINE
              → GLOBAL_WAGON_COUNT = GLOBAL_GAP_COUNT − 1
              → wagon window (ENGINE / BRAKE_VAN get no GW id)
              → per-wagon ALIGNED frame range for each of the 4 cameras
            Emits one IMMUTABLE roster GW_1..GW_N.  No later stage may
            count, re-segment, renumber or reorder it.
            The retained wagon_count subprocess counter stays selectable
            for rollback (--stage1-engine wagon_count); exactly one engine
            ever runs per batch.
    │
    ▼
[Stage 2] materializer.wagon_cache_builder
            single-pass per video → wagon_cache/<GW_n>/<camera>/*.jpg
            uses each wagon's ALIGNED per-camera frame range when the
            counting engine supplies one, instead of projecting
            master_time × local_fps (the four cameras are not guaranteed
            to share t=0, and one may run reversed)
    │
    ▼
[Stage 3] features.load  runs FIRST, to completion
            (damage reads the load JSON to drop floor_damage on LOADED
             wagons; running it first makes that read deterministic)
          then features.{door,ocr,damage} in parallel
            pure YOLO inference on cached frames
            persists evidence snapshots + metadata under evidence/
            → wagon_states/<feature>/<GW_n>.json
            → evidence/<GW_n>/<feature>/{*.jpg,metadata.json}
    │
    ▼
[Stage 4] fusion.wagon_state_builder
            authority-rule merge → wagon_states/unified/<GW_n>.json
    │
    ▼
[Stage 4b] rendering.feature_overlay_renderer  (visualization-only,
            consumes state + evidence + tracking JSON; never reruns
            any detector model)
            → processed_videos/<CAM>_processed.mp4
    │
    ▼
[Stage 5a] reporting.camera_reports   (legacy camera-wise hierarchy)
            4 camera-wise PDFs, each showing ONLY what that camera is
            authoritative for:
              RIGHT_UP     → right door, OCR, classification
              LEFT_UP      → left door
              RIGHT_UP_TOP → load, top damage
              LEFT_UP_TOP  → top-damage support / validation
            Each report: Camera Summary page (visible wagons, anomalies,
            processing confidence, coverage %), per-wagon pages (2×2
            quartile overview + per-detection detail pages, snapshots
            from THAT camera only), Anomaly Summary grouped by severity,
            and a Camera Evidence section.
            → reports/{right_up,left_up,right_up_top,left_up_top}_report.pdf
    │
    ▼
[Stage 5b] reporting.combined_train_report
            Aggregates the 4 camera reports into the unified train view
            (legacy CombinedReportGenerator visual identity, rebuilt
            against v4 state via _adapter.LegacyViewModel): navy title
            banner, VIDEO EVIDENCE table, PARTIAL REPORT banner,
            DETAILED CAMERA REPORTS links (LEFT/RIGHT/R-TOP/L-TOP),
            10-col INSPECTION SUMMARY KPI row (LOCO, RAKE TYPE, STATUS),
            7-col wagon inspection table with issue-row highlighting,
            and the "Damaged Wagon Report" per-anomaly evidence grid
            (camera-priority ordered).  Schema v4 JSON includes
            legacy_view_model.
            → reports/combined_train_report.{pdf,json}
    │
    ▼
[Stage 6] delivery.s3_upload + delivery.notification
            S3 archive (incl. evidence + processed_videos) +
            one email per batch
```

## Package layout

```
wagon_eye_v4/
├── README.md                          (this file)
├── requirements.txt
├── pytest.ini                         test collection scope
├── orchestrator/master_runner.py      ★ entry point
├── reconstruction/runner.py           Stage 1 (subprocess wagon_count)
├── materializer/wagon_cache_builder.py Stage 2
├── features/
│   ├── _common.py                     YOLO loader cache + helpers
│   ├── _evidence.py                   evidence persistence helpers
│   │                                   (per-wagon JPEG + metadata)
│   ├── inference_lib/                 mature tracking intelligence ported
│   │                                   from the legacy system (door tracker,
│   │                                   identity merger, illumination, shape
│   │                                   prior, damage tracker, OCR + number
│   │                                   aggregator, temporal reasoning)
│   ├── door/processor.py              door_state.pt
│   ├── load/processor.py              loaded.pt
│   ├── damage/processor.py            damage.pt
│   └── ocr/processor.py               wagon_id_counting.pt + easyocr
├── fusion/wagon_state_builder.py      Stage 4
├── rendering/
│   └── feature_overlay_renderer.py    Stage 4b (visualization-only;
│                                       never reruns any detector)
├── reporting/
│   ├── _brand.py                      Legacy WagonEye palette, paragraph
│   │                                   styles, anomaly + state helpers,
│   │                                   page widgets (logo, warning banner,
│   │                                   camera links).
│   ├── _adapter.py                    v4 backend -> legacy report view-
│   │                                   model (merged_wagons + per-camera
│   │                                   doors + KPIs).
│   ├── _evidence_lookup.py            Quartile + midpoint cache frame
│   │                                   resolution + evidence snapshot
│   │                                   path helpers.
│   ├── _pages.py                      Shared reportlab page widgets
│   │                                   (doc maker, bordered image,
│   │                                   detection-summary table, wagon
│   │                                   quartile overview, detail page,
│   │                                   simple-state page).
│   ├── camera_reports.py              Stage 5a (4 camera-wise PDFs by
│   │                                   camera authority: RIGHT_UP /
│   │                                   LEFT_UP / RIGHT_UP_TOP /
│   │                                   LEFT_UP_TOP).
│   ├── combined_train_report.py       Stage 5b (aggregates the 4 camera
│   │                                   reports; legacy visual identity:
│   │                                   navy title banner, 10-col KPI
│   │                                   summary, 7-col wagon table,
│   │                                   Damaged Wagon Report evidence
│   │                                   grid; schema v4).
│   └── assets/Logo.jpeg               Per-page logo (copied from the
│                                       legacy product).
├── delivery/
│   ├── s3_upload.py                   Stage 6: PDF/JSON + tree upload
│   └── notification.py                Stage 6: one email per batch
├── core/
│   ├── constants.py                   camera ids, classes, statuses
│   ├── unified_wagon_state.py         UnifiedWagonState dataclass
│   ├── global_state_loader.py         ★ THE counting→inspection adapter:
│   │                                   parses the engine JSON, freezes the
│   │                                   roster, exposes camera offsets,
│   │                                   roster fingerprint + integrity check
│   ├── feature_config.py              Stage-3 feature registry + toggles
│   ├── frame_quality.py               evidence-frame scoring heuristics
│   └── batch.py                       CameraVideo / TrainBatch
├── models/
│   ├── reconstruction/                Stage-1 counting weights, EXACT names
│   │                                   (right_up_wagon_gap.pt,
│   │                                    left_up_wagon_gap.pt, top_gap.pt,
│   │                                    side_classification.pt,
│   │                                    top_classification.pt = optional)
│   └── features/                      Stage-3 inspection weights
│                                       (door_state.pt, loaded.pt,
│                                        damage.pt, wagon_id_counting.pt)
├── tests/                             v4 integration + counting-swap
│                                       regression suite (70 tests)
├── _legacy_wagon_count_removed/       the REPLACED counting modules, kept
│                                       for review only.  Not a package,
│                                       not importable, asserted unreferenced.
│                                       Safe to delete after review.
└── wagon_count/                       ★ the counting engine — adopted
    ├── run_global_count.py             byte-identical from the proven
    ├── tracker_engine.py               correct-count implementation
    ├── fragment_stitching.py           tracker fragments → physical gaps
    ├── gap_validation.py               candidate → validated boundary
    ├── temporal_classification.py      class-sequence hysteresis
    ├── train_structure.py              wagon window (no GW id for
    │                                    ENGINE / BRAKE_VAN)
    ├── global_fusion.py                fixed-master fusion + camera offsets
    ├── global_alignment.py             segment build (+ retained legacy path)
    ├── global_train_state.py           engine-side data contracts
    ├── video_segmenter.py              Stage-1 debug overlay videos
    ├── evidence_report.py              engine's own PDF (skipped by v4)
    ├── rejection_report.py             gap-rejection CSV/JSON tooling
    └── tests/                          the engine's own suite (283 tests)
```

There are NO `RIGHT_UP/`, `LEFT_UP/`, `RIGHT_UP_TOP/`, `LEFT_UP_TOP/`
folders inside this package. Camera-centric assumptions, legacy
`DoorProcessor` / `DamageProcessor` wrappers, `cv2.VideoCapture` calls
downstream of Stage 1, and mini-mp4 reconstruction are all removed.

## Output per batch

```
batch_outputs/<batch_key>/
├── downloads/                         raw videos (downloaded from S3, or
│                                       local-passthrough)
├── global_state/
│   ├── global_train_state.json
│   ├── per_camera_tracking.json
│   └── processed_videos/              wagon_count's debug tracking videos
│       └── <CAM>_processed.mp4         (4 cameras)
├── wagon_cache/
│   ├── GW_1/{right_up,left_up,right_up_top,left_up_top}/frame_*.jpg
│   ├── GW_2/...
│   └── ...
├── wagon_states/
│   ├── door/GW_*.json
│   ├── load/GW_*.json
│   ├── damage/GW_*.json
│   ├── ocr/GW_*.json
│   └── unified/GW_*.json
├── evidence/                          best-frame snapshots + metadata
│   ├── GW_1/door/{left_best,left_crop,right_best,right_crop}.jpg
│   │       + metadata.json
│   ├── GW_1/damage/{track_1,track_1_crop,...}.jpg + metadata.json
│   ├── GW_1/ocr/{best_frame,number_crop}.jpg + metadata.json
│   ├── GW_1/load/best_frame.jpg + metadata.json
│   └── GW_2/...
├── processed_videos/                  rich feature-overlay videos
│   └── <CAM>_processed.mp4             (4 cameras; rendered from
│                                        state+evidence — no detector rerun)
├── reports/
│   ├── combined_train_report.json     schema v4 (legacy_view_model +
│   │                                   evidence_pages)
│   ├── combined_train_report.pdf      legacy WagonEye visual identity:
│   │                                   title banner + KPIs + wagon table +
│   │                                   Damaged Wagon Report
│   ├── right_up_report.pdf            camera-wise reports (Stage 5a):
│   ├── left_up_report.pdf              one per camera, each scoped to
│   ├── right_up_top_report.pdf         that camera's authority
│   └── left_up_top_report.pdf
└── archive/                           run logs (future)
```

## Deployment (EC2, CPU)

### 1. Install

```bash
git clone <this repo> wagon_eye_global_integrated
cd wagon_eye_global_integrated
WITH_OCR=0 bash scripts/setup_ec2.sh        # drop WITH_OCR=0 if you want OCR
```

`scripts/setup_ec2.sh` creates `.venv`, installs `requirements.txt`, clones (or
fast-forwards) the **external counting engine**, installs the engine's own
requirements, writes `wagoneye.env`, and finishes with preflight. Every path is
an environment variable with a default relative to `$HOME` or the repo —
nothing is hardcoded:

| Variable | Default | Meaning |
|---|---|---|
| `ENGINE_DIR` | `~/global_count_ec2` | where the counting engine is cloned |
| `RECON_MODELS_DIR` | `~/global_wagon_models` | the five counting weights |
| `FEAT_MODELS_DIR` | `<repo>/models/features` | the feature weights |
| `VENV_DIR` | `<repo>/.venv` | virtualenv |
| `WITH_OCR` | `1` | `0` skips the heavy `easyocr` install |

At runtime the same locations are read from the environment, so the run command
stays short:

```bash
source wagoneye.env
#   GLOBAL_WAGON_APP_DIR
#   WAGONEYE_RECON_MODELS_DIR
#   WAGONEYE_FEAT_MODELS_DIR
#   WAGONEYE_STAGE1_ENGINE      (optional; wagon_count to roll back)
```

`GLOBAL_WAGON_APP_DIR` may point at the clone **or** at the `global_wagon_app/`
package inside it, and `~` / `$VARS` are expanded. Per-run overrides:
`--global-engine-dir`, `--recon-models-dir`, `--feat-models-dir`.

### 2. Weights

Weights are never committed. These are the filenames the code actually opens —
if your copy spells them differently, **rename on download**; do not duplicate
weights into several folders.

Stage 1, in `$RECON_MODELS_DIR` (`global_counting/runner.py: MODEL_SLOTS`):

| Slot | Canonical name | Also accepted |
|---|---|---|
| side classification | `side_classification.pt` | `side_classify.pt`, `side.pt` |
| top classification | `top_classification.pt` | `top_classify.pt`, `top.pt` |
| right gap | `right_up_wagon_gap.pt` | `right_up_gap.pt`, `right_gap.pt`, `right_wagon_gap.pt` |
| left gap | `left_up_wagon_gap.pt` | `left_up_gap.pt`, `left_gap.pt`, `left_wagon_gap.pt` |
| top gap | `top_gap.pt` | `top_up_gap.pt`, `top_wagon_gap.pt` |

The canonical column is `core/constants.py`; the aliases exist because both
spellings occur in real model drops. Preflight prints which file matched.

Stage 3, in `$FEAT_MODELS_DIR` — exact names, no aliases:

| Feature | Filename | Constant |
|---|---|---|
| door | `door_state.pt` | `C.MODEL_DOOR_STATE` |
| load | `loaded.pt` | `C.MODEL_LOADED` |
| damage | `damage.pt` | `C.MODEL_DAMAGE` |
| ocr | `wagon_id_counting.pt` | `C.MODEL_WAGON_ID_COUNTING` |

### 3. Videos

Four videos, one per camera, in one folder. A filename must contain one of its
camera's accepted spellings — the two TOP cameras also accept the CCTV
exporter's `RIGHT_TOP` / `LEFT_TOP` form
(`core/constants.py: CAMERA_FILENAME_ALIASES`). Matching is longest-alias-first,
so `RIGHT_UP` never captures `RIGHT_UP_TOP`:

```
camera_..._2_RIGHT_UP_...mp4     -> RIGHT_UP
camera_..._1_LEFT_UP_...mp4      -> LEFT_UP
camera_..._5_RIGHT_TOP_...mp4    -> RIGHT_UP_TOP
camera_..._6_LEFT_TOP_...mp4     -> LEFT_UP_TOP
```

S3 input lives in `$WAGONEYE_S3_INPUT_BUCKET` (default: the trimmed-clip
bucket `biputri-wagon-pre-processed-video`), under the four camera folders in
`core/constants.py: CAMERA_S3_FOLDER`. `--auto` polls it; `--historical`
replays a time range out of it.

### 4. Preflight

Checks interpreter, wheels, the engine, all weights, the four videos and the
workspace **without loading a model or decoding a frame**. Exit 0 = ready.

```bash
python scripts/preflight.py     --local-inputs ~/wagon_eye_inputs/fresh_train     --recon-models-dir ~/global_wagon_models     --features door,load,damage
```

### 5. Run

Long CPU runs must survive a dropped SSH session:

```bash
tmux new -s wagoneye

cd ~/wagon_eye_global_integrated
source .venv/bin/activate
source wagoneye.env

python -m orchestrator.master_runner     --local-only     --local-inputs ~/wagon_eye_inputs/fresh_train     --recon-models-dir ~/global_wagon_models     --feat-models-dir ~/wagon_eye_global_integrated/models/features     --features door,load,damage     --no-interactive     --skip-upload --skip-email     2>&1 | tee ~/wagoneye_run.log

# detach: Ctrl-b d      reattach: tmux attach -t wagoneye
```

`--features door,load,damage` skips OCR entirely: `features.ocr.processor` is
never imported, EasyOCR is never initialized, `wagon_id_counting.pt` is never
loaded, and OCR is never submitted to the Stage-3 executor. Fusion and the
reports mark it `DISABLED BY USER` through the pipeline's existing sentinel
path, and the PDF is still produced. OCR stays available with `--features all`
or `--features ocr`.

### 6. Publish — only after inspecting the local output

```bash
BATCH=$(ls -1t batch_outputs | head -1)
aws s3 cp batch_outputs/$BATCH/reports/     s3://$WAGONEYE_S3_OUTPUT_BUCKET/train_batch/$BATCH/reports/ --recursive     --exclude "*" --include "*.pdf" --include "*.json"
```

Or drop `--skip-upload` to let Stage 6 deliver.

## Other run modes

```bash
python -m orchestrator.master_runner --auto                    # S3 polling
python -m orchestrator.master_runner --batch 20260408_032134   # replay
python -m orchestrator.master_runner --stage1-engine wagon_count ...  # rollback
```

## Tests

```bash
cd wagon_eye_v4
python -m pytest -q                      # 353 tests (v4 + counting engine)
python -m unittest discover -s tests     # 70 tests (v4 only, stdlib runner)
```

No model weights, no video decode and no GPU are needed: the counting modules
are pure stdlib, and the suites drive them with synthetic tracker output.
Nothing here runs production inference.

| Suite | Covers |
|---|---|
| `tests/test_counting_engine_swap.py` | the new engine is wired in; the old one is unreachable; no aliasing/download logic |
| `tests/test_global_roster_contract.py` | deterministic `GW_1..GW_N`, no duplicates/gaps, count independent of support cameras |
| `tests/test_roster_immutability.py` | inspection cannot append / remove / reorder / renumber / re-time the roster |
| `tests/test_downstream_contract.py` | door/load/damage/OCR + fusion + reporting still receive the expected structure |
| `tests/test_counting_integration.py` | four-camera input → engine → JSON → adapter → inspection; camera offsets reach materialization |
| `wagon_count/tests/` | the counting engine's own 283-test suite, adopted with it |

Tests that compare the engine against its source folder skip cleanly when
`wagon_count - Copy_correct_count/` is absent (it is gitignored).

## Run-mode flags

| Flag                        | Effect                                                |
|-----------------------------|-------------------------------------------------------|
| `--auto`                    | Continuous S3 polling.                                |
| `--once`                    | Process one batch then exit.                          |
| `--batch <key>`             | Replay / debug a specific batch_key.                  |
| `--local-only`              | Skip S3; use `--local-inputs` instead.                |
| `--local-inputs DIR`        | Folder to scan for the 4 videos.                      |
| `--workspace DIR`           | Output root (default: `./batch_outputs`).             |
| `--recon-models-dir DIR`    | Override `models/reconstruction/`.                    |
| `--feat-models-dir DIR`     | Override `models/features/`.                          |
| `--features LIST`           | Features to RUN: `all` or e.g. `door,load,damage`. Inverse of `--disable-features`; an unselected feature is never imported. |
| `--global-engine-dir DIR`   | Path to the external `global_wagon_app` engine. Overrides `$GLOBAL_WAGON_APP_DIR`. |
| `--stage1-engine NAME`      | `global_wagon_app` (default) or `wagon_count` (rollback). |
| `--skip-upload`             | Don't upload PDF/JSON, don't archive to S3.           |
| `--skip-email`              | Don't send the combined email.                        |
| `--poll-interval N`         | Continuous-mode S3 poll interval (seconds).           |
| `--partial-wait N`          | Wait this many minutes for missing cameras.           |
| `--disable-features LIST`   | Comma-separated feature keys to turn OFF (`door,ocr,load,damage`); skips the interactive prompt. |
| `--no-interactive`          | Never prompt for feature config (force all features ON unless `--disable-features` is given). |

## Feature configuration (enable/disable)

Before Stage 3 runs, you can choose which feature processors execute. The
feature set is **registry-driven** (`core/feature_config.py`) — adding a
feature there makes it appear here automatically.

Three ways to configure, in precedence order:

1. **CLI (non-interactive, scriptable):**
   ```bash
   python -m orchestrator.master_runner --local-only --local-inputs ./local_inputs \
          --disable-features ocr,damage
   ```
2. **Interactive prompt** — shown automatically on foreground runs
   (`--local-only` / `--once` / `--batch`) **only when stdin is a real
   terminal**:
   ```
   Current Feature Configuration:
     [ON]  Door
     [ON]  OCR
     [ON]  Load
     [ON]  Damage
   Turn OFF any feature? (y/n): y

   Select feature(s) to turn OFF (comma-separated numbers, e.g. 2,4):
     1. Door
     2. OCR
     3. Load
     4. Damage
   Disable: 2,4

   Final Feature Configuration:
     [ON]  Door
     [OFF] OCR
     [ON]  Load
     [OFF] Damage
   ```
3. **Default** — every feature ON. Continuous `--auto` polling, piped, and
   cron runs **never prompt** (safe for unattended operation); pass
   `--disable-features` to change the set there.

A disabled feature is skipped in Stage 3 and its per-wagon state is marked
`DISABLED_BY_USER`. In the camera-wise and combined reports its fields read
**`DISABLED BY USER`** (never flagged as an anomaly), and its overlay is
never drawn in the processed videos.

## Failure handling

| Failure                                  | Outcome                                            |
|------------------------------------------|----------------------------------------------------|
| Stage 1 reconstruction errors / 0 wagons | Batch marked `failed_no_global_state`. Abort.      |
| Stage 1 required counting model missing  | Engine exits non-zero naming the exact expected path. Batch marked `failed_no_global_state`. Abort. |
| Stage 1 malformed roster (duplicate / non-contiguous `GW_n`, count mismatch) | Rejected by `verify_roster_integrity()` at the Stage-1 boundary. Batch marked `failed_no_global_state`. Abort. |
| Inspection mutates the roster            | `RosterImmutabilityError` at the next stage guard. Loud failure, never silent corruption. |
| Stage 2 cannot open one video            | That camera's wagon_cache subtree empty. Continue. Batch is `completed_partial`. |
| Stage 3 one feature processor crashes    | Its per-wagon JSONs marked `FAILED`. Other features continue. |
| Stage 3 one wagon fails in one feature   | That `(feature, GW_n)` JSON marked `FAILED`. Rest unaffected. |
| Stage 4 fusion error per wagon           | Unified state for that wagon partial. Report still generated. |
| Stage 4b overlay renderer fails per cam  | That camera's mp4 missing; other cameras continue. Combined PDF still generated. |
| Stage 5a one camera report crashes       | That camera's PDF missing; other 3 camera reports + combined report unaffected. |
| Stage 5b combined PDF crashes            | Batch marked `report_failed`. JSON still written.  |
| PDF microservice down                    | S3 direct-upload fallback URL used.                |
| Email API down                           | Logged; batch outcome still persisted.             |

## Authority rules (fusion)

| Field                  | Authority                       |
|------------------------|---------------------------------|
| classification         | GlobalTrainState (RIGHT_UP)     |
| wagon_identifier (OCR) | RIGHT_UP only                   |
| right_door             | RIGHT_UP                        |
| left_door              | LEFT_UP                         |
| load_status            | RIGHT_UP_TOP (LEFT_UP_TOP fall) |
| top_damage             | any TOP camera reporting DAMAGE |

## Counting authority (Stage 1)

`wagon_count/` is the **only** component allowed to detect gaps, count, assign
wagon ids or define wagon boundaries.  It runs fixed-master fusion:

| Property | Guarantee |
|---|---|
| Global gap sequence | **exactly** the validated RIGHT_UP gaps |
| Support cameras | evidence + synchronization only; they can never create, delete, split or merge a global gap |
| Raw YOLO gap | a *candidate* — must pass motion / persistence / trajectory / duplicate validation |
| Camera clocks | per-camera offset estimated (`t_global = t_local + delta`), `RESOLVED` or `UNRESOLVED`; unresolved contributes 0.0, i.e. the shared-`t=0` assumption |
| Roster contents | **WAGON units only.**  ENGINE and BRAKE_VAN are preserved as `wagon_window` metadata but never receive a `GW_n` id and never extend the wagon timeline |
| Immutability | the roster is frozen the moment Stage 1 returns; `orchestrator` re-checks `roster_fingerprint()` after Stages 2, 3, 4 and 5 |

`core/global_state_loader.py` is the single adapter between this engine and
everything downstream — nothing else parses the counting engine's JSON.

### What "WAGON units only" changes in the output

`total_wagons` now counts **wagons**, not wagons + loco + brake van, so it
reads lower than a pre-swap run on the same train. Where the engine/brake-van
counts surface:

| Consumer | Value | Source |
|---|---|---|
| `GlobalTrainState.engine_count` / `.brake_van_count` | preserved | adapter reads `wagon_window` structure metadata |
| `reports/combined_train_report.json` → `legacy_view_model.summary_kpis.engine_count` | preserved | via the adapter above |
| `reports/combined_train_report.json` → `summary.engine_count` | **0** | `summarize_wagons()` counts over the roster, which is wagon-only |
| PDF "LOCO NUMBER" column | `Not Detected` | reads `summary_kpis.loco_numbers`, hard-coded `[]` in `_adapter.py` — unchanged by this swap, it behaved this way before |

The `summary` / `legacy_view_model` disagreement is pre-existing plumbing, left
as-is deliberately: reconciling it would mean editing the inspection/report
layer, which is out of scope for a counting-engine swap.

A camera that cannot see a given global wagon contributes **no frames** for it
(rather than clamping onto an unrelated frame); the feature processors then
report that camera's existing `NO_FRAMES` status.  No camera ever invents a
local wagon number.

## Constraints honored

- ❌ No `DoorProcessor.process_video()` or `DamageProcessor.process_video()`
  wrappers anywhere.
- ❌ No mini-mp4 reconstruction.
- ❌ No `cv2.VideoCapture` for **inference** outside Stage 2 (materializer)
  and Stage 1 (wagon_count subprocess).  Stage 4b's overlay renderer uses
  `cv2.VideoCapture` strictly for **visualization** — it never invokes any
  detector / YOLO / OCR / tracking model; everything it draws comes from
  already-persisted `GlobalTrainState`, `UnifiedWagonState`, evidence
  metadata, and `per_camera_tracking.json`.
- ❌ No per-camera folders (`RIGHT_UP/` etc.) in this package.
- ❌ No `wagon_gap.pt` recompute downstream of Stage 1.
- ✅ All models in a single centralized tree (`models/{reconstruction,features}/`).
- ✅ Frames extracted exactly once.
- ✅ GlobalTrainState is the immutable backbone — enforced, not just intended
  (frozen wagons, tuple roster, per-stage fingerprint guard).
- ❌ No per-camera independent wagon count is ever the final authority.
