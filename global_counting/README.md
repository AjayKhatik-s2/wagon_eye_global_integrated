# `global_counting/` — the NEW Stage-1 backbone

Stage 1 is served by the validated external **`global_wagon_app`** engine.
Nothing downstream changed: the same `global_train_state.json` contract feeds
the same materializer, features, fusion, rendering and reporting that commit
`f3d2d81` shipped.

```
4 RAW CAMERAS
      │
      ▼
NEW global_wagon_app                    (validated, frozen — external repo)
  classification → wagon-region trimming → gap detection → gap tracking
  → unique gap confirmation → 0–1000 normalization
  → dynamic master-camera selection → cross-camera alignment
  → missing-gap recovery → GLOBAL GAP TIMELINE → GLOBAL WAGON TIMELINE
      │
      ▼
global_counting/runner.py       drives the engine, harvests its state
      │
      ▼
global_counting/adapter.py      pure transformation, no engine imports
      │
      ▼
global_train_state.json  +  per_camera_tracking.json     ← unchanged schema
      │
      ▼
core.global_state_loader → materializer → features → fusion → reporting
```

## Division of responsibility

| | Source of truth for |
|---|---|
| `global_wagon_app` | wagon identity, global alignment, wagon boundaries, per-camera intervals |
| this repository | feature intelligence, fusion, rendering, reporting, delivery |

The engine is treated as **frozen**. `runner.py` calls its stage functions in
the engine's own order and changes no threshold, no algorithm and no
configuration value except two artifact toggles
(`GENERATE_TRIM_DEBUG_VIDEO`, `GENERATE_GAP_ANNOTATED_VIDEO` → off), because
this pipeline owns frame extraction and reporting. The engine's snapshot
extraction, figures and PDF are skipped for the same reason.

`GLOBAL_WAGON_COUNT = GLOBAL_GAP_COUNT − 1`, and the engine's wagon *N* becomes
**`GW_N`** — the old pipeline's format, and the only wagon identity in the
system. No second numbering exists anywhere.

## Engine location

**The engine is a separate repository and is deliberately not vendored here**,
so the validated counting algorithm has exactly one source of truth. Install it
once:

```bash
git clone https://github.com/AjayKhatik-s2/global_count_ec2.git ~/global_count_ec2
export GLOBAL_WAGON_APP_DIR=~/global_count_ec2
```

`runner.locate_engine()` searches, in order:

| # | Source | Path |
|---|---|---|
| 1 | `--global-engine-dir` / `engine_dir=` | as given |
| 2 | `$GLOBAL_WAGON_APP_DIR` | as given |
| 3 | beside the repo | `<repo>/global_wagon_app`, `<repo>/global_count_ec2` |
| 4 | above the repo | `<repo>/../global_wagon_app`, `<repo>/../global_count_ec2` |
| 5 | home | `~/global_wagon_app`, `~/global_count_ec2` |

Two things every candidate tolerates, because both bite in real deployments:

* **`~` and `$VARS` are expanded.** `GLOBAL_WAGON_APP_DIR="~/global_count_ec2"`
  set in an env file or a systemd unit is never expanded by a shell.
* **A parent is accepted.** A candidate matches if it *is* the package or if it
  *contains* `global_wagon_app/`, so pointing at the clone works too.

Failure is loud, never silent: the error says `global wagon engine not found`,
lists every configured path it tried, and prints the exact clone/export
commands that fix it. `runner.engine_search_paths()` returns that same list, and
`scripts/preflight.py` prints it with the winner marked.

## Import isolation — the collision that matters

The engine ships top-level `reporting.py` and `models.py`. This repository has a
`reporting/` **package** and a `models/` **directory**. A naive import either
shadows ours or hands the engine ours.

`runner.engine_session()` therefore stashes every engine module name out of
`sys.modules`, prepends the engine directory to `sys.path` for the duration, and
on exit drops whatever the engine imported and restores ours exactly — including
the "was never imported" case. Nothing the engine builds is allowed to outlive
the session either: everything is harvested into plain dicts and lists first, so
no lazy attribute access can reach a module that is no longer importable.

## Two conventions the adapter handles

**1. Trimmed vs original frames.** The engine works on each camera's *trimmed*
wagon-region clip. The old contract's `start_frame_master` / `start_time` are on
the *original* master clock, and the materializer opens the *original* videos.
Every frame index is therefore shifted by that camera's crop start.

**2. Per-camera windows are explicit.** The old contract derived a camera's
window as `(start_time − delta) × local_fps`: one shared clock plus a constant
per-camera offset. The new engine aligns cameras with a **scale and a
direction**, and supports a fully **reversed** timeline — which no single delta
can express. Every wagon therefore carries an additive
`camera_frame_ranges` entry with the aligned window per camera. The materializer
prefers it and falls back to the old formula when it is absent, so a state
written by the retained counter still materializes exactly as before.

For a reversed camera the adapter deliberately reports
`camera_offsets[cam].status = REVERSED_NOT_APPLICABLE`, so the old offset
consumers stay on their safe `0.0` path while the frame ranges carry the truth.
Inventing a delta there would corrupt every consumer of that contract.

## Classification, and why `engine_count` can be 0

`classification` is filled by majority vote over the per-frame classification
the trimming stage **already produced** for the master camera — no model is
re-run and no new algorithm is introduced. The engine's vocabulary maps as
`engine→ENGINE`, `brakevan→BRAKE_VAN`, `wagon`/`wagon_loaded`→`WAGON`.

> The engine trims to the confirmed wagon region **before** counting, so the
> locomotive and brake van are outside the global wagon timeline by design and
> receive no GW id. This is the same behaviour the retained master-fixed counter
> had — `core/global_state_loader.py` documents it as "the counting engine emits
> a WAGON-only roster".

Their counts are therefore reported through `wagon_window`
(`leading_non_wagon_classes` / `trailing_non_wagon_classes`), which is exactly
the channel `GlobalTrainState.engine_count` and `.brake_van_count` already read.
The adapter counts a *run* of consecutive non-wagon frames as one object, so an
eight-second locomotive counts once. **The reporting layer needs no change**,
and Stage-1 counting was not altered to make a KPI look better.

## Rollback

The retained `wagon_count` backbone is untouched and still selectable:

```bash
export WAGONEYE_STAGE1_ENGINE=wagon_count          # whole process
python -m orchestrator.master_runner --stage1-engine wagon_count ...   # one run
```

Precedence is `engine=` argument → `$WAGONEYE_STAGE1_ENGINE` →
`ENGINE_DEFAULT` (the new engine). **Exactly one engine runs per batch** — they
are never both run and never compared at runtime. To make the retained counter
the permanent default again, set `ENGINE_DEFAULT = ENGINE_WAGON_COUNT` in
`reconstruction/runner.py`.

## Audit artifacts

Written under `<batch>/global_state/global_counting/global/`, listed in the
Stage-1 log, and never colliding with downstream outputs:

```
global_gap_timeline.csv        global_wagon_timeline.csv
camera_alignment_summary.csv   normalized_gap_timelines.csv
unmatched_extra_detections.csv
```

## Tests

```bash
python -m pytest tests/test_global_counting_integration.py -q   # A-L, no models
```
