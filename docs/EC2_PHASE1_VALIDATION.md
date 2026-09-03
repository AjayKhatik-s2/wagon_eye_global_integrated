# Phase-1 validation on EC2 — manual procedure

Everything below runs on the EC2 box. **Nothing in this document has been
executed by the person who wrote it** — the authoring environment has no access
to that machine. Treat every expected value as a prediction to check, not a
result.

The repository-level checks (test suite, orchestration, isolation, failure
paths) are already green; see the bottom section for exactly which of those used
real models and which used stubs.

---

## 1. Pull the commit

```bash
cd ~/wagon_eye_global_integrated
git fetch origin
git checkout historical-sequential
git pull
git rev-parse --short HEAD
git log --oneline -4
```

## 2. Environment

```bash
source .venv/bin/activate
export GLOBAL_WAGON_APP_DIR=$HOME/global_count_ec2
export WAGONEYE_ENGINE_DIR=$HOME/global_count_ec2
```

`WAGONEYE_ENGINE_DIR` matters: without it the parity suite silently skips its
engine-backed tests and still reports success.

## 3. Test suite, with the real engine visible

```bash
python -m pytest -q
python -m pytest tests/test_batch_sequential_exact_parity.py -q   # 0 skipped
python -m pytest tests/test_local_state_adapter.py \
                tests/test_phase1_features.py \
                tests/test_phase1_end_to_end.py -q -v
```

The last command's `-v` matters: `TestRealModels` **skips** where models are
absent. On EC2 it should not skip. A skip there means the weights are not where
the test looks, and the Phase-1 feature run would fail for the same reason.

## 4. Input

The 26 July run reclaimed its `downloads/` directories, so re-fetch one train:

```bash
D=batch_outputs/historical/20260726_093321/downloads
mkdir -p $D
B=s3://biputri-wagoneye-pre-processed-video
aws s3 cp $B/camera_CCTV_HZBN_DHN_2_RIGHT_UP/camera_CCTV_HZBN_DHN_2_RIGHT_UP_20260726_093503_train.mp4   $D/ --region ap-south-1
aws s3 cp $B/camera_CCTV_HZBN_DHN_1_LEFT_UP/camera_CCTV_HZBN_DHN_1_LEFT_UP_20260726_093321_train.mp4     $D/ --region ap-south-1
aws s3 cp $B/camera_CCTV_HZBN_DHN_5_RIGHT_TOP/camera_CCTV_HZBN_DHN_5_RIGHT_TOP_20260726_093335_train.mp4 $D/ --region ap-south-1
aws s3 cp $B/camera_CCTV_HZBN_DHN_6_LEFT_TOP/camera_CCTV_HZBN_DHN_6_LEFT_TOP_20260726_093457_train.mp4   $D/ --region ap-south-1

python -c "
from core.batch import scan_local_video_dir
d = scan_local_video_dir('$D')
for k, v in sorted(d.items()): print('%-13s %s' % (k, v.rsplit('/',1)[-1]))
print('cameras:', len(d))"
```

Must print **4**.

## 5. The run

```bash
nohup python -m orchestrator.master_runner \
  --local-only --mode sequential \
  --local-inputs $D \
  --batch P1_VALIDATE --workspace $HOME/p1_ws \
  --recon-models-dir $HOME/global_wagon_models \
  --feat-models-dir models/features \
  --features door,load,damage \
  --no-interactive --skip-upload --skip-email \
  > ~/p1_validate.log 2>&1 &

sleep 30; pgrep -af master_runner; head -20 ~/p1_validate.log
```

`--skip-upload` suppresses delivery, so this reaches no dashboard and no bucket.

Watch:

```bash
tail -f ~/p1_validate.log | grep -aE "SEQ/P1|Camera .* SEALED|ALL REQUIRED|ASSEMBLY"
```

If you stop watching, re-run `tail` — do not re-run the script.

---

## 6. Log lines that prove Phase 1 completed

For **each** camera, in arrival order, these must appear together and before the
next camera's block:

```
[SEQ/<CAM>] unique gaps=N  region=[a..b]  trimmed frames=N
[SEQ/P1/<CAM>] cache=…s (N frames) door=…s load=…s damage=…s fusion=…s  local wagons=N
[SEQ] Camera <CAM> SEALED
```

The middle line is the one that did not exist before this work. Its absence
means Phase 1 ran no features and the camera report is counting-only.

Then, once:

```
[SEQ] ALL REQUIRED CAMERAS SEALED  (…four cameras…)
[SEQ] GLOBAL ASSEMBLY COMPLETE  master=… gaps=N wagons=N
```

**Interleaving check** — the `[SEQ/P1/…]` blocks must be spread through the log,
one per camera, not clustered at the end:

```bash
grep -an "SEQ/P1/.*cache=\|Camera .* SEALED\|ALL REQUIRED" ~/p1_validate.log
```

## 7. Artifacts to inspect

```bash
W=$HOME/p1_ws/P1_VALIDATE

# Phase-1 reports -- four cameras, JSON + PDF each
ls -la $W/camera_reports/*/

# Phase-1 isolation: local ids live ONLY here
ls $W/camera_local/
ls $W/camera_local/RIGHT_UP/
ls $W/camera_local/RIGHT_UP/wagon_cache/ | head
ls $W/camera_local/RIGHT_UP/wagon_states/

# Canonical trees -- Phase 2's, must contain NO local ids
ls $W/wagon_cache/ | head
ls $W/combined/
```

**The PDFs are the deliverable.** Open one and confirm it is a full inspection
report — summary KPI page, detection summary, anomaly summary, evidence pages,
per-wagon pages — not a count/gap page:

```bash
ls -la $W/camera_reports/RIGHT_UP/RIGHT_UP_report.pdf
```

A Phase-1 PDF of a few KB is the old counting report. A feature-rich one with
embedded evidence images is megabytes.

## 8. Assertions worth running rather than eyeballing

```bash
python - <<'PY'
import glob, json, os
W = os.path.expanduser("~/p1_ws/P1_VALIDATE")

print("=== Phase-1 camera reports ===")
for p in sorted(glob.glob(os.path.join(W, "camera_reports", "*",
                                       "*_report.json"))):
    d = json.load(open(p))
    wagons = (d.get("inspection") or {}).get("wagons") or []
    det = sum(len(w.get("detections") or []) for w in wagons)
    labels = sorted({x.get("label") for w in wagons
                     for x in (w.get("detections") or [])})
    print("%-13s local_wagons=%-4d detections=%-5d labels=%s canonical=%s"
          % (d.get("camera_id"), len(wagons), det, labels, d.get("canonical")))
    blob = json.dumps(d)
    assert '"GW_' not in blob, "canonical id leaked into %s" % p
    for w in wagons:
        assert str(w.get("local_wagon_id", "")).startswith(
            d["camera_id"] + "_W"), w.get("local_wagon_id")

print()
print("=== isolation: no local id in the canonical trees ===")
for tree in ("wagon_cache", "wagon_states/door", "evidence"):
    base = os.path.join(W, tree)
    if not os.path.isdir(base):
        print("  %-20s absent" % tree); continue
    stray = [n for n in os.listdir(base) if "_W" in n and not n.startswith("GW_")]
    print("  %-20s entries=%-4d stray_local_ids=%s"
          % (tree, len(os.listdir(base)), stray))
    assert not stray, stray
print("\nOK")
PY
```

**`detections` must be non-zero** for at least the side cameras. Zero everywhere
means the features ran but their results did not reach the report — the exact
failure this work exists to fix.

`local_wagons` may differ between cameras. That is an observation, not an error.

## 9. Measurements for commit 5

```bash
grep -a "SEQ/P1/.*cache=" ~/p1_validate.log
du -sh $HOME/p1_ws/P1_VALIDATE/camera_local
du -sh $HOME/p1_ws/P1_VALIDATE/wagon_cache 2>/dev/null
du -sh $HOME/p1_ws/P1_VALIDATE
df -h /
grep -a "total\|elapsed" ~/p1_validate.log | tail -5
```

Report these verbatim. No runtime or disk figure in this repository is measured
until they come back.

## 10. What a failure looks like, and what to send

| Symptom | Meaning |
|---|---|
| no `[SEQ/P1/…]` line | Phase-1 features did not run |
| `local wagons=0` | fewer than two confirmed gaps; counting-only report is correct |
| `detections=0` everywhere | features ran, results did not reach the report |
| stray `_W` id in a canonical tree | isolation broken; would reach S3 |
| Phase-1 PDF only a few KB | fell back to the counting report |
| `TestRealModels` skipped on EC2 | weights are not where the code looks |

On any failure send `~/p1_validate.log` and the output of section 8. Do not
adjust the code to make a check pass.

---

## Appendix — which repository tests used real models

**REAL** (`tests/test_phase1_end_to_end.py::TestRealModels`) — asserts the
engine checkout exposes `camera_pipeline.py` and `global_alignment.py`, that
`door_state.pt` / `loaded.pt` / `damage.pt` exist, and that the real processor
modules import with the signatures Batch calls. **No model is executed**, and
each test SKIPS with a named reason where the files are absent rather than
passing vacuously.

**STUB** — everything else in `test_phase1_features.py` and
`test_phase1_end_to_end.py`. The models are replaced deliberately: those tests
cover ordering, isolation, sealing, and the failure paths, and a real model
cannot be made to raise, return nothing, or produce an unreadable state on
demand.

**Neither is production validation.** Executing the real weights over real video
end to end is what this document is for.
