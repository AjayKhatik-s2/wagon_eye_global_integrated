#!/usr/bin/env bash
# Five-run Sequential validation. EC2 only. Batch is never modified.
#
#   1 BATCH REFERENCE     --skip-upload
#   2 BATCH CONTROL       --skip-upload   (measures provenance empirically)
#   3 SEQUENTIAL NORMAL   --skip-upload   arrival: RU, LU, RUT, LUT
#   4 SEQUENTIAL PERMUTED --skip-upload   arrival: LUT, RU, LU, RUT
#   5 SEQUENTIAL DELIVER  (no --skip-upload)  delivery ordering only
#
# Runs 1-4 use --skip-upload on BOTH sides deliberately: Batch never passes
# evidence_url_base, so it structurally cannot emit doors[].s3_url,
# wagon_frames or problem_frames. Sequential emits them only when delivering.
# With upload suppressed both omit them and the artifacts are comparable.
# Run 5 therefore exists separately and its artifacts are NOT compared to Batch.
set -u
cd ~/wagon_eye_global_integrated || exit 2
source .venv/bin/activate

export GLOBAL_WAGON_APP_DIR=$HOME/global_count_ec2
export WAGONEYE_ENGINE_DIR=$HOME/global_count_ec2
RECON=$HOME/global_wagon_models
FEAT=models/features
WS=$HOME/val5_ws
IN=batch_outputs/historical/20260726_093321/downloads

echo "=== 0. COMMIT UNDER TEST ==="
git rev-parse HEAD; git log --oneline -3
echo

echo "=== 0b. INPUT CONTENT IDENTITY (sha256, before anything) ==="
sha256sum $IN/*.mp4 | tee /tmp/sha_before.txt
echo

echo "=== 0c. TEST SUITE, real engine ==="
python -m pytest -q 2>&1 | tail -3
echo "--- exact-parity suite ---"
python -m pytest tests/test_batch_sequential_exact_parity.py -q 2>&1 | tail -2
echo "--- arrival-order + report-format suites ---"
python -m pytest tests/test_sequential_arrival_order.py \
                tests/test_sequential_report_format.py \
                tests/test_sequential_camera_delivery.py -q 2>&1 | tail -2
echo

rm -rf "$WS"; mkdir -p "$WS"

# set_arrival <cam1> <cam2> <cam3> <cam4>  -- ascending mtimes in that order
set_arrival() {
  python - "$@" <<'PY'
import os, sys, glob
from core.batch import scan_local_video_dir
order = sys.argv[1:]
paths = scan_local_video_dir(os.environ["VAL_IN"])
base = 1_700_000_000
for i, cam in enumerate(order):
    p = paths[cam]
    os.utime(p, (base + i * 3600, base + i * 3600))
print("   arrival signal set to: %s" % ", ".join(order))
for cam in order:
    print("      %-13s mtime=%d  %s" % (cam, os.path.getmtime(paths[cam]),
                                        os.path.basename(paths[cam])))
PY
}
export VAL_IN="$IN"

COMMON_BATCH="--local-only --mode batch --local-inputs $IN --workspace $WS \
 --recon-models-dir $RECON --feat-models-dir $FEAT --features door,load,damage \
 --no-interactive --skip-upload --skip-email"
COMMON_SEQ="--local-only --mode sequential --local-inputs $IN --workspace $WS \
 --recon-models-dir $RECON --feat-models-dir $FEAT --features door,load,damage \
 --no-interactive --skip-email"

echo "=== RUN 1: BATCH REFERENCE ==="; date -u
python -m orchestrator.master_runner $COMMON_BATCH --batch BATCH_REF \
  > ~/val_batch_ref.log 2>&1; echo "exit=$? at $(date -u)"
grep -a "wagons=\|STAGE 5b\|master camera" ~/val_batch_ref.log | tail -4
echo

echo "=== RUN 2: BATCH CONTROL ==="; date -u
python -m orchestrator.master_runner $COMMON_BATCH --batch BATCH_CTL \
  > ~/val_batch_ctl.log 2>&1; echo "exit=$? at $(date -u)"
echo

echo "=== RUN 3: SEQUENTIAL, NORMAL ARRIVAL ==="; date -u
set_arrival RIGHT_UP LEFT_UP RIGHT_UP_TOP LEFT_UP_TOP
python -m orchestrator.master_runner $COMMON_SEQ --skip-upload \
  --batch SEQ_NORM > ~/val_seq_norm.log 2>&1; echo "exit=$? at $(date -u)"
echo "--- ACTUAL PROCESSING ORDER (run 3) ---"
grep -a "Camera .* SEALED" ~/val_seq_norm.log | sed 's/.*Camera /   /'
echo

echo "=== RUN 4: SEQUENTIAL, PERMUTED ARRIVAL ==="; date -u
set_arrival LEFT_UP_TOP RIGHT_UP LEFT_UP RIGHT_UP_TOP
python -m orchestrator.master_runner $COMMON_SEQ --skip-upload \
  --batch SEQ_PERM > ~/val_seq_perm.log 2>&1; echo "exit=$? at $(date -u)"
echo "--- ACTUAL PROCESSING ORDER (run 4) ---"
grep -a "Camera .* SEALED" ~/val_seq_perm.log | sed 's/.*Camera /   /'
echo

echo "=== 0d. INPUT CONTENT UNCHANGED? (mtime moved, bytes must not) ==="
sha256sum $IN/*.mp4 > /tmp/sha_after.txt
if diff -q /tmp/sha_before.txt /tmp/sha_after.txt >/dev/null; then
  echo "   OK: all four videos byte-identical (only mtime changed)"
else
  echo "   !! CONTENT CHANGED -- permutation test invalid"; diff /tmp/sha_before.txt /tmp/sha_after.txt
fi
echo

echo "==================================================================="
echo " COMPARISON A: BATCH vs BATCH  (control -- measures provenance)"
echo "==================================================================="
python scripts/artifact_parity.py --batch "$WS/BATCH_REF" \
  --sequential "$WS/BATCH_CTL" \
  --batch-reports reports --sequential-reports reports --control
echo "control exit=$?"
echo

echo "==================================================================="
echo " COMPARISON B: BATCH vs SEQUENTIAL  -- global data (parity_diff)"
echo "==================================================================="
python scripts/parity_diff.py --batch "$WS/BATCH_REF" \
  --sequential "$WS/SEQ_NORM"
echo "parity_diff exit=$?"
echo

echo "==================================================================="
echo " COMPARISON C: BATCH vs SEQUENTIAL  -- rendered artifacts"
echo "==================================================================="
python scripts/artifact_parity.py --batch "$WS/BATCH_REF" \
  --sequential "$WS/SEQ_NORM"
echo "artifact_parity exit=$?"
echo

echo "==================================================================="
echo " COMPARISON D: SEQ_NORM vs SEQ_PERM  -- ORDER INDEPENDENCE"
echo "==================================================================="
python scripts/parity_diff.py --batch "$WS/SEQ_NORM" \
  --sequential "$WS/SEQ_PERM"
echo "parity_diff exit=$?"
python scripts/artifact_parity.py --batch "$WS/SEQ_NORM" \
  --sequential "$WS/SEQ_PERM" \
  --batch-reports combined --sequential-reports combined
echo "artifact_parity exit=$?"
echo

echo "==================================================================="
echo " CAMERA-WISE PHASE-1 REPORT FORMAT (camera-local, 4 cameras)"
echo "==================================================================="
python - <<'PY'
import glob, json, os
ws = os.path.expanduser("~/val5_ws")
for run in ("SEQ_NORM", "SEQ_PERM"):
    print("  %s:" % run)
    keys_seen = []
    for p in sorted(glob.glob(os.path.join(ws, run, "camera_reports",
                                           "*", "*_report.json"))):
        d = json.load(open(p))
        keys_seen.append(tuple(sorted(d)))
        print("     %-13s gaps=%-4s segments=%-4s canonical=%s type=%s"
              % (d.get("camera_id"),
                 (d.get("local_gaps") or {}).get("count"),
                 (d.get("local_segments") or {}).get("count"),
                 d.get("canonical"), d.get("report_type")))
        blob = json.dumps(d)
        assert '"GW_' not in blob, "canonical ID leaked into %s" % p
    if keys_seen:
        print("     identical key set across all four: %s"
              % (len(set(keys_seen)) == 1))
PY
echo

echo "==================================================================="
echo " RUNS 1-4 COMPLETE. Run 5 (delivery) is a SEPARATE, deliberate step:"
echo "   it posts to the dashboard, so it must not run unattended."
echo "   See ~/val_deliver.sh"
echo "==================================================================="
echo "=== DONE $(date -u) ==="
