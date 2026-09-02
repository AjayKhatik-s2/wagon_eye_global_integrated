#!/usr/bin/env bash
# RUN 5 -- Sequential WITH delivery. Validates the ORDERING of early camera
# delivery only. Its artifacts are deliberately NOT compared against Batch:
# Batch never passes evidence_url_base, so it cannot emit doors[].s3_url,
# wagon_frames or problem_frames, while a delivering Sequential run does.
set -u
cd ~/wagon_eye_global_integrated || exit 2
source .venv/bin/activate
export GLOBAL_WAGON_APP_DIR=$HOME/global_count_ec2
export WAGONEYE_ENGINE_DIR=$HOME/global_count_ec2

echo "=== endpoints -- MUST be UAT-only ==="
python -c "
from delivery import dashboard_ingest as D
from delivery import global_train_webhook as W
u = D.ingest_api_urls(); g = W.global_ingest_urls()
print('per-camera:', u); print('global    :', g)
bad = [x for x in u + g if 'uat' not in x]
raise SystemExit('REFUSING: non-UAT endpoint %s' % bad) if bad else None" || exit 2

python -m orchestrator.master_runner --local-only --mode sequential \
  --local-inputs batch_outputs/historical/20260726_093321/downloads \
  --batch SEQ_DELIVER --workspace $HOME/val5_ws \
  --recon-models-dir $HOME/global_wagon_models \
  --feat-models-dir models/features --features door,load,damage \
  --no-interactive --skip-email > ~/val_seq_deliver.log 2>&1
echo "exit=$? at $(date -u)"

echo "=== DELIVERY ORDERING (line numbers must INTERLEAVE) ==="
grep -an "Camera .* SEALED\|DELIVERED EARLY\|ALL REQUIRED\|ASSEMBLY COMPLETE\|combined JSON\|combined PDF\|DELIVERY [0-9]/3\|ingested  run_id\|GLOBAL-INGEST.*->" \
  ~/val_seq_deliver.log
