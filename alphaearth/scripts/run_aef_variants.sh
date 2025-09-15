#!/bin/bash
set -euo pipefail
mkdir -p logs runs

export CUDA_VISIBLE_DEVICES=1,2,3
export PYTHONUNBUFFERED=1

EPOCHS=1
BATCH=64
SCRIPT=aef_frcnn.py

run_variant_bg () {
  variant="$1"
  size="$2"
  outdir="runs/${variant}_s${size}_e${EPOCHS}_b${BATCH}"
  logfile="logs/${variant}_s${size}_e${EPOCHS}_b${BATCH}.log"

  echo "[BG] ${variant} s${size} -> ${logfile}"
  nohup python -u "${SCRIPT}" \
    --variant "${variant}" \
    --train_region uttar_pradesh \
    --in_region uttar_pradesh \
    --oor_regions pak_punjab bangladesh \
    --image_size "${size}" \
    --batch_size "${BATCH}" \
    --epochs "${EPOCHS}" \
    --save_dir "${outdir}" \
    > "${logfile}" 2>&1 &
}

# Launch all in background (6 jobs total)
run_variant_bg head_only   128
run_variant_bg head_only   800
run_variant_bg thin_cnn    128
run_variant_bg thin_cnn    800
run_variant_bg resnet18_ae 128
run_variant_bg resnet18_ae 800

echo "[SUBMITTED] Use: tail -f logs/<name>.log"