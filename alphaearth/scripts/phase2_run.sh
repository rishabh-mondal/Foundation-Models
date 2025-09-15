#!/usr/bin/env bash
# phase2_run.sh — one-command runner for Phase-2 FasterRCNN + DINOv3 + FiLM with many freeze schedules

set -euo pipefail

# --------------------------
# User-configurable defaults (override via: VAR=value ./phase2_run.sh)
# --------------------------
PY="${PY:-python}"
CUDA_DEV="${CUDA_DEV:-1}"

SCRIPT="${SCRIPT:-/home/rishabh.mondal/Brick-Kilns-project/ijcai_2025_kilns/Foundation-Models/alphaearth/scripts/geocontrast_phase2_fasterrcnn_film_freeze.py}"
GEO_CKPT="${GEO_CKPT:-/home/rishabh.mondal/Brick-Kilns-project/ijcai_2025_kilns/Foundation-Models/checkpoints/dinov3_geocontrast_map_all_splits.pth}"

# Data / regions
TRAIN_REGION="${TRAIN_REGION:-pak_punjab}"
IN_REGION="${IN_REGION:-pak_punjab}"
OOR_REGIONS="${OOR_REGIONS:-bangladesh uttar_pradesh}"

# Optional explicit CSVs (leave empty to auto-scan split folders)
TRAIN_CSV="${TRAIN_CSV:-/home/rishabh.mondal/Brick-Kilns-project/ijcai_2025_kilns/data/iclr_2026_processed_data/final_data/pak_punjab/pak_punjab_train_per_image_aef.csv}"
VAL_CSV="${VAL_CSV:-/home/rishabh.mondal/Brick-Kilns-project/ijcai_2025_kilns/data/iclr_2026_processed_data/final_data/pak_punjab/pak_punjab_val_per_image_aef.csv}"

# Core training knobs
EPOCHS="${EPOCHS:-6}"
IMAGE_SIZE="${IMAGE_SIZE:-800}"
BATCH_SIZE="${BATCH_SIZE:-14}"
NUM_WORKERS="${NUM_WORKERS:-8}"

# LRs
BACKBONE_LR="${BACKBONE_LR:-1e-5}"
FILM_LR="${FILM_LR:-1e-4}"
HEAD_LR="${HEAD_LR:-1e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.04}"

# Freeze schedule knobs (used by some modes)
WARMUP_N="${WARMUP_N:-2}"            # for modes B/C/G
UNFREEZE_LAST="${UNFREEZE_LAST:-6}"  # for modes C/G
FREEZE_FROM="${FREEZE_FROM:-3}"      # for modes E/F/G (0-based epoch)

# Book-keeping
RESULTS_DIR="${RESULTS_DIR:-phase2_runs}"
DRY_RUN="${DRY_RUN:-0}"              # 1 = print command only
EXTRA_TAG="${EXTRA_TAG:-map-pak_punjab}"        # optional suffix for log/ckpt names

# Evaluation-only extras
DETECTOR_CKPT="${DETECTOR_CKPT:-}"   # path to trained detector .pth when MODE=EVAL
EVAL_CSV_MAP="${EVAL_CSV_MAP:-}"     # e.g. 'uttar_pradesh=/abs/UP_test.csv bangladesh=/abs/BD_test.csv'

# Which schedule? (A,B,C, D1,D2, E,F,G, H, EVAL)
MODE="${MODE:-A}"

# --------------------------
# Helpers
# --------------------------
ts() { date +"%Y%m%d_%H%M%S"; }
die() { echo "ERROR: $*" >&2; exit 1; }

check_paths() {
  [[ -f "$SCRIPT" ]]   || die "SCRIPT not found: $SCRIPT"
  [[ -f "$GEO_CKPT" ]] || die "GEO_CKPT not found: $GEO_CKPT"
}

base_flags() {
  # Common flags used by all modes
  local common=(
    "--geocontrast_ckpt" "$GEO_CKPT"
    "--num_classes" "4"
    "--image_size" "$IMAGE_SIZE"
    "--batch_size" "$BATCH_SIZE"
    "--num_workers" "$NUM_WORKERS"
    "--epochs" "$EPOCHS"
    "--backbone_lr" "$BACKBONE_LR"
    "--film_lr" "$FILM_LR"
    "--head_lr" "$HEAD_LR"
    "--weight_decay" "$WEIGHT_DECAY"
    "--train_region" "$TRAIN_REGION"
    "--in_region" "$IN_REGION"
    "--results_dir" "$RESULTS_DIR"
  )
  # Forward OOR regions if provided
  if [[ -n "${OOR_REGIONS}" ]]; then
    # shellcheck disable=SC2206
    local oor_arr=($OOR_REGIONS)
    common+=("--oor_regions")
    common+=("${oor_arr[@]}")
  fi
  # Optional CSVs
  if [[ -n "${TRAIN_CSV}" ]]; then common+=("--train_csv" "$TRAIN_CSV"); fi
  if [[ -n "${VAL_CSV}" ]];  then common+=("--val_csv"   "$VAL_CSV");  fi
  # Optional eval-only props
  if [[ -n "${DETECTOR_CKPT}" ]]; then common+=("--detector_ckpt" "$DETECTOR_CKPT"); fi
  if [[ -n "${EVAL_CSV_MAP}" ]]; then
    # shellcheck disable=SC2206
    local eval_pairs=($EVAL_CSV_MAP)
    common+=("--eval_csv_map")
    common+=("${eval_pairs[@]}")
  fi
  printf '%q ' "${common[@]}"
}

# Per-mode flags (freezing or eval-only)
freeze_flags_for_mode() {
  case "$MODE" in
    A)  printf '%s' "--freeze_backbone_for 0 --freeze_mode none --freeze_backbone_from -1" ;;
    B)  printf '%s' "--freeze_backbone_for $WARMUP_N --freeze_mode all --freeze_film_during_freeze" ;;
    C)  printf '%s' "--freeze_backbone_for $WARMUP_N --freeze_mode last_n --unfreeze_last_blocks $UNFREEZE_LAST --freeze_film_during_freeze" ;;
    D1) printf '%s' "--freeze_backbone_for 999 --freeze_mode all" ;;
    D2) printf '%s' "--freeze_backbone_for 999 --freeze_mode all --freeze_film_during_freeze" ;;
    E)  printf '%s' "--freeze_backbone_from $FREEZE_FROM" ;;
    F)  printf '%s' "--freeze_backbone_from $FREEZE_FROM --freeze_film_after" ;;
    G)  printf '%s' "--freeze_backbone_for $WARMUP_N --freeze_mode last_n --unfreeze_last_blocks $UNFREEZE_LAST --freeze_film_during_freeze --freeze_backbone_from $FREEZE_FROM" ;;
    H)  printf '%s' "--freeze_backbone_for 999 --freeze_mode all" ;;
    EVAL) # Evaluation only: no training, ignore freeze knobs
         printf '%s' "--eval_only"
         ;;
    *)  die "Unknown MODE: $MODE (expected one of: A,B,C,D1,D2,E,F,G,H,EVAL)";;
  esac
}

build_name() {
  local stamp; stamp="$(ts)"
  local mode_tag="${MODE}"
  local tag="${mode_tag}_e${EPOCHS}_img${IMAGE_SIZE}_bs${BATCH_SIZE}"
  if [[ -n "$EXTRA_TAG" ]]; then tag="${tag}_${EXTRA_TAG}"; fi
  echo "${stamp}_${tag}"
}

# --------------------------
# Run
# --------------------------
main() {
  check_paths

  local name; name="$(build_name)"
  local log="${RESULTS_DIR}/${name}.log"
  mkdir -p "$RESULTS_DIR"

  local cmd="CUDA_VISIBLE_DEVICES=${CUDA_DEV} nohup ${PY} -u ${SCRIPT} \
    $(base_flags) \
    $(freeze_flags_for_mode) \
    > ${log} 2>&1 &"

  echo "================================================================"
  echo " Phase-2 run"
  echo "----------------------------------------------------------------"
  echo " MODE           : ${MODE}"
  echo " SCRIPT         : ${SCRIPT}"
  echo " GEO_CKPT       : ${GEO_CKPT}"
  echo " TRAIN_REGION   : ${TRAIN_REGION}"
  echo " IN_REGION      : ${IN_REGION}"
  echo " OOR_REGIONS    : ${OOR_REGIONS}"
  echo " IMAGE_SIZE     : ${IMAGE_SIZE}"
  echo " BATCH_SIZE     : ${BATCH_SIZE}"
  echo " EPOCHS         : ${EPOCHS}"
  echo " LRs            : BB=${BACKBONE_LR}, FiLM=${FILM_LR}, Head=${HEAD_LR}"
  echo " Freeze knobs   : WARMUP_N=${WARMUP_N}, UNFREEZE_LAST=${UNFREEZE_LAST}, FREEZE_FROM=${FREEZE_FROM}"
  echo " DETECTOR_CKPT  : ${DETECTOR_CKPT:-<none>}"
  echo " EVAL_CSV_MAP   : ${EVAL_CSV_MAP:-<auto-resolve>}"
  echo " RESULTS_DIR    : ${RESULTS_DIR}"
  echo " LOG            : ${log}"
  echo "----------------------------------------------------------------"
  echo "$cmd"
  echo "================================================================"

  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "[DRY-RUN] Not executing."
    exit 0
  fi

  eval "$cmd"
  echo "Started. Tail the log with:"
  echo "  tail -f ${log}"
}

main "$@"

# --------------------------
# Examples
# --------------------------
# MODE=EVAL CUDA_DEV=1 DETECTOR_CKPT=/abs/model.pth EVAL_CSV_MAP="uttar_pradesh=./UP_test.csv bangladesh=./BD_test.csv pak_punjab=./PK_test.csv" ./phase2_run.sh
# MODE=A     CUDA_DEV=1 EPOCHS=10 ./phase2_run.sh
# MODE=B     WARMUP_N=2 CUDA_DEV=1 EPOCHS=10 ./phase2_run.sh
# MODE=C     WARMUP_N=2 UNFREEZE_LAST=6 CUDA_DEV=1 EPOCHS=10 ./phase2_run.sh
# MODE=D1    CUDA_DEV=1 EPOCHS=10 ./phase2_run.sh
# MODE=D2    CUDA_DEV=1 EPOCHS=10 ./phase2_run.sh
# MODE=E     FREEZE_FROM=3 CUDA_DEV=1 EPOCHS=10 ./phase2_run.sh
# MODE=F     FREEZE_FROM=3 CUDA_DEV=1 EPOCHS=6  ./phase2_run.sh
# MODE=G     WARMUP_N=2 UNFREEZE_LAST=6 FREEZE_FROM=7 CUDA_DEV=1 EPOCHS=10 ./phase2_run.sh
# MODE=H     CUDA_DEV=1 EPOCHS=10 ./phase2_run.sh
# DRY_RUN=1  MODE=C WARMUP_N=2 UNFREEZE_LAST=6 CUDA_DEV=1 EPOCHS=10 ./phase2_run.sh