#!/bin/bash

# Verifica número mínimo de argumentos
if [ "$#" -lt 6 ]; then
  echo "Usage: $0 --rep <start> <end> --pre <pretrain1> [...] --finetune <finetune1> [...] [--caps <cap1> ...] [--gpus <gpu1> ...]"
  exit 1
fi

# Parâmetros fixos
BATCH_SIZE=8
FREEZE=False
LR=0.001
NUM_EPOCHS=50
GPUS=("0")  # padrão

# Parse dos argumentos
while [[ $# -gt 0 ]]; do
  key="$1"
  case $key in
    --rep)
      START_REP="$2"
      END_REP="$3"
      shift 3
      ;;
    --pre)
      PRETRAIN_DATASETS=()
      shift
      while [[ $# -gt 0 && "$1" != --* ]]; do
        PRETRAIN_DATASETS+=("$1")
        shift
      done
      ;;
    --finetune)
      FINETUNE_DATASETS=()
      shift
      while [[ $# -gt 0 && "$1" != --* ]]; do
        FINETUNE_DATASETS+=("$1")
        shift
      done
      ;;
    --caps)
      CAPS=()
      shift
      while [[ $# -gt 0 && "$1" != --* ]]; do
        CAPS+=("$1")
        shift
      done
      ;;
    --gpus)
      GPUS=()
      shift
      while [[ $# -gt 0 && "$1" != --* ]]; do
        GPUS+=("$1")
        shift
      done
      ;;
    --freeze)
      FREEZE=True
      shift
      ;;
    *)
      echo "Unknown argument: $1"
      exit 1
      ;;
  esac
done

# Função para obter o cap com fallback
get_cap_for_index() {
  local index=$1
  if [[ ${#CAPS[@]} -eq 0 ]]; then
    echo "1.0"
  elif [[ ${#CAPS[@]} -eq 1 ]]; then
    echo "${CAPS[0]}"
  else
    echo "${CAPS[$index]}"
  fi
}

# Loop para execução
for REP in $(seq "$START_REP" "$END_REP"); do
  for i in "${!FINETUNE_DATASETS[@]}"; do
    FINETUNE="${FINETUNE_DATASETS[$i]}"
    CAP=$(get_cap_for_index "$i")

    for PRE in "${PRETRAIN_DATASETS[@]}"; do
      for GPU in "${GPUS[@]}"; do
        echo "Running: rep=$REP | cap=$CAP | finetune=$FINETUNE | pretrain=$PRE | gpu=$GPU"
        if [ "$FREEZE" = True ]; then
          python cli_finetune.py \
            --pretrain_data "$PRE" \
            --finetune_data "$FINETUNE" \
            --num_epochs "$NUM_EPOCHS" \
            --batch_size "$BATCH_SIZE" \
            --repetition "$REP" \
            --learning_rate "$LR" \
            --cap "$CAP" \
            --freeze \
            --gpus "$GPU"
        else
          python cli_finetune.py \
            --pretrain_data "$PRE" \
            --finetune_data "$FINETUNE" \
            --num_epochs "$NUM_EPOCHS" \
            --batch_size "$BATCH_SIZE" \
            --repetition "$REP" \
            --learning_rate "$LR" \
            --cap "$CAP" \
            --gpus "$GPU"
        fi
      done
    done
  done
done
