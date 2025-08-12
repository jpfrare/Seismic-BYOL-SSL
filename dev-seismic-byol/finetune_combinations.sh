#!/bin/bash

# Usage information
if [ "$#" -lt 6 ]; then
  echo "Usage: $0 --rep <start> <end> --pre <pretrain1> [...] --finetune <finetune1> [...] [--caps <cap1> ...] [--gpus <gpu1> ...] [--freeze]"
  exit 1
fi

# Default parameters
BATCH_SIZE=8
FREEZE=False
LR=0.001
NUM_EPOCHS=6500
# parihaka size: 1120 com batch 8 = 140 batches por 50 épocas = 7000 passos
GPUS=(0)  # default GPU list

# Parse arguments
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

# Show parsed arguments
echo "------------------"
echo "Parsed Arguments:"
echo "Repetitions: $START_REP to $END_REP"
echo "Pretrain datasets: ${PRETRAIN_DATASETS[*]}"
echo "Finetune datasets: ${FINETUNE_DATASETS[*]}"
echo "Caps: ${CAPS[*]:-None (default=1.0)}"
echo "GPUs: ${GPUS[*]}"
echo "Freeze: $FREEZE"
echo "Batch size: $BATCH_SIZE"
echo "Learning rate: $LR"
echo "Epochs: $NUM_EPOCHS"
echo "------------------"

# Define listas válidas
VALID_PRETRAIN=("f3" "f3_N" "seam_ai" "seam_ai_N" "both" "both_N" "s0" "a700" "imagenet" "coco" "sup" "seg")
VALID_FINETUNE=("f3" "f3_N" "seam_ai" "seam_ai_N")

# Verifica datasets de pretreinamento
for dataset in "${PRETRAIN_DATASETS[@]}"; do
  if [[ ! " ${VALID_PRETRAIN[*]} " =~ " ${dataset} " ]]; then
    echo " Erro: Pretrain dataset inválido: '${dataset}'."
    echo " Válidos: ${VALID_PRETRAIN[*]}"
    exit 1
  fi
done

# Verifica datasets de fine-tuning
for dataset in "${FINETUNE_DATASETS[@]}"; do
  if [[ ! " ${VALID_FINETUNE[*]} " =~ " ${dataset} " ]]; then
    echo " Erro: Finetune dataset inválido: '${dataset}'."
    echo " Válidos: ${VALID_FINETUNE[*]}"
    exit 1
  fi
done

# Run loop: for each rep, finetune, pretrain, cap
for REP in $(seq "$START_REP" "$END_REP"); do
  for PRE in "${PRETRAIN_DATASETS[@]}"; do
    for FINETUNE in "${FINETUNE_DATASETS[@]}"; do
      for CAP in "${CAPS[@]:-1.0}"; do
        echo "Running: rep=$REP | cap=$CAP | finetune=$FINETUNE | pretrain=$PRE | gpus=${GPUS[*]}"
        
        CMD=(
          python cli_finetune_linear.py
          --pretrain_data "$PRE"
          --finetune_data "$FINETUNE"
          --num_epochs "$NUM_EPOCHS"
          --batch_size "$BATCH_SIZE"
          --repetition "$REP"
          --learning_rate "$LR"
          --cap "$CAP"
          --gpus "${GPUS[@]}"
          --steps
          --linear
        )

        # CMD=(
        #   python cli_finetune.py
        #   --pretrain_data "$PRE"
        #   --finetune_data "$FINETUNE"
        #   --num_epochs "$NUM_EPOCHS"
        #   --batch_size "$BATCH_SIZE"
        #   --repetition "$REP"
        #   --learning_rate "$LR"
        #   --cap "$CAP"
        #   --gpus "${GPUS[@]}"
        #   --steps
        # )

        if [ "$FREEZE" = True ]; then
          CMD+=(--freeze)
        fi

        "${CMD[@]}"
      done
    done
  done
done
