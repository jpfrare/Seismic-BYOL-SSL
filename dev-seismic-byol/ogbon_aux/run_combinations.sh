#!/bin/bash

# Usage check
if [ "$#" -lt 4 ]; then
  echo "Usage: $0 <start_rep> <end_rep> <gpu1> [<gpu2> ...] -- <dataset1> [<dataset2> ...] [--num_epochs <value>]"
  echo "Example: $0 0 4 0 1 -- f3 f3_N --num_epochs 300"
  exit 1
fi

# Fixed parameters
BATCH_SIZE=32
INPUT_SIZE=256
LR=0.2
NUM_EPOCHS=500  # Default

# Parse repetition range
START_REP=$1
END_REP=$2
shift 2

# Parse GPU list (up to "--")
GPUS=()
while [[ "$1" != "--" && "$#" -gt 0 ]]; do
  GPUS+=("$1")
  shift
done

# Remove "--"
shift

# Now parse datasets until we reach optional --num_epochs
DATASETS=()
while [[ "$#" -gt 0 ]]; do
  if [[ "$1" == "--num_epochs" ]]; then
    shift
    if [[ "$#" -eq 0 ]]; then
      echo "Error: --num_epochs requires a value"
      exit 1
    fi
    NUM_EPOCHS="$1"
    shift
  else
    DATASETS+=("$1")
    shift
  fi
done

# Loop over datasets and repetitions
for DATASET in "${DATASETS[@]}"; do
  for REP in $(seq "$START_REP" "$END_REP"); do
    echo "Running: dataset=$DATASET | repetition=$REP | gpus=${GPUS[*]} | epochs=$NUM_EPOCHS"
    python cli_pretrain.py \
      --batch_size "$BATCH_SIZE" \
      --input_size "$INPUT_SIZE" \
      --learning_rate "$LR" \
      --gpus "${GPUS[@]}" \
      --num_epochs "$NUM_EPOCHS" \
      --dataset_name "$DATASET" \
      --repetition "$REP"
  done
done
