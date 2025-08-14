#!/bin/bash

# Usage check
if [ "$#" -lt 3 ]; then
  echo "Usage: $0 <start_rep> <end_rep> <dataset1> [<dataset2> ...] [--gpus <gpus>] [--num_epochs <num_epochs>]"
  echo "Example: $0 0 4 f3 f3_N --gpus 0,1 --num_epochs 100"
  exit 1
fi

# Default parameters
BATCH_SIZE=32
INPUT_SIZE=256
LR=1e-5
GPUS=""
NUM_EPOCHS=200

# Parse positional args
START_REP=$1
END_REP=$2
shift 2  # remove first two args

# Prepare array for datasets
DATASETS=()

# Parse optional flags and datasets separately
while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --gpus)
      GPUS=$2
      shift 2
      ;;
    --num_epochs)
      NUM_EPOCHS=$2
      shift 2
      ;;
    --*)
      echo "Unknown option: $1"
      exit 1
      ;;
    *)
      # Anything else is a dataset name
      DATASETS+=("$1")
      shift
      ;;
  esac
done

# Echo parameters received
echo "Parameters received:"
echo "  Start repetition: $START_REP"
echo "  End repetition: $END_REP"
echo "  Datasets: ${DATASETS[*]}"
echo "  GPUs: ${GPUS:-<none>}"
echo "  Number of epochs: $NUM_EPOCHS"
echo "  Batch size: $BATCH_SIZE"
echo "  Input size: $INPUT_SIZE"
echo "  Learning rate: $LR"
echo ""

# Check if datasets were provided
if [ "${#DATASETS[@]}" -eq 0 ]; then
  echo "Error: No datasets specified"
  exit 1
fi

# Function to find a free port between 29500 and 30000
find_free_port() {
  for port in $(seq 29500 30000); do
    if ! ss -ltn | grep -q ":$port "; then
      echo "$port"
      return 0
    fi
  done
  echo "No free port found" >&2
  return 1
}

# Loop over datasets and repetitions
for DATASET in "${DATASETS[@]}"; do
  for REP in $(seq "$START_REP" "$END_REP"); do
    FREE_PORT=$(find_free_port)
    if [ $? -ne 0 ]; then
      echo "Failed to find a free port, exiting."
      exit 1
    fi

    echo "Running: dataset=$DATASET | repetition=$REP | gpus=$GPUS | num_epochs=$NUM_EPOCHS | master_port=$FREE_PORT"

    MASTER_PORT=0 python cli_pretrain.py \
      --batch_size "$BATCH_SIZE" \
      --input_size "$INPUT_SIZE" \
      --learning_rate "$LR" \
      --num_epochs "$NUM_EPOCHS" \
      --dataset_name "$DATASET" \
      --repetition "$REP" \
      ${GPUS:+--gpus "$GPUS"}
  done
done
