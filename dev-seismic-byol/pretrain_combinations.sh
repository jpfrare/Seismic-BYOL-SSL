# Usage check
if [ "$#" -lt 3 ]; then
  echo "Usage: $0 <start_rep> <end_rep> <dataset1> [<dataset2> ...]"
  echo "Example: $0 0 4 f3 f3_N"
  exit 1
fi

# Fixed parameters
BATCH_SIZE=54
INPUT_SIZE=256
LR=1e-4
GPUS=3
NUM_EPOCHS=500

# Repetition range
START_REP=$1
END_REP=$2
shift 2  # Remove first two arguments so $@ contains only dataset names

# Loop over datasets and repetitions
for DATASET in "$@"; do
  for REP in $(seq $START_REP $END_REP); do
    echo "Running: dataset=$DATASET | repetition=$REP"
    python cli_pretrain.py \
      --batch_size $BATCH_SIZE \
      --input_size $INPUT_SIZE \
      --learning_rate $LR \
      --gpus $GPUS \
      --num_epochs $NUM_EPOCHS \
      --dataset_name $DATASET \
      --repetition $REP
  done
done
