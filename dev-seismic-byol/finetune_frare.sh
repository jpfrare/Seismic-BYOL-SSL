#!/bin/bash

# --------------------------
# Configuration
# --------------------------
SCRIPT_PATH="/petrobr/parceirosbr/home/joao.frare/workspace/spfm/Seismic-Byol/dev-seismic-byol/new_cli_finetune.py"
WORKSPACE="/petrobr/parceirosbr/home/joao.frare/workspace"
export SIF="/petrobr/parceirosbr/spfm/singularity/amd64/deeprock/ngc/MINERVA_v0_3_9-beta-SPINN_v0_0_1.sif"

FLAGS=(
  "--repetition 1 --input_size 256 --batch_size 32 --dataset_name seam_ai_N --num_epochs 125000"
  "--repetition 2 --input_size 256 --batch_size 32 --dataset_name both_N --num_epochs 125000"
)