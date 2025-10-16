#!/bin/bash

# --------------------------
# Configuration
# --------------------------
SCRIPT_PATH="/petrobr/parceirosbr/home/vinicius.soares/workspace/Seismic-Byol/dev-seismic-byol/cli_pretrain.py"
WORKSPACE="/petrobr/parceirosbr/home/vinicius.soares/workspace"
SIF="/petrobr/parceirosbr/home/vinicius.soares/singularity/minerva_dev.sif"

# Training flags
FLAGS=(
  "--repetition 4 --input_size 256 --batch_size 128 --dataset_name namss --num_epochs 125000"
  "--repetition 4 --input_size 256 --batch_size 128 --dataset_name a700 --num_epochs 125000"
)

# --------------------------
# Submit one job per flag set
# --------------------------
for f in "${FLAGS[@]}"; do
  sbatch <<EOT
#!/bin/bash
#SBATCH --job-name=finetune_job
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1       
#SBATCH --partition=ict-h100
#SBATCH --account=spfm
#SBATCH --time=24:00:00
#SBATCH --output=jobs_out/finetune_%j.out
#SBATCH --error=jobs_out/finetune_%j.err

cd "\$SLURM_SUBMIT_DIR"

echo "Allocated nodes: \$SLURM_JOB_NODELIST"
nvidia-smi

# --------------------------
# Run training in container
# --------------------------
singularity exec --nv \
    --bind "$WORKSPACE" \
    --bind /petrobr/parceirosbr/spfm \
    "$SIF" python "$SCRIPT_PATH" $f

EOT
done
