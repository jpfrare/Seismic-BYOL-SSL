#!/bin/bash

# --------------------------
# Configuration
# --------------------------
SCRIPT_PATH="/petrobr/parceirosbr/home/joao.frare/workspace/spfm/Seismic-Byol/dev-seismic-byol/cli_pretrain.py"
WORKSPACE="/petrobr/parceirosbr/home/joao.frare/workspace"
export SIF="/petrobr/parceirosbr/spfm/singularity/amd64/deeprock/ngc/MINERVA_v0_3_9-beta-SPINN_v0_0_1.sif"

FLAGS=(
  "--repetition 0 --input_size 256 --batch_size 32 --dataset_name seam_ai_N --num_epochs 125000"
  "--repetition 2 --input_size 256 --batch_size 32 --dataset_name both_N --num_epochs 125000"
)

mkdir -p jobs_out

for f in "${FLAGS[@]}"; do
  sbatch <<EOT
#!/bin/bash
#SBATCH --job-name=pretrain_job
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1       
#SBATCH --partition=ict-h100
#SBATCH --account=spfm
#SBATCH --time=24:00:00
#SBATCH --output=jobs_out/pretrain_%j.out
#SBATCH --error=jobs_out/pretrain_%j.err

cd "\$SLURM_SUBMIT_DIR"

echo "Allocated nodes: \$SLURM_JOB_NODELIST"
echo "Running flags: $f"
nvidia-smi

singularity exec --nv \
    --bind "$WORKSPACE":"$WORKSPACE" \
    --bind /petrobr/parceirosbr/home/vinicius.soares/workspace:/petrobr/parceirosbr/home/vinicius.soares/workspace \
    --bind /petrobr/parceirosbr/spfm \
    "$SIF" \
    bash -c "
        export PYTHONPATH=/petrobr/parceirosbr/home/joao.frare/workspace/spfm/Seismic-Byol/Minerva-dev:\$PYTHONPATH
        python3 /petrobr/parceirosbr/home/joao.frare/workspace/spfm/Seismic-Byol/dev-seismic-byol/cli_pretrain.py $f
    "

EOT
done