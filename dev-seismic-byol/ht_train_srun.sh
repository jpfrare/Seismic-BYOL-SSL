#!/bin/bash

# --------------------------
# Configuration
# --------------------------
SCRIPT_PATH="/petrobr/parceirosbr/home/vinicius.soares/workspace/Seismic-Byol/dev-seismic-byol/ht_train.py"
WORKSPACE="/petrobr/parceirosbr/home/vinicius.soares/workspace"
SIF="/petrobr/parceirosbr/home/vinicius.soares/singularity/minerva_dev.sif"

# Training flags
FLAGS=(
  "--freeze --linear --gpus 0"
)

# --------------------------
# Submit one job per flag set
# --------------------------
for f in "${FLAGS[@]}"; do
  sbatch <<EOT
#!/bin/bash
#SBATCH --job-name=finetune_job
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4       
#SBATCH --partition=ict-h100
#SBATCH --account=spfm
#SBATCH --time=24:00:00
#SBATCH --output=jobs_out/finetune_%j.out
#SBATCH --error=jobs_out/finetune_%j.err

cd "\$SLURM_SUBMIT_DIR"

echo "Allocated nodes: \$SLURM_JOB_NODELIST"
nvidia-smi

# Detect network interface for NCCL
IFACE=\$(ip route get 8.8.8.8 2>/dev/null | awk '{for(i=1;i<=NF;i++) if (\$i=="dev") {print \$(i+1); exit}}')
if [ -z "\$IFACE" ]; then
  IFACE="eth0"
fi
echo "Using interface: \$IFACE"

# NCCL settings for multi-GPU
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME="\$IFACE"
export GLOO_SOCKET_IFNAME="\$IFACE"
export NCCL_P2P_DISABLE=0
export NCCL_DEBUG=INFO

# --------------------------
# Run training in container
# --------------------------
singularity exec --nv \
    --bind "$WORKSPACE" \
    --bind /petrobr/parceirosbr/spfm \
    "$SIF" python "$SCRIPT_PATH" $f

EOT
done
