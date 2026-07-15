#!/bin/bash

# --------------------------
# Configuration
# --------------------------
SCRIPT_PATH="/petrobr/parceirosbr/home/joao.frare/workspace/spfm/Seismic-Byol/dev-seismic-byol/imagenet/imagenetTrain.py"
WORKSPACE="/petrobr/parceirosbr/home/joao.frare/workspace"
export SIF="/petrobr/parceirosbr/spfm/singularity/amd64/deeprock/ngc/MINERVA_v0_3_9-beta-SPINN_v0_0_1.sif"

repetition=(0 1 2)
per_class=640
num_classes=125
level=3


for r in "${repetition[@]}"; do

    FLAGS="--reduction_mode default --num_classes ${num_classes} --per_class ${per_class} --repetition ${r}"
    mkdir -p /petrobr/parceirosbr/home/joao.frare/workspace/spfm/Seismic-Byol/dev-seismic-byol/imagenet/jobs_out/imagenetTraining/repetition_${r}/default

    sbatch <<EOT
#!/bin/bash

#SBATCH --job-name=imgnet_full_r${r}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=20
#SBATCH --gpus-per-node=2
#SBATCH --partition=ict-h100
#SBATCH --account=spfm
#SBATCH --time=24:00:00
#SBATCH --output=/petrobr/parceirosbr/home/joao.frare/workspace/spfm/Seismic-Byol/dev-seismic-byol/imagenet/jobs_out/imagenetTraining/repetition_${r}/default/train_r${r}_p_${per_class}_n_${num_classes}_%j.out
#SBATCH --error=/petrobr/parceirosbr/home/joao.frare/workspace/spfm/Seismic-Byol/dev-seismic-byol/imagenet/jobs_out/imagenetTraining/repetition_${r}/default/train_r${r}_p_${per_class}_n_${num_classes}_%j.err

cd "\$SLURM_SUBMIT_DIR"

echo "=== Informações do Job ==="
echo "ID do Job: \$SLURM_JOB_ID"
echo "Nós alocados: \$SLURM_JOB_NODELIST"
echo "Flags utilizadas: $FLAGS"
echo "Data de início: \$(date)"
echo "=========================="

nvidia-smi

# Exporta as variáveis de ambiente necessárias para o Singularity
export SINGULARITYENV_CUDA_VISIBLE_DEVICES=\$CUDA_VISIBLE_DEVICES

# SOLUÇÃO DO ERRO: Sorteia uma porta e um IP dinâmicos para o DDP antes do srun
export SINGULARITYENV_MASTER_PORT=\$(shuf -i 50000-65000 -n 1)
export SINGULARITYENV_MASTER_ADDR=\$(hostname -i)

echo "Porta DDP Sorteada: \$SINGULARITYENV_MASTER_PORT"
echo "IP do Nó Master: \$SINGULARITYENV_MASTER_ADDR"

srun --unbuffered singularity exec --nv \
    --bind "$WORKSPACE":"$WORKSPACE" \
    --bind /petrobr/parceirosbr/home/vinicius.soares/workspace:/petrobr/parceirosbr/home/vinicius.soares/workspace \
    --bind /petrobr/parceirosbr/spfm:/petrobr/parceirosbr/spfm \
    "\$SIF" \
    bash -c "
        export PYTHONPATH=/petrobr/parceirosbr/home/joao.frare/workspace/spfm/Seismic-Byol/Minerva-dev:\$PYTHONPATH
        python3 -u $SCRIPT_PATH $FLAGS
    "
EOT
done