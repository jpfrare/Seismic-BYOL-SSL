#!/bin/bash

# --------------------------
# Configuration
# --------------------------
SCRIPT_PATH="/petrobr/parceirosbr/home/joao.frare/workspace/spfm/Seismic-Byol/dev-seismic-byol/imagenet/imagenetTrainARM.py"
WORKSPACE="/petrobr/parceirosbr/home/joao.frare/workspace"
export SIF="/petrobr/parceirosbr/spfm/singularity/arm64/deeprock/ngc/MINERVA_v0_3_9-beta-SPINN_v0_0_1.sif"

repetition=(3)
version=traditional
red_mode=default
per_class=(320)
num_classes=1000

for p in "${per_class[@]}"; do
for r in "${repetition[@]}"; do
    FLAGS="--reduction_mode ${red_mode} --version ${version} --num_classes ${num_classes} --per_class ${p} --repetition ${r}"
    root=/petrobr/parceirosbr/home/joao.frare/workspace/spfm/Seismic-Byol/dev-seismic-byol/imagenet/jobs_out/imagenetTraining/${version}/repetition_${r}/${red_mode}/${num_classes}
    mkdir -p ${root}

    sbatch <<EOT
#!/bin/bash

#SBATCH --job-name=ARMp${p}r${r}
#SBATCH --nodes=1
#SBATCH --exclude=sdumont2nd2045
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --gpus-per-node=1
#SBATCH --partition=ict-gh200
#SBATCH --account=spfm
#SBATCH --time=24:00:00
#SBATCH --output=${root}/${p}_%j.out
#SBATCH --error=${root}/${p}_%j.err

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
echo "Data de Término: \$(date)"
EOT
done
done

