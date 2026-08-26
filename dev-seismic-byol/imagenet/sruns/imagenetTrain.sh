#!/bin/bash

# --------------------------
# Configuration
# --------------------------
SCRIPT_PATH="/petrobr/parceirosbr/home/joao.frare/workspace/spfm/Seismic-Byol/dev-seismic-byol/imagenet/imagenetTrain.py"
WORKSPACE="/petrobr/parceirosbr/home/joao.frare/workspace"
export SIF="/petrobr/parceirosbr/spfm/singularity/arm64/deeprock/ngc/MINERVA_v0_3_9-beta-SPINN_v0_0_1.sif"

repetition=(0 1 2)
version=traditional
red_mode=taxonomic
num_classes=(477 200 80 9)

for n in "${num_classes[@]}"; do
for r in "${repetition[@]}"; do

    FLAGS="--reduction_mode default --version ${version} --num_classes ${n} --per_class 1300 --repetition ${r}"
    root=/petrobr/parceirosbr/home/joao.frare/workspace/spfm/Seismic-Byol/dev-seismic-byol/imagenet/jobs_out/imagenetTraining/${version}/repetition_${r}/${red_mode}/${n}
    mkdir -p ${root}

    sbatch <<EOT
#!/bin/bash

#SBATCH --job-name=${n}dr${r}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --gpus-per-node=1
#SBATCH --partition=ict-gh200
#SBATCH --account=spfm
#SBATCH --time=24:00:00
#SBATCH --output=${root}/${red_mode}_%j.out
#SBATCH --error=${root}/${red_mode}_%j.err

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

