#!/bin/bash

# --------------------------
# Configuration
# --------------------------

SCRIPT_PATH="/petrobr/parceirosbr/home/joao.frare/workspace/spfm/Seismic-Byol/dev-seismic-byol/imagenet/imagenetTrain.py"
WORKSPACE="/petrobr/parceirosbr/home/joao.frare/workspace"
export SIF="/petrobr/parceirosbr/spfm/singularity/arm64/deeprock/ngc/MINERVA_v0_3_9-beta-SPINN_v0_0_1.sif"

repetition=(1)
version=traditional
red_mode=taxonomic
level=(3)

#--reduction_mode taxonomic --version traditional --top_down --level 6 --repetition 0

for l in "${level[@]}"; do
for r in "${repetition[@]}"; do

    if [ ${l} -eq 9 ]; then
        n=477
    elif [ ${l} -eq 7 ]; then
        n=200
    elif [ ${l} -eq 6 ]; then
        n=80
    elif [ ${l} -eq 3 ]; then
        n=9
    else
        echo "erro"
        exit
    fi

    FLAGS="--reduction_mode ${red_mode} --version ${version} --top_down --level ${l} --repetition ${r}"
    root=/petrobr/parceirosbr/home/joao.frare/workspace/spfm/Seismic-Byol/dev-seismic-byol/imagenet/jobs_out/imagenetTraining/${version}/repetition_${r}/taxonomic/${n}
    mkdir -p ${root}

    sbatch <<EOT
#!/bin/bash

#SBATCH --job-name=${red_mode}l${l}r${r}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --gpus-per-node=1
#SBATCH --partition=ict-gh200
#SBATCH --account=spfm
#SBATCH --time=24:00:00
#SBATCH --output=${root}/taxonomic_%j.out
#SBATCH --error=${root}/taxonomic_%j.err

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
