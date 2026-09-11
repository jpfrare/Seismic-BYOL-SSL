#!/bin/bash

# --------------------------
# Configuration
# --------------------------
SCRIPT_PATH="/petrobr/parceirosbr/home/joao.frare/workspace/spfm/Seismic-Byol/dev-seismic-byol/imagenet/imagenetPretrainEvaluation.py"
WORKSPACE="/petrobr/parceirosbr/home/joao.frare/workspace"
export SIF="/petrobr/parceirosbr/spfm/singularity/arm64/deeprock/ngc/MINERVA_v0_3_9-beta-SPINN_v0_0_1.sif"

repetition=(0 1 2)
version=(modern traditional)
num_classes=(1000 500 477 250 200 125 80 9)
red_mode=default
task=PretrainEval

for v in "${version[@]}"; do
for r in "${repetition[@]}"; do
for n in "${num_classes[@]}"; do

    if [ ${n} -eq 1000 ]; then
        per_class=(640 320 160 80 40 20)

    elif [ ${n} -eq 500 ]; then
        per_class=(1300 640 320 160 80 40)

    elif [ ${n} -eq 250 ]; then
        per_class=(1300 640 320 160 80)
    
    elif [ ${n} -eq 125 ]; then
        per_class=(1300 640 320 160)
    else
        per_class=(1300)

    fi

    for p in ${per_class[@]}; do

    FLAGS="--reduction_mode ${red_mode} --version ${v} --per_class ${p} --num_classes ${n} --repetition ${r} --eval_acc"
    ROOT=/petrobr/parceirosbr/home/joao.frare/workspace/spfm/Seismic-Byol/dev-seismic-byol/imagenet/jobs_out/${task}/${v}/repetition_${r}/${red_mode}/${n}
    mkdir -p ${ROOT}

    sbatch <<EOT
#!/bin/bash

#SBATCH --job-name=v${v}t${red_mode}r${r}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --gpus-per-node=1
#SBATCH --partition=ict-gh200
#SBATCH --account=spfm
#SBATCH --time=24:00:00
#SBATCH --output=${ROOT}/${p}_%j.out
#SBATCH --error=${ROOT}/${p}_%j.err

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
EOT
done
done
done 
done