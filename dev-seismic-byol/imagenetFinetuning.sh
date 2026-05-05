#!/bin/bash

# --------------------------
# Configuration
# --------------------------
SCRIPT_PATH="/petrobr/parceirosbr/home/joao.frare/workspace/spfm/Seismic-Byol/dev-seismic-byol/imagenetFinetuning.py"
WORKSPACE="/petrobr/parceirosbr/home/joao.frare/workspace"
export SIF="/petrobr/parceirosbr/spfm/singularity/amd64/deeprock/ngc/MINERVA_v0_3_9-beta-SPINN_v0_0_1.sif"


repetition=(0 1 2)
finetune_dataset=('seam_ai_N' 'f3_N')
per_class=(10)

for r in "${repetition[@]}"; do
    for p in "${per_class[@]}"; do
        for d in "${finetune_dataset[@]}"; do

        FLAGS="--per_class ${p} --repetition ${r} --finetune_dataset ${d}"
        mkdir -p jobs_out/imagenetFinetune/finetune_${d}/repetition_${r}

        sbatch <<EOT
#!/bin/bash

#SBATCH --job-name=imgnet_${p}_r${r}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --gpus-per-node=1       
#SBATCH --partition=ict-h100
#SBATCH --account=spfm
#SBATCH --time=24:00:00
#SBATCH --output=jobs_out/imagenetFinetune/finetune_${d}/repetition_${r}/train_${p}_r${r}_%j.out
#SBATCH --error=jobs_out/imagenetFinetune/finetune_${d}/repetition_${r}/train_${p}_r${r}_%j.err

cd "\$SLURM_SUBMIT_DIR"

echo "=== Informações do Job ==="
echo "ID do Job: \$SLURM_JOB_ID"
echo "Nós alocados: \$SLURM_JOB_NODELIST"
echo "Flags utilizadas: $FLAGS"
echo "Data de início: \$(date)"
echo "=========================="

nvidia-smi

singularity exec --nv \
    --bind "$WORKSPACE":"$WORKSPACE" \
    --bind /petrobr/parceirosbr/home/vinicius.soares/workspace:/petrobr/parceirosbr/home/vinicius.soares/workspace \
    --bind /petrobr/parceirosbr/spfm:/petrobr/parceirosbr/spfm\
    "$SIF" \
    bash -c "
        export PYTHONPATH=/petrobr/parceirosbr/home/joao.frare/workspace/spfm/Seismic-Byol/Minerva-dev:\$PYTHONPATH
        python3 $SCRIPT_PATH $FLAGS
    "
EOT
done
done
done