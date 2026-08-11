#!/bin/bash

# --------------------------
# Configuration
# --------------------------
SCRIPT_PATH="/petrobr/parceirosbr/home/joao.frare/workspace/spfm/Seismic-Byol/dev-seismic-byol/imagenet/imagenetFinetuning.py"
WORKSPACE="/petrobr/parceirosbr/home/joao.frare/workspace"
export SIF="/petrobr/parceirosbr/spfm/singularity/amd64/deeprock/ngc/MINERVA_v0_3_9-beta-SPINN_v0_0_1.sif"

repetition=(3 4)
finetune_dataset=('seam_ai_N' 'f3_N')
protocol=('full_freeze' 'full_finetuning')
level=(9 7 6 3)

for l in "${level[@]}"; do
    for r in "${repetition[@]}"; do
        for d in "${finetune_dataset[@]}"; do
            for p in "${protocol[@]}"; do

                if [ "${p}" = 'full_finetuning' ]; then
                    freeze='full_finetuning'
                    head='deeplab'

                else 
                    freeze='full_freeze'
                    head='linear'

                fi

                FLAGS="--reduction_mode taxonomic --level ${l} --top_down --backbone_freeze ${freeze} --pred_head ${head} --repetition ${r} --finetune_dataset ${d}"
                root=/petrobr/parceirosbr/home/joao.frare/workspace/spfm/Seismic-Byol/dev-seismic-byol/imagenet/jobs_out/imagenetFinetune/finetune_${d}/taxonomic/repetition_${r}
                mkdir -p "${root}"

        sbatch <<EOT
#!/bin/bash

#SBATCH --job-name=${p}_taxonomic_${r}_${d}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --gpus-per-node=1       
#SBATCH --partition=ict-h100
#SBATCH --account=spfm
#SBATCH --time=01:30:00
#SBATCH --output=${root}/taxonomic_level_${l}_${p}_%j.out
#SBATCH --error=${root}/taxonomic_level_${l}_${p}_%j.err

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
done