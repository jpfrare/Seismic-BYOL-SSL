#!/bin/bash

# --------------------------
# Configuration
# --------------------------
SCRIPT_PATH="/petrobr/parceirosbr/home/joao.frare/workspace/spfm/Seismic-Byol/dev-seismic-byol/imagenet/imagenetFinetuning.py"
WORKSPACE="/petrobr/parceirosbr/home/joao.frare/workspace"
export SIF="/petrobr/parceirosbr/spfm/singularity/arm64/deeprock/ngc/MINERVA_v0_3_9-beta-SPINN_v0_0_1.sif"

repetition=(0 1 2)
finetune_dataset=(f3_N seam_ai_N)
protocol=(linear_readout full_finetuning)
red_mode=full

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
                FLAGS="--reduction_mode ${red_mode} --scratch --backbone_freeze ${freeze} --version modern --pred_head ${head} --finetune_dataset ${d} --repetition ${r} --eval"
                root=/petrobr/parceirosbr/home/joao.frare/workspace/spfm/Seismic-Byol/dev-seismic-byol/imagenet/jobs_out/imagenetFinetune/finetune_${d}/scratch/repetition_${r}
                mkdir -p "${root}"

    sbatch <<EOT
#!/bin/bash

#SBATCH --job-name=scratch_d${d}_p${p}_v${v}_r${r}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --gpus-per-node=1       
#SBATCH --partition=ict-gh200
#SBATCH --account=spfm
#SBATCH --time=01:30:00
#SBATCH --output=${root}/${d}_${p}_${v}_${r}_%j.out
#SBATCH --error=${root}/${d}_${p}_${v}_${r}_%j.err

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

