#!/bin/bash

# --------------------------
# Configuration
# --------------------------
SCRIPT_PATH="/petrobr/parceirosbr/home/joao.frare/workspace/spfm/Seismic-Byol/dev-seismic-byol/imagenet/imagenetFinetuning.py"
WORKSPACE="/petrobr/parceirosbr/home/joao.frare/workspace"
export SIF="/petrobr/parceirosbr/spfm/singularity/arm64/deeprock/ngc/MINERVA_v0_3_9-beta-SPINN_v0_0_1.sif"

#80 9

repetition=(0 1 2)
finetune_dataset=(f3_N seam_ai_N)
protocol=(linear_readout full_finetuning)
version=(traditional modern)
num_classes=(80 9)
red_mode=default

for n in "${num_classes[@]}"; do
    for v in "${version[@]}"; do
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

                    for pc in "${per_class[@]}"; do

                        FLAGS="--reduction_mode ${red_mode} --num_classes ${n} --per_class ${pc} --version ${v} --backbone_freeze ${freeze} --pred_head ${head} --finetune_dataset ${d} --repetition ${r} --eval"
                        root=/petrobr/parceirosbr/home/joao.frare/workspace/spfm/Seismic-Byol/dev-seismic-byol/imagenet/jobs_out/imagenetFinetune/finetune_${d}/${red_mode}/${n}/${pc}/repetition_${r}
                        mkdir -p "${root}"

            sbatch <<EOT
#!/bin/bash

#SBATCH --job-name=${red_mode}_n${n}_pc${pc}_d${d}_p${p}_v${v}_r${r}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --gpus-per-node=1       
#SBATCH --partition=ict-gh200
#SBATCH --account=spfm
#SBATCH --time=01:30:00
#SBATCH --output=${root}/${pc}_${d}_${p}_${v}_${r}_%j.out
#SBATCH --error=${root}/${pc}_${d}_${p}_${v}_${r}_%j.err

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
done 
done