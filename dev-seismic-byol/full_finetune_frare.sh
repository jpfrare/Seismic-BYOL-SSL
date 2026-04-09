#!/bin/bash

# --------------------------
# Configuration
# --------------------------
SCRIPT_PATH="/petrobr/parceirosbr/home/joao.frare/workspace/spfm/Seismic-Byol/dev-seismic-byol/new_cli_finetune.py"
WORKSPACE="/petrobr/parceirosbr/home/joao.frare/workspace"
export SIF="/petrobr/parceirosbr/spfm/singularity/amd64/deeprock/ngc/MINERVA_v0_3_9-beta-SPINN_v0_0_1.sif"

mkdir -p jobs_out
mkdir -p jobs_out/full_finetune

PRETRAINS=("seam_ai_N" "f3_N" "both_N" "imagenet" "coco")
FINETUNEDATA=("seam_ai_N" "f3_N")
REPETITIONS=(0 1 2)

for p in "${PRETRAINS[@]}"; do
  for f in "${FINETUNEDATA[@]}"; do
    for r in "${REPETITIONS[@]}"; do

    FLAGS="--pretrain_data ${p} --finetune_data ${f} --num_epochs 50 --batch_size 8 --repetition ${r} --learning_rate 1e-5 --cap 1.0"
    
    sbatch <<EOT
#!/bin/bash
#SBATCH --job-name=full_finetune_job_p${p}_f${f}_r${r}
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1       
#SBATCH --partition=ict-h100
#SBATCH --account=spfm
#SBATCH --time=04:00:00
#SBATCH --output=jobs_out/full_finetune/full_finetune_%j.out
#SBATCH --error=jobs_out/full_finetune/full_finetune_%j.err

cd "\$SLURM_SUBMIT_DIR"

echo "Allocated nodes: \$SLURM_JOB_NODELIST"
echo "Running flags: $FLAGS"
echo "pretrain: $p; fine_tune_data: $f; repetition: $r"
nvidia-smi


singularity exec --nv \
    --bind "$WORKSPACE":"$WORKSPACE" \
    --bind /petrobr/parceirosbr/home/vinicius.soares/workspace:/petrobr/parceirosbr/home/vinicius.soares/workspace \
    --bind /petrobr/parceirosbr/spfm \
    "$SIF" \
    bash -c "
        export PYTHONPATH=/petrobr/parceirosbr/home/joao.frare/workspace/spfm/Seismic-Byol/Minerva-dev:\$PYTHONPATH
        python3 $SCRIPT_PATH $FLAGS
    "
EOT

    done
  done
done

