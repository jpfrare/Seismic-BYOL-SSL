SCRIPT_PATH="/petrobr/parceirosbr/home/joao.frare/workspace/spfm/Seismic-Byol/dev-seismic-byol/new_cli_finetune.py"
WORKSPACE="/petrobr/parceirosbr/home/joao.frare/workspace"
export SIF="/petrobr/parceirosbr/spfm/singularity/amd64/deeprock/ngc/MINERVA_v0_3_9-beta-SPINN_v0_0_1.sif"

mkdir -p jobs_out
mkdir -p jobs_out/linear_redout

PRETRAINS=("seam_ai_N" "f3_N" "both_N" "imagenet" "coco")
FINETUNEDATA=("seam_ai_N" "f3_N")
REPETITIONS=(0 1 2)
CAPS=(2 8 128 1.0)

for p in "${PRETRAINS[@]}"; do
    for f in "${FINETUNEDATA[@]}"; do
        for r in "${REPETITIONS[@]}"; do
            for c in "${CAPS[@]}"; do

                FLAGS="--pretrain_data ${p} --finetune_data ${f} --num_epochs 50 --batch_size 8 --repetition ${r} --learning_rate 1e-5 --cap ${c}"

                sbatch << EOT
#!/bin/bash
#SBATCH --job-name=linear_redout_job_p${p}_f${f}_r${r}_c${c}
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1       
#SBATCH --partition=ict-h100
#SBATCH --account=spfm
#SBATCH --time=04:00:00
#SBATCH --output=jobs_out/linear_redout/linear_redout_%j.out
#SBATCH --error=jobs_out/linear_redout/linear_redout_%j.err

cd "\$SLURM_SUBMIT_DIR"

echo "Allocated nodes: \$SLURM_JOB_NODELIST"
echo "Running flags: $FLAGS"
echo "pretrain: $p; fine_tune_data: $f; repetition: $r; cap: ${c}"
nvidia-smi


singularity exec --nv \
    --bind "$WORKSPACE":"$WORKSPACE" \
    --bind /petrobr/parceirosbr/home/vinicius.soares/workspace:/petrobr/parceirosbr/home/vinicius.soares/workspace \
    --bind /petrobr/parceirosbr/spfm \
    "$SIF" \
    bash -c "
        export PYTHONPATH=/petrobr/parceirosbr/home/joao.frare/workspace/spfm/Seismic-Byol/Minerva-dev:\$PYTHONPATH
        python3 "$SCRIPT_PATH" $FLAGS
    "
EOT

    done
  done
done