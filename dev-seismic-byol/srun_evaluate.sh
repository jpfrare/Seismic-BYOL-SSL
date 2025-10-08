# #!/bin/bash
# #SBATCH --nodes=1
# #SBATCH --ntasks=1
# #SBATCH --gpus=4
# #SBATCH --gpus-per-node=4
# #SBATCH --partition=ict-h100
# #SBATCH --account=spfm
# #SBATCH --time=24:00:00
# #SBATCH -J finetune_seismic_job

#!/bin/bash

SCRIPT_PATH="/petrobr/parceirosbr/home/vinicius.soares/workspace/Seismic-Byol/dev-seismic-byol/cli_evaluate.py"
WORKSPACE=/petrobr/parceirosbr/home/vinicius.soares/workspace
SIF=/petrobr/parceirosbr/home/vinicius.soares/singularity/minerva_dev.sif


FLAGS=(
  "--repetition 5"
  # "--repetition 6"
  # "--repetition 7"
  # "--repetition 8"
  # "--repetition 9"
)

# Loop para enviar um job por flag
for f in "${FLAGS[@]}"; do
  sbatch <<EOT
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --partition=ict-h100
#SBATCH --account=spfm
#SBATCH --time=24:00:00
#SBATCH -J finetune_job
#SBATCH --output=jobs_out/finetune_%j.out
#SBATCH --error=jobs_out/finetune_%j.err

mkdir jobs_out
cd \$SLURM_SUBMIT_DIR

echo "Nó(s) alocado(s): \$SLURM_JOB_NODELIST"
nvidia-smi

# Carrega o Singularity e executa o script
singularity exec --nv \
    --bind $WORKSPACE \
    --bind /petrobr/parceirosbr/spfm \
    $SIF bash $WORKSPACE/Seismic-Byol/dev-seismic-byol/finetune_combinations.sh $f
EOT
done
