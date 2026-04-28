#!/bin/bash

# Caminho para o contêiner Singularity
WORKSPACE="/petrobr/parceirosbr/home/joao.frare/spfm/Seismic-Byol"
SIF="/petrobr/parceirosbr/home/joao.frare/singularity/minerva_dev.sif"
# Flags/argumentos do script (exemplo)
FLAGS=(
  "--repetition 5 --gpus 0"
  "--repetition 6 --gpus 0"
  "--repetition 7 --gpus 0"
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
#SBATCH -J evaluate_job
#SBATCH --output=jobs_out/finetune_%j.out
#SBATCH --error=jobs_out/finetune_%j.err

cd \$SLURM_SUBMIT_DIR

echo "Nó(s) alocado(s): \$SLURM_JOB_NODELIST"
nvidia-smi

# Carrega o Singularity e executa o script
singularity exec --nv \
    --bind $WORKSPACE \
    --bind /petrobr/parceirosbr/spfm \
    $SIF python $WORKSPACE/Seismic-Byol/dev-seismic-byol/cli_evaluate.py $f
EOT
done
