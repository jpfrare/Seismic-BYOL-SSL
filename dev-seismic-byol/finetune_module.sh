#!/bin/bash

# Caminho para o contêiner Singularity
SIF_PATH="/petrobr/parceirosbr/home/vinicius.soares/singularity/minerva_dev.sif"

# Caminho para o script que você quer rodar
SCRIPT_PATH="/petrobr/parceirosbr/home/vinicius.soares/workspace/Seismic-Byol/dev-seismic-byol/finetune_combinations.sh"


WORKSPACE=/petrobr/parceirosbr/home/vinicius.soares/workspace
SIF=/petrobr/parceirosbr/home/vinicius.soares/singularity/minerva_dev.sif
# Flags/argumentos do script (exemplo)
FLAGS=(
  #
  
    # " --pretrain_data both_N --finetune_data seam_ai_N --num_epochs 6500 --batch_size 8 --repetition 9 --cap 1.0 --freeze_list '0 0 0 0 1' --steps"
  # " --pretrain_data both_N --finetune_data seam_ai_N --num_epochs 6500 --batch_size 8 --repetition 9 --cap 1.0 --freeze_list '0 0 0 1 0' --steps"
  # " --pretrain_data both_N --finetune_data seam_ai_N --num_epochs 6500 --batch_size 8 --repetition 9 --cap 1.0 --freeze_list '0 0 1 0 0' --steps"
  " --pretrain_data both_N --finetune_data seam_ai_N --num_epochs 6500 --batch_size 8 --repetition 9 --cap 1.0 --freeze_list '0 1 0 0 0' --steps"
  # " --pretrain_data both_N --finetune_data seam_ai_N --num_epochs 6500 --batch_size 8 --repetition 9 --cap 1.0 --freeze_list '1 0 0 0 0' --steps" " --pretrain_data both_N --finetune_data seam_ai_N --num_epochs 6500 --batch_size 8 --repetition 9 --cap 1.0 --freeze_list '0 0 0 0 1' --steps"
  # " --pretrain_data both_N --finetune_data seam_ai_N --num_epochs 6500 --batch_size 8 --repetition 9 --cap 1.0 --freeze_list '0 0 0 1 0' --steps"
  # " --pretrain_data both_N --finetune_data seam_ai_N --num_epochs 6500 --batch_size 8 --repetition 9 --cap 1.0 --freeze_list '0 0 1 0 0' --steps"
  " --pretrain_data both_N --finetune_data seam_ai_N --num_epochs 6500 --batch_size 8 --repetition 9 --cap 1.0 --freeze_list '0 1 0 0 0' --steps"
  # " --pretrain_data both_N --finetune_data seam_ai_N --num_epochs 6500 --batch_size 8 --repetition 9 --cap 1.0 --freeze_list '1 0 0 0 0' --steps"
)

# Loop para enviar um job por flag
# SBtch --array=0-4

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

cd \$SLURM_SUBMIT_DIR

echo "Nó(s) alocado(s): \$SLURM_JOB_NODELIST"
nvidia-smi

# Carrega o Singularity e executa o script
singularity exec --nv \
    --bind $WORKSPACE \
    --bind /petrobr/parceirosbr/spfm \
    $SIF python $WORKSPACE/Seismic-Byol/dev-seismic-byol/cli_finetune.py $f
EOT
done
