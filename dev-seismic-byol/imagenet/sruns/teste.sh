#!/bin/bash

SCRIPT_PATH="/petrobr/parceirosbr/home/joao.frare/workspace/spfm/Seismic-Byol/dev-seismic-byol/imagenet/teste_information_organizer.py"
WORKSPACE="/petrobr/parceirosbr/home/joao.frare/workspace"
export SIF="/petrobr/parceirosbr/spfm/singularity/amd64/deeprock/ngc/MINERVA_v0_3_9-beta-SPINN_v0_0_1.sif"

mkdir -p /petrobr/parceirosbr/home/joao.frare/workspace/spfm/Seismic-Byol/dev-seismic-byol/imagenet/jobs_out/teste_organizer

FLAGS="--reduction_mode full --repetition 0 --finetune_dataset f3_N"

sbatch <<EOT
#!/bin/bash
#SBATCH --job-name=pretrain_job
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1       
#SBATCH --partition=ict-h100
#SBATCH --account=spfm
#SBATCH --time=00:02:00
#SBATCH --output=/petrobr/parceirosbr/home/joao.frare/workspace/spfm/Seismic-Byol/dev-seismic-byol/imagenet/jobs_out/teste_organizer/%j.out
#SBATCH --error=/petrobr/parceirosbr/home/joao.frare/workspace/spfm/Seismic-Byol/dev-seismic-byol/imagenet/jobs_out/teste_organizer/%j.err

cd "\$SLURM_SUBMIT_DIR"

echo "Allocated nodes: \$SLURM_JOB_NODELIST"
nvidia-smi

# Ajustado o terceiro --bind para mapear origem:destino corretamente
singularity exec --nv --bind "$WORKSPACE":"$WORKSPACE" --bind /petrobr/parceirosbr/home/vinicius.soares/workspace:/petrobr/parceirosbr/home/vinicius.soares/workspace --bind /petrobr/parceirosbr/spfm:/petrobr/parceirosbr/spfm "$SIF" bash -c "
    python3 $SCRIPT_PATH $FLAGS
"
EOT