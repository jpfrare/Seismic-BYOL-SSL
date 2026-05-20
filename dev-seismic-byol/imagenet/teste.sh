#!/bin/bash

SCRIPT_PATH="/petrobr/parceirosbr/home/joao.frare/workspace/spfm/Seismic-Byol/dev-seismic-byol/imagenet/teste_readers.py"
WORKSPACE="/petrobr/parceirosbr/home/joao.frare/workspace"
export SIF="/petrobr/parceirosbr/spfm/singularity/amd64/deeprock/ngc/MINERVA_v0_3_9-beta-SPINN_v0_0_1.sif"

mkdir -p jobs_out/testereaders

sbatch <<EOT
#!/bin/bash
#SBATCH --job-name=pretrain_job
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1       
#SBATCH --partition=ict-h100
#SBATCH --account=spfm
#SBATCH --time=00:02:00
#SBATCH --output=jobs_out/testereaders/teste_readers%j.out
#SBATCH --error=jobs_out/testereaders/teste_readers_%j.err

cd "\$SLURM_SUBMIT_DIR"

echo "Allocated nodes: \$SLURM_JOB_NODELIST"
nvidia-smi

singularity exec --nv \\
    --bind "$WORKSPACE":"$WORKSPACE" \\
    --bind /petrobr/parceirosbr/home/vinicius.soares/workspace:/petrobr/parceirosbr/home/vinicius.soares/workspace \\
    --bind /petrobr/parceirosbr/spfm \\
    "$SIF" \\
    bash -c "
        # 1. Concatenando seu projeto E a pasta de libs externas no PYTHONPATH
        export PYTHONPATH=/petrobr/parceirosbr/home/joao.frare/workspace/spfm/Seismic-Byol/Minerva-dev:/petrobr/parceirosbr/home/joao.frare/workspace/spfm/Seismic-Byol/dev-seismic-byol/imagenet/base/libs:\$PYTHONPATH
        
        # 2. Apontando para onde estão os dados baixados do NLTK (tokenizers, etc)
        export NLTK_DATA=/petrobr/parceirosbr/home/joao.frare/workspace/spfm/Seismic-Byol/dev-seismic-byol/imagenet/base/nltk_data
        
        # 3. Executa o seu teste convencionalmente
        python3 '$SCRIPT_PATH'
    "
EOT