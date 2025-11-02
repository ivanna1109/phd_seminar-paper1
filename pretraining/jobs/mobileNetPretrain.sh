#!/bin/bash
# set the number of nodes and processes per node

#SBATCH --job-name=ssl_MobileNet
# set the number of nodes and processes per node
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --nodelist=n16

#SBATCH --partition=cuda
# set max wallclock time
#SBATCH --time=48:00:00
#SBATCH --output=/home/ivanam/Seminar1/pretraining/output_logs/mobile/ssl_mobile%j.log
#SBATCH --error=/home/ivanam/Seminar1/pretraining/output_logs/mobile/ssl_mobile%j.err


module load python/miniconda3.10 
eval "$(conda shell.bash hook)"
conda activate
#conda install -c conda-forge rdkit

PYTHON_EXECUTABLE=$(which python)

${PYTHON_EXECUTABLE} pretraining/pt_mobileNet.py

echo "Test done."