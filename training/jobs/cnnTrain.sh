#!/bin/bash
# set the number of nodes and processes per node

#SBATCH --job-name=cnnTrain
# set the number of nodes and processes per node
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --nodelist=n17
#SBATCH --partition=cuda
# set max wallclock time
#SBATCH --time=48:00:00
#SBATCH --output=/home/ivanam/Seminar1/training/output_logs/cnn/cnn_train%j.log
#SBATCH --error=/home/ivanam/Seminar1/training/output_logs/cnn/cnn_train%j.err


module load python/miniconda3.10 
eval "$(conda shell.bash hook)"
conda activate
#conda install -c conda-forge rdkit

PYTHON_EXECUTABLE=$(which python)

${PYTHON_EXECUTABLE} /home/ivanam/Seminar1/training/baseline_cnn/baseline_cnn_train.py

echo "Test done."