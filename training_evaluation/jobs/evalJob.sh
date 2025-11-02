#!/bin/bash
# set the number of nodes and processes per node

#SBATCH --job-name=mobileEval
# set the number of nodes and processes per node
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --nodelist=n16
#SBATCH --partition=cuda
# set max wallclock time
#SBATCH --time=48:00:00
#SBATCH --output=/home/ivanam/Seminar1/training_evaluation/logs/mNetEval_%j.log
#SBATCH --error=/home/ivanam/Seminar1/training_evaluation/logs/mNetEval_%j.err


module load python/miniconda3.10 
eval "$(conda shell.bash hook)"
conda activate
#conda install -c conda-forge rdkit

PYTHON_EXECUTABLE=$(which python)

#${PYTHON_EXECUTABLE} /home/ivanam/Seminar1/training_evaluation/modelCNN_evaluation.py
#${PYTHON_EXECUTABLE} /home/ivanam/Seminar1/training_evaluation/eNet_evaluation.py
${PYTHON_EXECUTABLE} /home/ivanam/Seminar1/training_evaluation/mNet_evaluation.py
#${PYTHON_EXECUTABLE} /home/ivanam/Seminar1/training_evaluation/resNet_evaluation.py
#${PYTHON_EXECUTABLE} /home/ivanam/Seminar1/training_evaluation/embryo_evaluation.py



echo "Test done."