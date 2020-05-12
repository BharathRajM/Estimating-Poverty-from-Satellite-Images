#!/bin/bash
#SBATCH --time=10:30:00

#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --job-name=SNI
#SBATCH --mem=16000

module load Python/3.7.4-GCCcore-8.3.0
module load TensorFlow/2.1.0-fosscuda-2019b-Python-3.7.4 
module load scikit-learn/0.21.3-foss-2019b-Python-3.7.4  
module load matplotlib/3.1.1-foss-2019b-Python-3.7.4

python Train_Batches64_with_Checkpoints.py -c Output_Checkpoints64/ -d Dataset/ -p Plot64/ -s 10 -m Output_Checkpoints64/epoch_10.h5