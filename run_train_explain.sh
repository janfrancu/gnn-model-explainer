#!/bin/bash
#SBATCH --partition=cpufast
#SBATCH --time=4:00:00
#SBATCH --nodes=1 --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --qos==collaborator

DATASET=$1

module load Python/3.8.2-GCCcore-9.3.0

# load virtualenv
source ./pyenv_gnn/bin/activate

python ./train.py --dataset=$DATASET
python ./explainer_main.py --dataset=$DATASET