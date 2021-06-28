#!/bin/bash
#SBATCH --partition=cpufast
#SBATCH --time=4:00:00
#SBATCH --nodes=1 --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --qos==collaborator

DATASET=$1
EXP_MODEL=$2
SEED=$3

module load Python/3.8.2-GCCcore-9.3.0

# load virtualenv
source ./pyenv_gnn/bin/activate

python ./train.py --dataset=$DATASET --method=$EXP_MODEL --seed=$SEED
python ./explainer_main.py --dataset=$DATASET --model=$EXP_MODEL --method=$EXP_MODEL --seed=$SEED