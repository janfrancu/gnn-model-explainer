#!/bin/bash
REPETITIONS=$1
EXP_MODEL=$2
NUM_CONC=1

LOG_DIR="${HOME}/logs/GNNExplainer/"
if [ ! -d "$LOG_DIR" ]; then
	mkdir $LOG_DIR
fi

datasets=("syn1" "syn2" "syn3" "syn4" "syn5")

for d in ${datasets[*]}
do
	sbatch \
	--array=1-${REPETITIONS}%${NUM_CONC} \
	--output="${LOG_DIR}/${d}-%A_%a.out" \
	./run_train_explain.sh ${d} ${EXP_MODEL}
done