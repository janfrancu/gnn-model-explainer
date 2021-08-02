#!/bin/bash
MAX_SEED=$1
EXP_MODEL=$2

LOG_DIR="${HOME}/logs/GNNExplainer/"
if [ ! -d "$LOG_DIR" ]; then
	mkdir $LOG_DIR
fi

datasets=("syn1" "syn2" "syn3" "syn4" "syn5")

for d in ${datasets[*]}
do
	for s in $(seq 0 ${MAX_SEED-1})
	do	
		sbatch \
		--output="${LOG_DIR}/${d}_${s}_${EXP_MODEL}-%A_%a.out" \
		./run_train_explain.sh ${d} ${EXP_MODEL} ${s}
	done
done