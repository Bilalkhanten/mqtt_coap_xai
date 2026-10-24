#!/bin/bash
# Full pipeline: data -> tuning -> experiments -> figures.  Run from anywhere: bash src/run_all.sh
# Data generation takes ~15 s per site; tuning and the experiments take the bulk of the time.
set -u
cd "$(dirname "$0")"
mkdir -p ../data ../results ../figs
LOG=../results/run_all.log
step() { echo "=== $1 start $(date +%T)" >> $LOG; python3 "$@" > ../results/$1.log 2>&1 || echo "!!! $1 FAILED" >> $LOG; echo "=== $1 end $(date +%T)" >> $LOG; }
[ -f ../data/siteA.pkl ] && [ -f ../data/siteB.pkl ] || step generate_data.py
[ -f ../results/tuning.json ] || step tune.py
for s in exp_main exp_stats exp_protocols exp_ablation exp_robust exp_xai exp_latency exp_latency2; do step $s.py; done
echo ALL_EXPERIMENTS_DONE >> $LOG
bash make_figures.sh
