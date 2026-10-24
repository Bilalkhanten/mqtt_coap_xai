#!/bin/bash
# Regenerate all figures into ../figs (needs results/ and data/ from the experiments).
set -e
cd "$(dirname "$0")"
# figures_a.py draws only 8 of its 11 figures by default and figures_c.py draws nothing without arguments,
# so every figure is named explicitly.
python3 figures_a.py models confusion confusion_lr perclass cd anatomy protocols ablation imbalance calibration fdia_auc
python3 figures_b.py
python3 figures_c.py xai_global xai_faith xai_cases latency evasion transfer
python3 figures_d.py
