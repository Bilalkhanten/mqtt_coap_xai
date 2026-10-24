#!/bin/bash
# Regenerate all figures into ../figs (needs results/ and data/ from the experiments).
cd "$(dirname "$0")"
python3 figures_a.py
python3 figures_b.py
python3 figures_c.py
python3 figures_d.py
