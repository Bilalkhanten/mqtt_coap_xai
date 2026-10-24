"""Generate the two simulated sites (30 days each) and the attack-burst schedules.

Writes ../data/siteA.pkl, ../data/siteB.pkl (pandas DataFrames, one row per 30 s window with
labels, scope, FDIA variant, benign-event bitmask and the 68 features) and ../data/bursts.pkl.
Runs are deterministic: all randomness derives from the seeds in config.py.
Usage: python3 generate_data.py [A|B|AB]
"""
import os, sys, pickle
from common import DATA
from simulate import simulate
from config import SITE_A, SITE_B

if __name__ == "__main__":
    which = sys.argv[1].upper() if len(sys.argv) > 1 else "AB"
    os.makedirs(DATA, exist_ok=True)
    bp = DATA + "/bursts.pkl"
    bursts = pickle.load(open(bp, "rb")) if os.path.exists(bp) else {}
    for key, cfg in (("A", SITE_A), ("B", SITE_B)):
        if key not in which:
            continue
        df, meta, _ = simulate(cfg)
        df.to_pickle(f"{DATA}/site{key}.pkl")
        bursts[f"bursts_{key}"] = meta["bursts"]
        print(f"site {key}: {df.shape}, bursts {len(meta['bursts'])}", flush=True)
    pickle.dump(bursts, open(bp, "wb"))
