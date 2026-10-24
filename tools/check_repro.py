"""Consistency checks used by CI. Read-only: nothing in the repository is modified.

    python tools/check_repro.py artifacts
    python tools/check_repro.py data  --shipped DIR      # after `python src/generate_data.py`
    python tools/check_repro.py stats --shipped DIR      # after `python src/exp_stats.py`

DIR holds copies of the shipped data/bursts.pkl and results/stats.json taken before the run.

artifacts  required files exist; test indices and stored probabilities are consistent; the headline numbers
           in the README equal the numbers in results/main.json and results/stats.json.
data       both simulated sites have the documented shape and their burst schedules equal the shipped
           data/bursts.pkl: classes, scopes, ids and counts exactly, floats to 1e-9 relative (compared by content;
           the pickle bytes differ because of object sharing, and on other CPUs a few floats differ in the last bit).
stats      the re-computed results/stats.json equals the shipped one.
"""
import json
import pickle
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
MODELS = ["lr", "rf", "xgb", "lgbm", "mlp", "cnnlstm", "xgblag"]
N_WINDOWS, N_TEST, N_CLASSES = 86400, 20160, 7
errors = []


def fail(msg):
    errors.append(msg)
    print("FAIL:", msg)


def load_json(rel):
    return json.loads((ROOT / rel).read_text())


def close(a, b, tol):
    return abs(a - b) <= tol


def check_artifacts():
    required = ["README.md", "requirements.txt", "data/bursts.pkl", "results/main_probs.npz", "results/test_idx.npy"]
    required += [f"results/{n}.json" for n in ("main", "stats", "tuning", "xai", "latency", "protocols", "ablation", "robust", "imbalance")]
    required += [f"src/{n}.py" for n in ("config", "common", "generate_data", "simulate", "exp_main", "exp_stats", "figures_a")]
    for f in required:
        if not (ROOT / f).is_file():
            fail(f"missing file: {f}")
    if errors:
        return

    main, stats = load_json("results/main.json"), load_json("results/stats.json")
    if list(main) != MODELS:
        fail(f"main.json models are {list(main)}, expected {MODELS}")

    te = np.load(ROOT / "results/test_idx.npy")
    if te.shape != (N_TEST,) or len(np.unique(te)) != N_TEST or te.min() < 0 or te.max() >= N_WINDOWS:
        fail(f"test_idx.npy is not {N_TEST} unique indices below {N_WINDOWS}")
    with np.load(ROOT / "results/main_probs.npz") as z:
        for m in MODELS:
            p = z[m]
            if p.ndim != 3 or p.shape[1:] != (N_TEST, N_CLASSES):
                fail(f"main_probs.npz[{m}] has shape {p.shape}, expected (seeds, {N_TEST}, {N_CLASSES})")
            elif not np.allclose(p.sum(-1), 1, atol=1e-4):
                fail(f"main_probs.npz[{m}]: class probabilities do not sum to 1")
    cm = np.array(stats["cm_best"])
    if cm.sum() != N_TEST:
        fail(f"stats.json cm_best sums to {cm.sum()}, expected {N_TEST}")

    text = (ROOT / "README.md").read_text()
    agg = {m: main[m]["agg"] for m in MODELS}
    f1 = {m: agg[m]["macro_f1"]["mean"] for m in MODELS}

    row = re.search(r"XGBoost, 5-window lag \|\s*([\d.]+) \(95 % block-bootstrap CI ([\d.]+)[–-]([\d.]+)\), MCC ([\d.]+), "
                    r"accuracy ([\d.]+), macro-AUC ([\d.]+)", text)
    if not row:
        fail("README headline row for 'XGBoost, 5-window lag' not found or reworded")
    else:
        got = [float(x) for x in row.groups()]
        lo, hi = main["xgblag"]["ci_macro_f1"]
        want = [f1["xgblag"], lo, hi, agg["xgblag"]["mcc"]["mean"], agg["xgblag"]["acc"]["mean"], agg["xgblag"]["macro_auc"]["mean"]]
        for name, g, w in zip(("macro-F1", "CI low", "CI high", "MCC", "accuracy", "macro-AUC"), got, want):
            if not close(g, w, 0.00051):
                fail(f"README xgblag {name}: README {g}, results file {w:.4f}")

    for label, models in (("XGBoost / LightGBM / Random forest", ["xgb", "lgbm", "rf"]),
                          ("CNN–LSTM / MLP / Logistic regression", ["cnnlstm", "mlp", "lr"])):
        row = re.search(re.escape(label) + r" \|\s*([\d.]+) / ([\d.]+) / ([\d.]+)", text)
        if not row:
            fail(f"README headline row '{label}' not found or reworded")
            continue
        for m, g in zip(models, row.groups()):
            if not close(float(g), f1[m], 0.00051):
                fail(f"README {m} macro-F1: README {g}, results file {f1[m]:.4f}")

    fdia = re.search(r"FDIA is the hard class \(F1 ([\d.]+)\): recall ([\d.]+) % \(V1 bias\), ([\d.]+) % \(V2 ramp\), ([\d.]+) % \(V3", text)
    if not fdia:
        fail("README FDIA sentence not found or reworded")
    else:
        g = [float(x) for x in fdia.groups()]
        v = stats["fdia_by_variant"]
        want = [agg["xgblag"]["per_f1"]["mean"][1]] + [100 * v[k]["recall"] for k in ("V1-bias", "V2-ramp", "V3-balanced")]
        for name, a, b, tol in zip(("FDIA F1", "V1 recall", "V2 recall", "V3 recall"), g, want, (0.00051, 0.051, 0.051, 0.051)):
            if not close(a, b, tol):
                fail(f"README {name}: README {a}, results file {b:.4f}")
    print(f"checked {len(MODELS)} models, {N_TEST} test windows, README headline numbers")


def is_float_noise(a, b, tol=1e-9):
    return isinstance(a, float) and isinstance(b, float) and abs(a - b) <= tol * max(abs(a), abs(b))


def burst_diffs(ref, got):
    """[(index, {field: (shipped, regenerated)})] for every burst that is not identical."""
    out = []
    for i, (a, b) in enumerate(zip(ref, got)):
        if a == b:
            continue
        fields = {}
        for k in a.keys() | b.keys():
            if k == "params":
                for kk in a[k].keys() | b[k].keys():
                    if a[k].get(kk) != b[k].get(kk):
                        fields[f"params.{kk}"] = (a[k].get(kk), b[k].get(kk))
            elif a.get(k) != b.get(k):
                fields[k] = (a.get(k), b.get(k))
        out.append((i, fields))
    return out


def check_data(shipped):
    for site in ("A", "B"):
        p = ROOT / f"data/site{site}.pkl"
        if not p.is_file():
            fail(f"{p.name} not found; run `python src/generate_data.py` first")
            continue
        df = pd.read_pickle(p)
        print(f"site {site}: {df.shape}")
        if len(df) != N_WINDOWS:
            fail(f"site {site} has {len(df)} windows, expected {N_WINDOWS}")
        if not set(df["label"].unique()) <= set(range(N_CLASSES)):
            fail(f"site {site}: labels outside 0..{N_CLASSES - 1}")
        if site == "A" and np.load(ROOT / "results/test_idx.npy").max() >= len(df):
            fail("test_idx.npy points outside site A")
    got = pickle.loads((ROOT / "data/bursts.pkl").read_bytes())
    ref = pickle.loads((Path(shipped) / "bursts.pkl").read_bytes())
    for key in ref:
        diffs = burst_diffs(ref[key], got.get(key, []))
        print(f"{key}: shipped {len(ref[key])}, regenerated {len(got.get(key, []))}, differing {len(diffs)}")
        # categories, ids and counts must be identical; floats may differ in the last bits (libm/CPU differences)
        real = [(i, f) for i, f in diffs if any(not is_float_noise(a, b) for a, b in f.values())]
        if diffs:
            rel = [abs(a - b) / max(abs(a), abs(b)) for _, f in diffs for a, b in f.values() if isinstance(a, float) and isinstance(b, float)]
            print(f"  {len(diffs)} burst(s) differ in the last float bits only: {not real}; max relative difference {max(rel, default=0):.1e}")
            for i, f in (real or diffs)[:3]:
                print(f"  burst index {i}: {f}")
        if len(ref[key]) != len(got.get(key, [])) or real:
            fail(f"{key} differs from the shipped data/bursts.pkl")


def diff(a, b, path=""):
    """Paths where two JSON values differ; floats compare with a 1e-9 relative tolerance."""
    if isinstance(a, dict) and isinstance(b, dict):
        return [d for k in sorted(a.keys() | b.keys()) for d in diff(a.get(k), b.get(k), f"{path}/{k}")]
    if isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
        return [d for i, (x, y) in enumerate(zip(a, b)) for d in diff(x, y, f"{path}[{i}]")]
    if isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool):
        return [] if abs(a - b) <= 1e-9 * max(1.0, abs(a), abs(b)) else [f"{path}: {a} vs {b}"]
    return [] if a == b else [f"{path}: {a!r} vs {b!r}"]


def check_stats(shipped):
    ref = json.loads((Path(shipped) / "stats.json").read_text())
    got = load_json("results/stats.json")
    d = diff(ref, got)
    print(f"stats.json: {len(d)} difference(s) from the shipped file")
    for line in d[:10]:
        fail(f"stats.json {line}")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else ""
    shipped = sys.argv[sys.argv.index("--shipped") + 1] if "--shipped" in sys.argv else None
    if mode == "artifacts":
        check_artifacts()
    elif mode == "data" and shipped:
        check_data(shipped)
    elif mode == "stats" and shipped:
        check_stats(shipped)
    else:
        sys.exit(__doc__)
    if errors:
        sys.exit(f"{len(errors)} check(s) failed")
    print(f"{mode}: all checks passed")
