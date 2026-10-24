"""Main benchmark: 7 models x 5 seeds on the chronological split (test = days 23-29)."""
import sys, time
import numpy as np
from common import *

D = Data(DATA + "/siteA.pkl")
tr, va, te = D.chrono()
tuned = jload("tuning")
names = ["lr", "rf", "xgb", "lgbm", "mlp", "cnnlstm", "xgblag"]
SEEDS = [0, 1, 2, 3, 4]
yte = D.y[te]
res, probs, curves = {}, {}, {}
for nm in names:
    params = tuned[nm]["best"]
    per_seed, P, fit_t = [], [], []
    for s in (SEEDS if nm != "lr" else [0]):
        m = make(nm, params, seed=s)
        _, ft = timed(m.fit, D, tr, va)
        p = m.predict_proba(D, te)
        P.append(p.astype(np.float32))
        per_seed.append(metrics(yte, p))
        fit_t.append(ft)
        if nm == "cnnlstm" and s == 0:
            curves["cnnlstm"] = m.curve
        if nm == "mlp" and s == 0:
            curves["mlp"] = m.curve
        if nm in ("xgb", "lgbm") and s == 0:
            curves[nm + "_best_iter"] = int(getattr(m.m, "best_iteration", getattr(m.m, "best_iteration_", -1)) or -1)
        print(f"{nm} seed {s}: macroF1={per_seed[-1]['macro_f1']:.4f} acc={per_seed[-1]['acc']:.4f} mcc={per_seed[-1]['mcc']:.4f} fit={ft:.0f}s", flush=True)
    P = np.stack(P)
    probs[nm] = P
    agg = {}
    for k in per_seed[0]:
        v = np.array([r[k] for r in per_seed], dtype=float)
        agg[k] = dict(mean=np.nanmean(v, 0).tolist(), sd=(np.nanstd(v, 0, ddof=1).tolist() if len(v) > 1 else 0.0))
    ens = metrics(yte, P.mean(0))
    # block-bootstrap CI for macro-F1 using seed-0 predictions
    ci = block_bootstrap_ci(yte, P[0].argmax(1), D.blocks[te], macro_f1_cm, B=1000)
    # per-day macro-F1 (mean over seeds)
    perday = np.array([[f1_score(yte[D.day[te] == d], p.argmax(1)[D.day[te] == d], average="macro", labels=range(K), zero_division=0)
                        for d in range(23, 30)] for p in P]).mean(0)
    res[nm] = dict(name=MODEL_NAMES[nm], agg=agg, seed0=per_seed[0], ensemble=ens, ci_macro_f1=ci, fit_s=float(np.mean(fit_t)),
                   per_day_f1=perday.tolist(), params=params)
    jdump(res, "main")
    np.savez_compressed(f"{RES}/main_probs.npz", **{k: v for k, v in probs.items()})
jdump(dict(curves={k: v for k, v in curves.items()}), "curves")
np.save(f"{RES}/test_idx.npy", te)
print("MAIN DONE", flush=True)
