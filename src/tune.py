"""Random-search hyper-parameter tuning on the chronological validation block (days 20-22)."""
import sys, time
import numpy as np
from common import *

D = Data(DATA + "/siteA.pkl")
tr, va, te = D.chrono()
rng = np.random.default_rng(7)
lu = lambda lo, hi: float(np.exp(rng.uniform(np.log(lo), np.log(hi))))
SPACES = dict(
    lr=(6, lambda: dict(C=lu(1e-3, 1e2))),
    rf=(8, lambda: dict(n_estimators=int(rng.choice([200, 300, 400])), max_depth=[10, 14, 18, 24, None][int(rng.integers(0, 5))],
                       min_samples_leaf=int(rng.choice([1, 2, 4])), max_features=["sqrt", 0.3, 0.5][int(rng.integers(0, 3))])),
    xgb=(12, lambda: dict(n_estimators=600, max_depth=int(rng.integers(3, 10)), learning_rate=lu(0.03, 0.3),
                         subsample=float(rng.uniform(.6, 1)), colsample_bytree=float(rng.uniform(.5, 1)),
                         reg_lambda=lu(.1, 10), gamma=float(rng.uniform(0, 1)), min_child_weight=lu(1, 10))),
    lgbm=(10, lambda: dict(n_estimators=600, num_leaves=int(rng.choice([15, 31, 63, 127])), learning_rate=lu(0.03, 0.3),
                          subsample=float(rng.uniform(.6, 1)), colsample_bytree=float(rng.uniform(.5, 1)),
                          reg_lambda=lu(.1, 10), min_child_samples=int(rng.choice([10, 20, 50])))),
    mlp=(8, lambda: dict(hidden=int(rng.choice([64, 128, 256])), depth=int(rng.choice([2, 3])), dropout=float(rng.choice([.1, .3])),
                        lr=float(rng.choice([1e-3, 3e-3])), wd=float(rng.choice([1e-5, 1e-4, 1e-3])))),
    cnnlstm=(6, lambda: dict(conv=int(rng.choice([32, 64])), lstm=int(rng.choice([64, 100])), dropout=float(rng.choice([.2, .4])),
                            lr=float(rng.choice([1e-3, 2e-3])), wd=float(rng.choice([1e-5, 1e-4])), epochs=30)),
    xgblag=(6, lambda: dict(n_estimators=500, max_depth=int(rng.integers(3, 9)), learning_rate=lu(0.04, 0.3),
                           subsample=float(rng.uniform(.6, 1)), colsample_bytree=float(rng.uniform(.3, .8)),
                           reg_lambda=lu(.1, 10), gamma=float(rng.uniform(0, 1)), min_child_weight=lu(1, 10))),
)
which = sys.argv[1:] or list(SPACES)
try:
    out = jload("tuning")
except Exception:
    out = {}
for name in which:
    ntr, sampler = SPACES[name]
    trials = []
    for t in range(ntr):
        p = sampler()
        p = {k: (v.item() if hasattr(v, "item") else v) for k, v in p.items()}
        m = make(name, p, seed=0)
        _, ft = timed(m.fit, D, tr, va)
        pv = m.predict_proba(D, va)
        f1 = f1_score(D.y[va], pv.argmax(1), average="macro")
        trials.append(dict(params=p, val_macro_f1=float(f1), fit_s=ft))
        print(f"{name} trial {t+1}/{ntr} val macro-F1={f1:.4f}  fit={ft:.0f}s  {p}", flush=True)
    best = max(trials, key=lambda r: r["val_macro_f1"])
    out[name] = dict(best=best["params"], best_val_f1=best["val_macro_f1"], trials=trials)
    jdump(out, "tuning")
print("TUNING DONE", flush=True)
