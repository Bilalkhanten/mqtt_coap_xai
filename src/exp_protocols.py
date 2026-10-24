"""E1: evaluation-protocol study.  How much does a random split inflate the score?

For RF and XGBoost (tuned hyper-parameters) we compare
  random   : stratified random 70/10/20 split of windows (what most IDS papers do)
  grouped  : 5-fold GroupKFold, group = attack burst (or 1-hour normal block)
  chrono   : chronological train / validation / test with a purge gap
and report the distance (in 30 s windows) from every test window to its nearest training window.
"""
import numpy as np
from sklearn.model_selection import GroupKFold
from common import *

D = Data(DATA + "/siteA.pkl")
tuned = jload("tuning")
out = {}


def nn_dist(tr, te, n):
    mask = np.zeros(n, bool)
    mask[tr] = True
    idx = np.where(mask)[0]
    pos = np.searchsorted(idx, te)
    left = np.where(pos > 0, te - idx[np.maximum(pos - 1, 0)], 10 ** 9)
    right = np.where(pos < len(idx), idx[np.minimum(pos, len(idx) - 1)] - te, 10 ** 9)
    return np.minimum(left, right)


for nm in ["rf", "xgb"]:
    params = tuned[nm]["best"]
    out[nm] = {}
    # ---- chronological
    tr, va, te = D.chrono()
    ms = []
    for s in range(3):
        m = make(nm, params, seed=s).fit(D, tr, va)
        ms.append(metrics(D.y[te], m.predict_proba(D, te), full=False))
    out[nm]["chrono"] = dict(macro_f1=[r["macro_f1"] for r in ms], per_f1=np.mean([r["per_f1"] for r in ms], 0).tolist(),
                             acc=[r["acc"] for r in ms], mcc=[r["mcc"] for r in ms],
                             nn_dist_median=float(np.median(nn_dist(tr, te, D.n))), nn_dist_p05=float(np.percentile(nn_dist(tr, te, D.n), 5)))
    print(nm, "chrono", np.mean(out[nm]["chrono"]["macro_f1"]), flush=True)
    # ---- random
    ms, nd = [], []
    for s in range(3):
        tr, va, te = D.random_split(seed=s)
        m = make(nm, params, seed=s).fit(D, tr, va)
        ms.append(metrics(D.y[te], m.predict_proba(D, te), full=False))
        nd.append(nn_dist(tr, te, D.n))
    nd = np.concatenate(nd)
    out[nm]["random"] = dict(macro_f1=[r["macro_f1"] for r in ms], per_f1=np.mean([r["per_f1"] for r in ms], 0).tolist(),
                             acc=[r["acc"] for r in ms], mcc=[r["mcc"] for r in ms],
                             nn_dist_median=float(np.median(nd)), nn_dist_p05=float(np.percentile(nd, 5)))
    print(nm, "random", np.mean(out[nm]["random"]["macro_f1"]), flush=True)
    # ---- grouped 5-fold (days >= 3)
    idx = np.where(D.day >= 3)[0]
    g = D.groups()[idx]
    ms, nd = [], []
    for f, (a, b) in enumerate(GroupKFold(5).split(idx, groups=g)):
        tr, te = idx[a], idx[b]
        rng = np.random.default_rng(f)
        va = rng.choice(tr, size=len(tr) // 8, replace=False)          # early-stopping subset (grouped leakage negligible for ES)
        tr = np.setdiff1d(tr, va)
        m = make(nm, params, seed=f).fit(D, tr, va)
        ms.append(metrics(D.y[te], m.predict_proba(D, te), full=False))
        nd.append(nn_dist(np.concatenate([tr, va]), te, D.n))
    nd = np.concatenate(nd)
    out[nm]["grouped"] = dict(macro_f1=[r["macro_f1"] for r in ms], per_f1=np.mean([r["per_f1"] for r in ms], 0).tolist(),
                              acc=[r["acc"] for r in ms], mcc=[r["mcc"] for r in ms],
                              nn_dist_median=float(np.median(nd)), nn_dist_p05=float(np.percentile(nd, 5)))
    print(nm, "grouped", np.mean(out[nm]["grouped"]["macro_f1"]), flush=True)
    jdump(out, "protocols")
print("PROTOCOLS DONE", flush=True)
