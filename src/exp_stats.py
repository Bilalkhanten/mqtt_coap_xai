"""Statistics on the main benchmark: Friedman/Nemenyi, Holm-corrected pairwise block bootstrap, error anatomy."""
import itertools
import numpy as np
from scipy.stats import friedmanchisquare, rankdata
from common import *

D = Data(DATA + "/siteA.pkl")
te = np.load(f"{RES}/test_idx.npy")
P = np.load(f"{RES}/main_probs.npz")
res = jload("main")
names = list(P.keys())
y = D.y[te]
blocks = D.blocks[te]
out = {}

# ---- Friedman on per-day macro-F1 (7 days x k models)
M = np.array([res[n]["per_day_f1"] for n in names]).T          # (7, k)
chi, p = friedmanchisquare(*[M[:, j] for j in range(M.shape[1])])
ranks = np.array([rankdata(-row) for row in M])
avg = ranks.mean(0)
k, N = len(names), M.shape[0]
q = {5: 2.728, 6: 2.850, 7: 2.949, 8: 3.031}[k]
cd = q * np.sqrt(k * (k + 1) / (6 * N))
out["friedman"] = dict(chi2=float(chi), p=float(p), k=k, N=N, avg_rank=dict(zip(names, avg.tolist())), cd=float(cd), q=q)

# ---- pairwise block bootstrap with Holm
pairs = list(itertools.combinations(names, 2))
rows = []
for a, b in pairs:
    d, lo, hi, pv = block_bootstrap_diff(y, P[a][0].argmax(1), P[b][0].argmax(1), blocks, B=2000)
    rows.append(dict(a=a, b=b, diff=d, lo=lo, hi=hi, p=pv))
order = np.argsort([r["p"] for r in rows])
m = len(rows)
prev = 0
for rank, i in enumerate(order):
    adj = min(1.0, (m - rank) * rows[i]["p"])
    prev = max(prev, adj)
    rows[i]["p_holm"] = prev
out["pairwise"] = rows

# ---- error anatomy for the best model (by seed-0 macro-F1) --------------------------------
best = max(names, key=lambda n: res[n]["agg"]["macro_f1"]["mean"])
out["best_model"] = best
pr = P[best][0].argmax(1)
df = D.df.iloc[te].reset_index(drop=True)
# FDIA recall by variant
fd = {}
for v in ["V1-bias", "V2-ramp", "V3-balanced"]:
    mk = (y == 1) & (df.variant.values == v)
    fd[v] = dict(n=int(mk.sum()), recall=float((pr[mk] == 1).mean()) if mk.any() else None,
                 as_normal=float((pr[mk] == 0).mean()) if mk.any() else None)
out["fdia_by_variant"] = fd
# recall by scope (mqtt/coap) per class
sc = {}
for c in range(1, K):
    sc[CLASSES[c]] = {}
    for s in ["mqtt", "coap", "both"]:
        mk = (y == c) & (df.scope.values == s)
        if mk.sum() > 0:
            sc[CLASSES[c]][s] = dict(n=int(mk.sum()), recall=float((pr[mk] == c).mean()))
out["recall_by_scope"] = sc
# false-positive rate by benign type among Normal windows
fp = {}
nm = y == 0
for nme, bit in [("none", 0), ("outage", 1), ("route", 2), ("reconnect", 4), ("admin", 8), ("fault", 16)]:
    mk = nm & ((df.benign.values == 0) if bit == 0 else ((df.benign.values & bit) > 0))
    fp[nme] = dict(n=int(mk.sum()), fpr=float((pr[mk] != 0).mean()) if mk.any() else None)
out["fpr_by_benign"] = fp
# detection delay: windows from burst start to first correct detection (per burst)
bursts = df.burst.values
delays = {c: [] for c in range(1, K)}
missed = {c: 0 for c in range(1, K)}
for b in np.unique(bursts[bursts >= 0]):
    ii = np.where(bursts == b)[0]
    c = int(y[ii[0]]) if y[ii[0]] > 0 else int(np.bincount(y[ii]).argmax())
    if c == 0:
        continue
    hit = np.where(pr[ii] == c)[0]
    if len(hit):
        delays[c].append(int(hit[0]))
    else:
        missed[c] += 1
out["detection_delay_windows"] = {CLASSES[c]: dict(n=len(delays[c]) + missed[c], missed=missed[c],
                                  median=(float(np.median(delays[c])) if delays[c] else None),
                                  p90=(float(np.percentile(delays[c], 90)) if delays[c] else None)) for c in range(1, K)}
# burst-level recall
out["burst_detected_frac"] = {CLASSES[c]: (1 - missed[c] / max(1, len(delays[c]) + missed[c])) for c in range(1, K)}
# confusion matrix (seed 0 + ensemble)
cm = np.zeros((K, K), int)
np.add.at(cm, (y, pr), 1)
out["cm_best"] = cm.tolist()
for n in names:
    c_ = np.zeros((K, K), int)
    np.add.at(c_, (y, P[n][0].argmax(1)), 1)
    out.setdefault("cm", {})[n] = c_.tolist()
jdump(out, "stats")
print("best", best, "friedman", out["friedman"]["chi2"], out["friedman"]["p"], "CD", cd)
print("STATS DONE")
