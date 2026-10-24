import numpy as np, pandas as pd
from common import *
DA = Data(DATA + "/siteA.pkl"); DB = Data(DATA + "/siteB.pkl")
tr, va, te = DA.chrono(); tuned = jload("tuning")
m = make("xgb", tuned["xgb"]["best"], 0).fit(DA, tr, va)
te_b = np.where(DB.day >= 23)[0]
pr = m.predict_proba(DB, te_b).argmax(1)
y = DB.y[te_b]
print([c for c in DB.df.columns if c not in FEATURES][:30])
nm = y == 0
bv = DB.df.benign.values[te_b]
out = {"normal_windows": int(nm.sum()), "flagged_overall": float((pr[nm] > 0).mean()), "flagged_as": np.bincount(pr[nm], minlength=7).tolist(), "by_context": {}}
for nme, bit in [("none", 0), ("outage", 1), ("route", 2), ("reconnect", 4), ("admin", 8), ("fault", 16)]:
    mk = nm & ((bv == 0) if bit == 0 else ((bv & bit) > 0))
    out["by_context"][nme] = dict(n=int(mk.sum()), flagged=float((pr[mk] > 0).mean()) if mk.any() else None, as_mitm=float((pr[mk] == 4).mean()) if mk.any() else None)
out["shift"] = {}
for f in ["m_rtt_mean", "c_rtt_mean", "m_skew_med", "c_skew_med"]:
    i = FEATURES.index(f)
    a = DA.X[te][DA.y[te] == 0][:, i]; b = DB.X[te_b][nm][:, i]
    out["shift"][f] = dict(A_median=float(np.median(a)), B_median=float(np.median(b)))
# the same model, route-change windows at site A
pa = m.predict_proba(DA, te).argmax(1); ya = DA.y[te]; ba = DA.df.benign.values[te]
mk = (ya == 0) & ((ba & 2) > 0)
out["siteA_route"] = dict(n=int(mk.sum()), as_mitm=float((pa[mk] == 4).mean()))
jdump(out, "diag_siteb")
print(out)
