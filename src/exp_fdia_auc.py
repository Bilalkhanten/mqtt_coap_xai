"""Univariate detectability of the three FDIA variants: ROC-AUC of every feature (Normal vs. variant), days >= 3."""
import numpy as np
from sklearn.metrics import roc_auc_score
from common import *
D = Data(DATA + "/siteA.pkl")
df = D.df
m = (df.day >= 3).values
out = {}
nm = m & (df.label.values == 0)
for v in ["V1-bias", "V2-ramp", "V3-balanced"]:
    mv = m & (df.label.values == 1) & (df.variant.values == v)
    res = {}
    for f in FEATURES:
        x = np.concatenate([df.loc[nm, f].values, df.loc[mv, f].values]).astype(float)
        y = np.concatenate([np.zeros(nm.sum()), np.ones(mv.sum())])
        ok = np.isfinite(x)
        a = roc_auc_score(y[ok], x[ok])
        res[f] = float(max(a, 1 - a))
    top = sorted(res.items(), key=lambda kv: -kv[1])[:5]
    out[v] = dict(n=int(mv.sum()), top=top, best=top[0][1], all=res)
    print(v, mv.sum(), [(k, round(a, 3)) for k, a in top])
jdump(out, "fdia_auc")
