"""E7 unseen-FDIA-variant generalisation, E8 problem-space evasion (attack-intensity scaling), E10 cross-site transfer."""
import dataclasses, copy
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from common import *
from simulate import simulate
from config import SITE_A, SITE_B

DA = Data(DATA + "/siteA.pkl")
DB = Data(DATA + "/siteB.pkl")
tr, va, te = DA.chrono()
tuned = jload("tuning")
yte = DA.y[te]
out = {}
VARS = ["V1-bias", "V2-ramp", "V3-balanced"]

# ------------------------------------------------------------------ E7 leave-one-variant-out
e7 = {}
for nm in ["xgb", "rf"]:
    e7[nm] = {}
    for v in ["none"] + VARS:
        rec, f1s = [], []
        for s in range(3):
            drop = (DA.y == 1) & (DA.df.variant.values == v) if v != "none" else np.zeros(DA.n, bool)
            trv = tr[~drop[tr]]
            m = make(nm, tuned[nm]["best"], seed=s).fit(DA, trv, va)
            pr = m.predict_proba(DA, te).argmax(1)
            f1s.append(f1_score(yte, pr, average="macro"))
            r = {}
            for vv in VARS:
                mm = (DA.df.variant.values[te] == vv) & (yte == 1)
                r[vv] = float((pr[mm] == 1).mean())
            rec.append(r)
        e7[nm][v] = dict(macro_f1=f1s, recall={vv: [r[vv] for r in rec] for vv in VARS})
        print("E7", nm, "held-out", v, "macroF1 %.3f" % np.mean(f1s), {vv: round(np.mean([r[vv] for r in rec]), 3) for vv in VARS}, flush=True)
out["e7"] = e7
jdump(out, "robust")

# ------------------------------------------------------------------ E8 evasion by attack-intensity scaling
base_models = {nm: make(nm, tuned[nm]["best"], seed=0).fit(DA, tr, va) for nm in ["xgb", "rf", "cnnlstm", "lr"]}
SCALES = [1.0, 0.5, 0.25, 0.1, 0.05]
e8 = {}
for s in SCALES:
    sc = dict(rate=s, bias=s, frac=s)
    df_s, _, _ = simulate(SITE_A, scale=sc, verbose=False)
    path = DATA + f"/siteA_scale{s}.pkl"
    df_s.to_pickle(path)
    DS = Data(path)
    # shared commissioning-calibration of the *base* site must be reused: scaled sims share plant and commissioning
    te_s = np.where(DS.day >= 23)[0]
    e8[str(s)] = {}
    for nm, m in base_models.items():
        P = m.predict_proba(DS, te_s)
        pr = P.argmax(1)
        y = DS.y[te_s]
        e8[str(s)][nm] = dict(macro_f1=float(f1_score(y, pr, average="macro")), per_r=precision_recall_fscore_support(y, pr, labels=range(K), zero_division=0)[1].tolist(),
                              per_p=precision_recall_fscore_support(y, pr, labels=range(K), zero_division=0)[0].tolist(),
                              per_f1=f1_score(y, pr, average=None, labels=range(K)).tolist())
    print("E8 scale", s, {nm: round(e8[str(s)][nm]["macro_f1"], 3) for nm in base_models}, flush=True)
    out["e8"] = e8
    jdump(out, "robust")

# ------------------------------------------------------------------ E10 cross-site transfer + few-shot adaptation
te_b = np.where(DB.day >= 23)[0]
yb = DB.y[te_b]
e10 = {"zero_shot": {}, "few_shot": {}}
for nm, m in base_models.items():
    P = m.predict_proba(DB, te_b)
    e10["zero_shot"][nm] = metrics(yb, P, full=False)
    print("E10 zero-shot", nm, round(e10["zero_shot"][nm]["macro_f1"], 3), np.round(e10["zero_shot"][nm]["per_f1"], 2), flush=True)
out["e10"] = e10
jdump(out, "robust")
for k in [0, 1, 2, 4, 8]:
    e10["few_shot"][str(k)] = {}
    adapt = np.where((DB.day >= 3) & (DB.day < 3 + k))[0]
    for nm in ["xgb", "rf"]:
        Xtr = np.vstack([DA.X[tr], DB.X[adapt]]) if k else DA.X[tr]
        ytr = np.concatenate([DA.y[tr], DB.y[adapt]]) if k else DA.y[tr]
        if nm == "xgb":
            m = XGB(seed=0, **tuned["xgb"]["best"])
            m.fit(DA, tr, va, Xtr=Xtr, ytr=ytr)
            P = m.m.predict_proba(DB.X[te_b])
        else:
            kw = dict(tuned["rf"]["best"]); kw["n_estimators"] = min(kw.get("n_estimators", 300), 200)
            rf = RandomForestClassifier(random_state=0, n_jobs=2, class_weight="balanced_subsample", **kw).fit(Xtr, ytr)
            P = rf.predict_proba(DB.X[te_b])
        e10["few_shot"][str(k)][nm] = metrics(yb, P, full=False)
        print("E10 few-shot k=%d" % k, nm, round(e10["few_shot"][str(k)][nm]["macro_f1"], 3), flush=True)
    out["e10"] = e10
    jdump(out, "robust")
print("ROBUST DONE", flush=True)
