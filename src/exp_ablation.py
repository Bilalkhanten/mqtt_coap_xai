"""E4 feature-group ablation and E5 class-imbalance handling (XGBoost, 3 seeds, chronological split)."""
import copy
import numpy as np
from common import *

D = Data(DATA + "/siteA.pkl")
tr, va, te = D.chrono()
tuned = jload("tuning")
xp = dict(tuned["xgb"]["best"])
yte = D.y[te]
SEEDS = [0, 1, 2]


def subset(D, cols):
    S = copy.copy(D)
    S.X = D.X[:, cols]
    S.Z = D.Z[:, cols]
    return S


def run_xgb(S, seeds=SEEDS):
    ms, P = [], []
    for s in seeds:
        m = make("xgb", xp, seed=s).fit(S, tr, va)
        p = m.predict_proba(S, te)
        ms.append(metrics(yte, p, full=False))
        P.append(p)
    return dict(macro_f1=[r["macro_f1"] for r in ms], per_f1=np.mean([r["per_f1"] for r in ms], 0).tolist(),
                mcc=[r["mcc"] for r in ms], acc=[r["acc"] for r in ms],
                per_r=np.mean([r["per_r"] for r in ms], 0).tolist(), per_p=np.mean([r["per_p"] for r in ms], 0).tolist())


out = {}
allc = list(range(len(FEATURES)))
gcols = {g: [FEATURES.index(f) for f in fs] for g, fs in GROUPS.items()}
out["full"] = run_xgb(D)
print("full", np.mean(out["full"]["macro_f1"]), flush=True)
out["drop"], out["only"] = {}, {}
for g, cols in gcols.items():
    keep = [c for c in allc if c not in cols]
    out["drop"][g] = run_xgb(subset(D, keep))
    out["only"][g] = run_xgb(subset(D, cols))
    print(f"{g}: drop {np.mean(out['drop'][g]['macro_f1']):.4f}  only {np.mean(out['only'][g]['macro_f1']):.4f}", flush=True)
    jdump(out, "ablation")
net = [c for g in list(GROUPS)[:5] for c in gcols[g]]
en = [c for g in ["Energy residuals", "Energy bias (multi-scale)"] for c in gcols[g]]
out["combo"] = dict(network_only=run_xgb(subset(D, net)), energy_only=run_xgb(subset(D, en)),
                    no_context=run_xgb(subset(D, [c for c in allc if c not in gcols["Context"]])),
                    no_multiscale=run_xgb(subset(D, [c for c in allc if c not in gcols["Energy bias (multi-scale)"]])))
jdump(out, "ablation")

# ---------------------------------------------------------------- E5 imbalance handling
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.ensemble import RandomForestClassifier
imb = {}
rfp = dict(tuned["rf"]["best"])
for model in ["xgb", "rf"]:
    imb[model] = {}
    for strat in ["none", "class-weight", "SMOTE", "undersample", "SMOTE+class-weight"]:
        ms, ft = [], []
        for s in SEEDS:
            Xtr, ytr = D.X[tr], D.y[tr]
            sw = None
            if strat in ("SMOTE", "SMOTE+class-weight"):
                Xtr, ytr = SMOTE(random_state=s, k_neighbors=5).fit_resample(Xtr, ytr)
            elif strat == "undersample":
                Xtr, ytr = RandomUnderSampler(random_state=s).fit_resample(Xtr, ytr)
            use_w = strat in ("class-weight", "SMOTE+class-weight")
            t0 = time.time()
            if model == "xgb":
                m = XGB(seed=s, weighted=False, **xp)
                sw = class_weights(ytr, 0.5)[ytr] if use_w else np.ones(len(ytr))
                m.fit(D, tr, va, sample_weight=sw, Xtr=Xtr, ytr=ytr)
                p = m.predict_proba(D, te)
            else:
                kw = dict(rfp)
                kw["n_estimators"] = min(kw.get("n_estimators", 300), 200)
                rf = RandomForestClassifier(random_state=s, n_jobs=2, class_weight=("balanced_subsample" if use_w else None), **kw)
                rf.fit(Xtr, ytr)
                p = rf.predict_proba(D.X[te])
            ft.append(time.time() - t0)
            ms.append(metrics(yte, p, full=False))
        imb[model][strat] = dict(macro_f1=[r["macro_f1"] for r in ms], per_f1=np.mean([r["per_f1"] for r in ms], 0).tolist(),
                                 per_r=np.mean([r["per_r"] for r in ms], 0).tolist(), per_p=np.mean([r["per_p"] for r in ms], 0).tolist(),
                                 fit_s=float(np.mean(ft)), n_train=int(len(ytr)))
        print(model, strat, np.mean(imb[model][strat]["macro_f1"]), flush=True)
        jdump(imb, "imbalance")
print("ABLATION DONE", flush=True)
