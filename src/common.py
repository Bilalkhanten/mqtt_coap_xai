"""Shared experiment utilities: data access, splits, preprocessing, models, metrics."""
import json, time, os, warnings
import numpy as np
import pandas as pd
from sklearn.metrics import (f1_score, precision_recall_fscore_support, matthews_corrcoef, cohen_kappa_score,
                             average_precision_score, roc_auc_score, balanced_accuracy_score, confusion_matrix)
from config import *

warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA, RES, FIGS = ROOT + "/data", ROOT + "/results", ROOT + "/figs"
K = len(CLASSES)

# ----------------------------------------------------------------------------
# feature groups
# ----------------------------------------------------------------------------
def _both(names):
    return [p + n for p in ("m_", "c_") for n in names]

GROUPS = {
    "Traffic volume": _both(["rate", "brate", "iat_cv", "iat_fast", "size_mean", "size_std", "out_in"]),
    "Source dynamics": _both(["n_src", "src_entropy", "top_src", "new_src"]),
    "Temporal integrity": _both(["rtt_mean", "ttl_dev", "gap_rate", "dup_rate", "skew_med", "skew_p95"]),
    "MQTT semantics": ["m_topic_entropy", "m_topics", "m_connect_rate", "m_subscribe_rate", "m_wildcard", "m_qos0_share"],
    "CoAP semantics": ["c_con_share", "c_notfound", "c_paths", "c_sports", "c_get_share"],
    "Energy residuals": ["e_bal_mean", "e_bal_max", "e_phys_m", "e_physx_m", "e_innov_m", "e_cusum_m", "e_flat_m",
                         "e_phys_c", "e_physx_c", "e_innov_c", "e_cusum_c", "e_flat_c", "e_peer"],
    "Energy bias (multi-scale)": ["e_bal_bias", "e_bal_bias10", "e_bias_m", "e_bias10_m", "e_bias_c", "e_bias10_c",
                                  "e_peer_bias", "e_peer_bias10"],
    "Context": ["tod_sin", "tod_cos"],
}
FEATURES = [f for g in GROUPS.values() for f in g]
GROUP_OF = {f: g for g, fs in GROUPS.items() for f in fs}
GROUP_COLORS = ["#4C6EF5", "#15AABF", "#12B886", "#82C91E", "#FAB005", "#FD7E14", "#E64980", "#868E96"]


class Data:
    """One simulated site: feature matrices, labels, and commissioning-calibrated preprocessing."""

    def __init__(self, path, nc=3):
        self.df = pd.read_pickle(path)
        df = self.df
        missing = [f for f in FEATURES if f not in df.columns]
        assert not missing, missing
        self.y = df.label.values.astype(int)
        self.day = df.day.values
        self.n = len(df)
        raw = df[FEATURES].values.astype(np.float64)
        comm = self.day < nc
        med = np.nanmedian(raw[comm], axis=0)
        # forward-fill (within the whole series) then commissioning-median fill
        raw = pd.DataFrame(raw).ffill().values
        raw = np.where(np.isnan(raw), med, raw)
        self.X = raw.astype(np.float32)                          # tree models
        tf = np.where(np.arange(len(FEATURES)) < 0, raw, np.log1p(np.maximum(raw, 0)))
        ctx = [FEATURES.index(f) for f in GROUPS["Context"]]
        tf[:, ctx] = raw[:, ctx]
        mu, sd = tf[comm].mean(0), tf[comm].std(0) + 1e-6
        self.Z = np.clip((tf - mu) / sd, -10, 10).astype(np.float32)   # neural / linear models
        self.blocks = np.arange(self.n) // 20                          # 10-minute blocks
        self.is_benign = (df.benign.values > 0) & (self.y == 0)

    # --- split protocols -------------------------------------------------------
    def chrono(self, gap=24):
        """Chronological split with a 12-minute purge gap before each later segment.
        commissioning: days 0-2 (never used for fitting) | train: 3-19 | validation: 20-22 | test: 23-29"""
        d, w = self.day, np.arange(self.n)
        tr = (d >= 3) & (d < 20)
        va = (d >= 20) & (d < 23)
        te = d >= 23
        b1, b2 = 20 * WINS_PER_DAY, 23 * WINS_PER_DAY
        tr &= ~((w >= b1 - gap) & (w < b1))
        va &= ~((w >= b2 - gap) & (w < b2))
        return np.where(tr)[0], np.where(va)[0], np.where(te)[0]

    def random_split(self, seed=0):
        rng = np.random.default_rng(seed)
        idx = np.where(self.day >= 3)[0]
        y = self.y[idx]
        tr, va, te = [], [], []
        for c in range(K):
            ii = rng.permutation(idx[y == c])
            n = len(ii)
            tr += list(ii[: int(.7 * n)]); va += list(ii[int(.7 * n):int(.8 * n)]); te += list(ii[int(.8 * n):])
        return np.array(sorted(tr)), np.array(sorted(va)), np.array(sorted(te))

    def groups(self):
        """Group id = burst id for any window touching an attack burst, else the 1-hour block."""
        df = self.df
        return np.where(df.burst.values >= 0, df.burst.values, 100000 + df.win.values // 120)

    def seq(self, idx, T=5, Z=None):
        Z = self.Z if Z is None else Z
        ii = np.asarray(idx)[:, None] - np.arange(T - 1, -1, -1)[None, :]
        ii = np.clip(ii, 0, self.n - 1)
        return Z[ii]

    def lag(self, idx, T=5):
        return self.seq(idx, T, self.X).reshape(len(idx), -1)


def class_weights(y, power=1.0):
    cnt = np.bincount(y, minlength=K).astype(float)
    w = (len(y) / (K * np.maximum(cnt, 1))) ** power
    return w


# ----------------------------------------------------------------------------
# metrics
# ----------------------------------------------------------------------------
def ece_top(proba, y, bins=15):
    conf = proba.max(1)
    pred = proba.argmax(1)
    acc = (pred == y).astype(float)
    e = 0.0
    edges = np.linspace(0, 1, bins + 1)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            e += m.mean() * abs(acc[m].mean() - conf[m].mean())
    return float(e)


def metrics(y, proba, full=True):
    pred = proba.argmax(1)
    p, r, f, s = precision_recall_fscore_support(y, pred, labels=range(K), zero_division=0)
    out = dict(acc=float((pred == y).mean()), bal_acc=float(balanced_accuracy_score(y, pred)),
               macro_p=float(p.mean()), macro_r=float(r.mean()), macro_f1=float(f.mean()),
               weighted_f1=float(np.average(f, weights=s)), mcc=float(matthews_corrcoef(y, pred)),
               kappa=float(cohen_kappa_score(y, pred)), per_p=p.tolist(), per_r=r.tolist(), per_f1=f.tolist(),
               support=s.tolist())
    if full:
        oh = np.eye(K)[y]
        ap = [average_precision_score(oh[:, c], proba[:, c]) if oh[:, c].sum() > 0 else np.nan for c in range(K)]
        auc = [roc_auc_score(oh[:, c], proba[:, c]) if 0 < oh[:, c].sum() < len(y) else np.nan for c in range(K)]
        out.update(per_ap=ap, per_auc=auc, macro_ap=float(np.nanmean(ap)), macro_auc=float(np.nanmean(auc)),
                   ece=ece_top(proba, y), brier=float(((proba - oh) ** 2).sum(1).mean()),
                   attack_f1=float(f[1:].mean()))
        out["fpr_normal"] = float(((pred != 0) & (y == 0)).sum() / max((y == 0).sum(), 1))
    return out


def block_bootstrap_ci(y, pred, blocks, fn, B=1000, seed=0, alpha=0.05):
    """Moving-block (cluster) bootstrap of a confusion-matrix statistic. `fn(cm)->float`."""
    rng = np.random.default_rng(seed)
    ub = np.unique(blocks)
    cms = np.zeros((len(ub), K, K))
    pos = {b: i for i, b in enumerate(ub)}
    bi = np.array([pos[b] for b in blocks])
    np.add.at(cms, (bi, y, pred), 1)
    vals = []
    for _ in range(B):
        s = rng.integers(0, len(ub), len(ub))
        vals.append(fn(cms[s].sum(0)))
    return float(np.percentile(vals, 100 * alpha / 2)), float(np.percentile(vals, 100 * (1 - alpha / 2)))


def macro_f1_cm(cm):
    tp = np.diag(cm)
    p = tp / np.maximum(cm.sum(0), 1)
    r = tp / np.maximum(cm.sum(1), 1)
    f = 2 * p * r / np.maximum(p + r, 1e-12)
    return f.mean()


def block_bootstrap_diff(y, pa, pb, blocks, B=1000, seed=0):
    rng = np.random.default_rng(seed)
    ub = np.unique(blocks)
    pos = {b: i for i, b in enumerate(ub)}
    bi = np.array([pos[b] for b in blocks])
    ca, cb = np.zeros((len(ub), K, K)), np.zeros((len(ub), K, K))
    np.add.at(ca, (bi, y, pa), 1)
    np.add.at(cb, (bi, y, pb), 1)
    d = []
    for _ in range(B):
        s = rng.integers(0, len(ub), len(ub))
        d.append(macro_f1_cm(ca[s].sum(0)) - macro_f1_cm(cb[s].sum(0)))
    d = np.array(d)
    p = 2 * min((d <= 0).mean(), (d >= 0).mean())
    return float(np.mean(d)), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5)), float(max(p, 1.0 / B))


def jdump(obj, name):
    def conv(o):
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(type(o))
    with open(f"{RES}/{name}.json", "w") as f:
        json.dump(obj, f, indent=1, default=conv)


def jload(name):
    with open(f"{RES}/{name}.json") as f:
        return json.load(f)


# ----------------------------------------------------------------------------
# models  (all expose fit(D, tr, va) / predict_proba(D, idx))
# ----------------------------------------------------------------------------
class LR:
    name = "Logistic Regression"

    def __init__(self, C=1.0, seed=0, **kw):
        self.C, self.seed = C, seed

    def fit(self, D, tr, va):
        from sklearn.linear_model import LogisticRegression
        self.m = LogisticRegression(C=self.C, max_iter=400, class_weight="balanced")
        self.m.fit(D.Z[tr], D.y[tr])
        return self

    def predict_proba(self, D, idx):
        return self.m.predict_proba(D.Z[idx])


class RF:
    name = "Random Forest"

    def __init__(self, n_estimators=300, max_depth=18, min_samples_leaf=2, max_features="sqrt", seed=0, cw="balanced_subsample", **kw):
        self.kw = dict(n_estimators=n_estimators, max_depth=max_depth, min_samples_leaf=min_samples_leaf,
                       max_features=max_features, random_state=seed, n_jobs=2, class_weight=cw)

    def fit(self, D, tr, va):
        from sklearn.ensemble import RandomForestClassifier
        self.m = RandomForestClassifier(**self.kw).fit(D.X[tr], D.y[tr])
        return self

    def predict_proba(self, D, idx):
        return self.m.predict_proba(D.X[idx])


class XGB:
    name = "XGBoost"

    def __init__(self, n_estimators=400, max_depth=6, learning_rate=0.08, subsample=0.8, colsample_bytree=0.8,
                 reg_lambda=1.0, gamma=0.0, min_child_weight=1.0, seed=0, weighted=True, lag=False, **kw):
        self.p = dict(n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate, subsample=subsample,
                      colsample_bytree=colsample_bytree, reg_lambda=reg_lambda, gamma=gamma,
                      min_child_weight=min_child_weight, random_state=seed)
        self.weighted, self.lag = weighted, lag
        if lag:
            self.name = "XGBoost (5-window lag)"

    def _x(self, D, idx):
        return D.lag(idx) if self.lag else D.X[idx]

    def fit(self, D, tr, va, sample_weight=None, Xtr=None, ytr=None):
        import xgboost as xgb
        Xt = self._x(D, tr) if Xtr is None else Xtr
        yt = D.y[tr] if ytr is None else ytr
        sw = class_weights(yt, 0.5)[yt] if (self.weighted and sample_weight is None) else sample_weight
        self.m = xgb.XGBClassifier(objective="multi:softprob", tree_method="hist", n_jobs=2, eval_metric="mlogloss",
                                   early_stopping_rounds=30, verbosity=0, **self.p)
        sv = class_weights(D.y[va], 0.5)[D.y[va]] if self.weighted else None
        self.m.fit(Xt, yt, sample_weight=sw, eval_set=[(self._x(D, va), D.y[va])], sample_weight_eval_set=[sv] if sv is not None else None, verbose=False)
        return self

    def predict_proba(self, D, idx):
        return self.m.predict_proba(self._x(D, idx))


class LGBM:
    name = "LightGBM"

    def __init__(self, n_estimators=400, num_leaves=31, learning_rate=0.08, subsample=0.8, colsample_bytree=0.8,
                 reg_lambda=1.0, min_child_samples=20, seed=0, **kw):
        self.p = dict(n_estimators=n_estimators, num_leaves=num_leaves, learning_rate=learning_rate, subsample=subsample,
                      subsample_freq=1, colsample_bytree=colsample_bytree, reg_lambda=reg_lambda,
                      min_child_samples=min_child_samples, random_state=seed)

    def fit(self, D, tr, va):
        import lightgbm as lgb
        yt = D.y[tr]
        sw = class_weights(yt, 0.5)[yt]
        self.m = lgb.LGBMClassifier(objective="multiclass", n_jobs=2, verbose=-1, **self.p)
        self.m.fit(D.X[tr], yt, sample_weight=sw, eval_set=[(D.X[va], D.y[va])],
                   callbacks=[lgb.early_stopping(30, verbose=False)])
        return self

    def predict_proba(self, D, idx):
        return self.m.predict_proba(D.X[idx])


def _torch():
    import torch
    torch.set_num_threads(2)
    return torch


class _NNBase:
    kind = "mlp"

    def __init__(self, hidden=128, depth=2, dropout=0.2, lr=1e-3, wd=1e-4, epochs=40, patience=6, batch=256, seed=0,
                 T=5, conv=64, lstm=100, **kw):
        self.h = dict(hidden=hidden, depth=depth, dropout=dropout, lr=lr, wd=wd, epochs=epochs, patience=patience,
                      batch=batch, seed=seed, T=T, conv=conv, lstm=lstm)
        self.temperature = 1.0

    def _build(self, F):
        torch = _torch()
        nn = torch.nn
        h = self.h
        if self.kind == "mlp":
            layers, d = [], F
            for _ in range(h["depth"]):
                layers += [nn.Linear(d, h["hidden"]), nn.ReLU(), nn.Dropout(h["dropout"])]
                d = h["hidden"]
            layers += [nn.Linear(d, K)]
            return nn.Sequential(*layers)

        class CNNLSTM(nn.Module):
            def __init__(s):
                super().__init__()
                s.c1 = nn.Conv1d(F, h["conv"], 3, padding=1)
                s.c2 = nn.Conv1d(h["conv"], 2 * h["conv"], 3, padding=1)
                s.lstm = nn.LSTM(2 * h["conv"], h["lstm"], batch_first=True)
                s.fc1 = nn.Linear(h["lstm"], 64)
                s.dp = nn.Dropout(h["dropout"])
                s.fc2 = nn.Linear(64, K)

            def forward(s, x):                      # x: (B, T, F)
                z = torch.relu(s.c1(x.transpose(1, 2)))
                z = torch.relu(s.c2(z)).transpose(1, 2)
                o, _ = s.lstm(z)
                return s.fc2(s.dp(torch.relu(s.fc1(o[:, -1]))))

        return CNNLSTM()

    def _x(self, D, idx):
        return D.seq(idx, self.h["T"]) if self.kind == "cnnlstm" else D.Z[idx]

    def fit(self, D, tr, va):
        torch = _torch()
        h = self.h
        torch.manual_seed(h["seed"]); np.random.seed(h["seed"])
        Xt, yt = torch.from_numpy(self._x(D, tr)), torch.from_numpy(D.y[tr])
        Xv = torch.from_numpy(self._x(D, va))
        self.net = self._build(Xt.shape[-1])
        w = torch.tensor(class_weights(D.y[tr], 0.5), dtype=torch.float32)
        lossf = torch.nn.CrossEntropyLoss(weight=w)
        opt = torch.optim.Adam(self.net.parameters(), lr=h["lr"], weight_decay=h["wd"])
        best, bad, best_state = -1, 0, None
        self.curve = []
        for ep in range(h["epochs"]):
            self.net.train()
            perm = torch.randperm(len(Xt))
            tl = 0.0
            for i in range(0, len(Xt), h["batch"]):
                b = perm[i:i + h["batch"]]
                opt.zero_grad()
                loss = lossf(self.net(Xt[b]), yt[b])
                loss.backward(); opt.step()
                tl += loss.item() * len(b)
            self.net.eval()
            with torch.no_grad():
                lv = self.net(Xv)
                vl = torch.nn.functional.cross_entropy(lv, torch.from_numpy(D.y[va]), weight=w).item()
                pv = lv.argmax(1).numpy()
            f1 = f1_score(D.y[va], pv, average="macro")
            self.curve.append((ep + 1, tl / len(Xt), vl, f1))
            if f1 > best:
                best, bad = f1, 0
                best_state = {k: v.clone() for k, v in self.net.state_dict().items()}
            else:
                bad += 1
                if bad >= h["patience"]:
                    break
        self.net.load_state_dict(best_state)
        self.net.eval()
        return self

    def logits(self, D, idx, bs=8192):
        torch = _torch()
        out = []
        with torch.no_grad():
            for i in range(0, len(idx), bs):
                out.append(self.net(torch.from_numpy(self._x(D, idx[i:i + bs]))).numpy())
        return np.concatenate(out)

    def predict_proba(self, D, idx):
        z = self.logits(D, np.asarray(idx)) / self.temperature
        z = z - z.max(1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(1, keepdims=True)


class MLP(_NNBase):
    name = "MLP"
    kind = "mlp"


class CNNLSTM(_NNBase):
    name = "CNN-LSTM"
    kind = "cnnlstm"


MODELS = dict(lr=LR, rf=RF, xgb=XGB, lgbm=LGBM, mlp=MLP, cnnlstm=CNNLSTM)
MODEL_NAMES = {"lr": "Logistic Regression", "rf": "Random Forest", "xgb": "XGBoost", "lgbm": "LightGBM", "mlp": "MLP",
               "cnnlstm": "CNN-LSTM", "xgblag": "XGBoost (5-window lag)"}


def make(name, params=None, seed=0):
    params = dict(params or {})
    if name == "xgblag":
        return XGB(lag=True, seed=seed, **params)
    return MODELS[name](seed=seed, **params)


def timed(fn, *a, **k):
    t = time.time()
    r = fn(*a, **k)
    return r, time.time() - t
