"""Latency / throughput: per-window inference and explanation cost (measured, single process)."""
import platform, os, time, sys
import numpy as np
import shap
from common import *

D = Data(DATA + "/siteA.pkl")
tr, va, te = D.chrono()
tuned = jload("tuning")
rng = np.random.default_rng(0)
idx = rng.choice(te, 600, replace=False)
out = dict(hardware=dict(cpu_count=os.cpu_count(), platform=platform.platform(), python=sys.version.split()[0]))
try:
    out["hardware"]["cpu_model"] = [l.split(":")[1].strip() for l in open("/proc/cpuinfo") if l.startswith("model name")][0]
except Exception:
    out["hardware"]["cpu_model"] = "unknown"


def lat(fn, items, warm=20):
    for it in items[:warm]:
        fn(it)
    t = []
    for it in items:
        a = time.perf_counter()
        fn(it)
        t.append((time.perf_counter() - a) * 1000)
    t = np.array(t)
    return dict(p50=float(np.percentile(t, 50)), p95=float(np.percentile(t, 95)), p99=float(np.percentile(t, 99)), mean=float(t.mean()), samples=t.tolist())


models = {n: make(n, tuned[n]["best"], 0).fit(D, tr, va) for n in ["lr", "rf", "xgb", "lgbm", "mlp", "cnnlstm"]}
X1 = [D.X[i:i + 1] for i in idx]
res = {}
for n, m in models.items():
    if n in ("lr", "rf", "xgb", "lgbm"):
        fn = (lambda x, m=m: m.m.predict_proba(x))
    else:
        fn = (lambda i, m=m: m.predict_proba(D, np.array([i])))
        X1_ = [int(i) for i in idx]
    items = X1 if n in ("lr", "rf", "xgb", "lgbm") else [int(i) for i in idx]
    res[n] = lat(fn, items)
    print(n, {k: round(v, 3) for k, v in res[n].items() if k != "samples"}, flush=True)
out["predict_ms"] = res

# batch throughput
thr = {}
for n, m in models.items():
    b = te[:5000]
    a = time.perf_counter()
    m.predict_proba(D, b)
    dt = time.perf_counter() - a
    thr[n] = float(len(b) / dt)
out["throughput_win_per_s"] = thr

# TreeSHAP explanation latency (per window)
ex = shap.TreeExplainer(models["xgb"].m)
out["treeshap_xgb_ms"] = lat(lambda x: ex.shap_values(x), X1[:300])
exr = shap.TreeExplainer(models["rf"].m)
out["treeshap_rf_ms"] = lat(lambda x: exr.shap_values(x, check_additivity=False), X1[:100], warm=5)
print("treeshap xgb", {k: round(v, 2) for k, v in out["treeshap_xgb_ms"].items() if k != "samples"}, flush=True)

# feature extraction latency: network features (both transports) for one day of messages + energy features for the whole horizon
from simulate import simulate
from config import SITE_A, WINS_PER_DAY, DT
from features import tr_features
df_, meta_, (truth_, meas_, O_, aux_) = simulate(SITE_A, only_days=[25], verbose=False, return_msgs_day=[25])
M = meta_["msgs"][25]
t0 = 25 * 86400
reps = []
for _ in range(3):
    a_ = time.perf_counter()
    tr_features(M, 0, t0, WINS_PER_DAY, meta_["base_ttl"], meta_["base_skew"])
    tr_features(M, 1, t0, WINS_PER_DAY, meta_["base_ttl"], meta_["base_skew"])
    reps.append(time.perf_counter() - a_)
net_day = float(np.median(reps))
em = meta_["em"]
a_ = time.perf_counter()
em.window_features(O_)
en_all = time.perf_counter() - a_
nwin_all = O_.shape[0] // 6
out["messages_day25"] = int(len(M["t"]))
out["feat_network_ms_per_window"] = net_day / WINS_PER_DAY * 1000
out["feat_energy_ms_per_window"] = en_all / nwin_all * 1000
out["feature_extraction_per_window_ms"] = out["feat_network_ms_per_window"] + out["feat_energy_ms_per_window"]
out["windows_per_day"] = WINS_PER_DAY
out["window_seconds"] = 30
out["end_to_end_xgb_treeshap_p95_ms"] = float(out["predict_ms"]["xgb"]["p95"] + out["treeshap_xgb_ms"]["p95"] + out["feature_extraction_per_window_ms"])
jdump(out, "latency")
print("LATENCY DONE", flush=True)
