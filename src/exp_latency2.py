"""Latency of the lagged XGBoost model (the best detector): single-window predict incl. assembly of the 340-input vector."""
import time
import numpy as np
from common import *
D = Data(DATA + "/siteA.pkl")
tr, va, te = D.chrono()
tuned = jload("tuning")
m = make("xgblag", tuned["xgblag"]["best"], 0).fit(D, tr, va)
rng = np.random.default_rng(0)
idx = [int(i) for i in rng.choice(te, 600, replace=False)]
for i in idx[:20]:
    m.predict_proba(D, np.array([i]))
t = []
for i in idx:
    a = time.perf_counter(); m.predict_proba(D, np.array([i])); t.append((time.perf_counter() - a) * 1000)
t = np.array(t)
out = dict(p50=float(np.percentile(t, 50)), p95=float(np.percentile(t, 95)), p99=float(np.percentile(t, 99)), mean=float(t.mean()), samples=t.tolist())
b = te[:5000]
a = time.perf_counter(); m.predict_proba(D, b); out["throughput_win_per_s"] = float(len(b) / (time.perf_counter() - a))
jdump(out, "latency_lag")
print({k: v for k, v in out.items() if k != "samples"})
