"""Window-level feature extraction.

Network features are computed from the message stream seen at the gateway,
separately for MQTT (prefix m_) and CoAP (prefix c_).  Energy features are
computed from the control-centre view of the sensor matrix after calibrating a
physics-based consistency model on a clean commissioning interval.
"""
import numpy as np
import pandas as pd
from config import *
from traffic import *

KEYB = 1 << 24
PV, WI = [0, 1, 2], [3, 4]
MQ_S = [i for i in range(N_SENS) if SENS_TRANSPORT[i] == 0]
CO_S = [i for i in range(N_SENS) if SENS_TRANSPORT[i] == 1]


def _bc(w, nw, x=None):
    return np.bincount(w, weights=x, minlength=nw)


def _uniq(w, key, nw):
    code = w.astype(np.int64) * KEYB + key.astype(np.int64)
    u, c = np.unique(code, return_counts=True)
    return (u >> 24).astype(np.int64), (u & (KEYB - 1)).astype(np.int64), c


def _entropy(wu, c, cnt, nw):
    p = c / cnt[wu]
    return _bc(wu, nw, -p * np.log2(p))


def _qwin(w, x, q, nw):
    o = np.lexsort((x, w))
    xs = x[o]
    cnt = np.bincount(w, minlength=nw)
    start = np.concatenate([[0], np.cumsum(cnt)[:-1]])
    pos = start + np.floor(q * np.maximum(cnt - 1, 0)).astype(int)
    out = np.full(nw, np.nan)
    ok = cnt > 0
    out[ok] = xs[np.minimum(pos[ok], len(xs) - 1)]
    return out


def calibrate_network(msgs_comm):
    """Per-source median time-stamp skew from the clean commissioning interval."""
    base = np.zeros(200)
    src, sk = msgs_comm["src"], msgs_comm["skew"]
    for s in range(N_SENS):
        m = (src == s) & ~np.isnan(sk)
        if m.sum() > 10:
            base[s] = np.median(sk[m])
    return base


def tr_features(M, tr, t0, nw, base_ttl, base_skew):
    pre = "m_" if tr == 0 else "c_"
    m = M["tr"] == tr
    t = M["t"][m]
    o = np.argsort(t, kind="stable")
    g = {k: M[k][m][o] for k in FIELDS}
    w = np.floor((g["t"] - t0) / WIN).astype(np.int64)
    ok = (w >= 0) & (w < nw)
    g = {k: v[ok] for k, v in g.items()}
    w = w[ok]
    t = g["t"]
    cnt = np.bincount(w, minlength=nw).astype(float)
    F = {}
    nz = np.maximum(cnt, 1)
    F["rate"] = cnt / WIN
    F["brate"] = _bc(w, nw, g["sin"].astype(float) + g["sout"]) / WIN
    # inter-arrival statistics
    if len(t) > 1:
        dt = np.diff(t)
        same = w[1:] == w[:-1]
        dts, wi = dt[same], w[1:][same]
        ni = np.bincount(wi, minlength=nw).astype(float)
        s1, s2 = _bc(wi, nw, dts), _bc(wi, nw, dts ** 2)
        mu = s1 / np.maximum(ni, 1)
        var = np.maximum(s2 / np.maximum(ni, 1) - mu ** 2, 0)
        F["iat_cv"] = np.where(ni > 1, np.sqrt(var) / np.maximum(mu, 1e-9), np.nan)
        F["iat_fast"] = np.where(ni > 0, _bc(wi[dts < 0.01], nw) / np.maximum(ni, 1), np.nan)
    else:
        F["iat_cv"] = np.full(nw, np.nan)
        F["iat_fast"] = np.full(nw, np.nan)
    sin = g["sin"].astype(float)
    sm = _bc(w, nw, sin) / nz
    F["size_mean"] = np.where(cnt > 0, sm, np.nan)
    F["size_std"] = np.where(cnt > 0, np.sqrt(np.maximum(_bc(w, nw, sin ** 2) / nz - sm ** 2, 0)), np.nan)
    F["out_in"] = np.where(cnt > 0, _bc(w, nw, g["sout"].astype(float)) / np.maximum(_bc(w, nw, sin), 1), np.nan)
    # sources
    wu, su, c = _uniq(w, g["src"], nw)
    F["n_src"] = _bc(wu, nw)
    F["src_entropy"] = np.where(cnt > 0, _entropy(wu, c, cnt, nw), np.nan)
    top = pd.Series(c).groupby(wu).max()
    ts_ = np.full(nw, np.nan)
    ts_[top.index.values] = top.values
    F["top_src"] = ts_ / nz
    known_u = (su < 200) & (base_ttl[np.minimum(su, 199)] >= 0)
    F["new_src"] = _bc(wu[~known_u], nw)
    # temporal-integrity features (known sources only)
    kn = (g["src"] < 200) & (base_ttl[np.minimum(g["src"], 199)] >= 0)
    ckn = _bc(w[kn], nw)
    ckn_s = np.maximum(ckn, 1)
    rtt = g["rtt"]
    v = ~np.isnan(rtt)
    F["rtt_mean"] = np.where(_bc(w[v], nw) > 0, _bc(w[v], nw, rtt[v].astype(float)) / np.maximum(_bc(w[v], nw), 1), np.nan)
    dev = kn & (g["ttl"] != base_ttl[np.minimum(g["src"], 199)])
    F["ttl_dev"] = np.where(ckn > 0, _bc(w[dev], nw) / ckn_s, np.nan)
    F["gap_rate"] = np.where(ckn > 0, _bc(w[kn & ((g["flags"] & GAP) > 0)], nw) / ckn_s, np.nan)
    F["dup_rate"] = np.where(ckn > 0, _bc(w[kn & ((g["flags"] & DUP) > 0)], nw) / ckn_s, np.nan)
    sk = g["skew"]
    vs = kn & ~np.isnan(sk)
    adj = np.abs(sk[vs] - base_skew[np.minimum(g["src"][vs], 199)])
    F["skew_med"] = _qwin(w[vs], adj, 0.5, nw) if vs.any() else np.full(nw, np.nan)
    F["skew_p95"] = _qwin(w[vs], adj, 0.95, nw) if vs.any() else np.full(nw, np.nan)
    if tr == 0:
        tp = g["topic"] >= 0
        wu2, tu2, c2 = _uniq(w[tp], g["topic"][tp], nw)
        cnt_tp = np.bincount(w[tp], minlength=nw).astype(float)
        F["topic_entropy"] = np.where(cnt_tp > 0, _entropy(wu2, c2, np.maximum(cnt_tp, 1), nw), np.nan)
        F["topics"] = _bc(wu2, nw)
        F["connect_rate"] = _bc(w[g["kind"] == 1], nw) / (WIN / 60)
        F["subscribe_rate"] = _bc(w[g["kind"] == 2], nw) / (WIN / 60)
        F["wildcard"] = _bc(w[(g["flags"] & WILD) > 0], nw)
        pub = g["kind"] == 0
        F["qos0_share"] = np.where(_bc(w[pub], nw) > 0, _bc(w[pub & (g["qos"] == 0)], nw) / np.maximum(_bc(w[pub], nw), 1), np.nan)
    else:
        F["con_share"] = np.where(cnt > 0, _bc(w[(g["flags"] & CON) > 0], nw) / nz, np.nan)
        F["notfound"] = np.where(cnt > 0, _bc(w[(g["flags"] & NF) > 0], nw) / nz, np.nan)
        wu2, _, _ = _uniq(w, g["topic"], nw)
        F["paths"] = _bc(wu2, nw)
        wu3, _, _ = _uniq(w, g["sport"], nw)
        F["sports"] = _bc(wu3, nw)
        F["get_share"] = np.where(cnt > 0, _bc(w[g["kind"] == 3], nw) / nz, np.nan)
    return {pre + k: v for k, v in F.items()}


# ----------------------------------------------------------------------------
# energy-physics consistency model
# ----------------------------------------------------------------------------
class EnergyModel:
    """Calibrated on the clean commissioning interval (robust estimators)."""

    def fit(self, O, ghi_cs, cfg):
        self.cfg = cfg
        self.rat = np.array(list(cfg.pv_rated) + list(cfg.wind_rated) + [cfg.dg_rated, cfg.bat_power] +
                            list(cfg.load_peak) + [1000.0, 12.0])
        X = O[:, :7].sum(1)
        L = O[:, 7] + O[:, 8]
        self.a = float((X * L).sum() / (L * L).sum())
        r = X - self.a * L
        self.sig_bal = max(1.4826 * np.median(np.abs(r - np.median(r))), 0.05)
        G = O[:, 9]
        day = G > 50
        self.c = np.array([(O[day, i] * G[day]).sum() / (G[day] ** 2).sum() for i in PV])
        self.sig_pv = []
        for k, i in enumerate(PV):
            pred = self.c[k] * G
            res = O[:, i] - pred
            A = np.stack([np.ones(len(pred)), np.abs(pred)], 1)
            coef, *_ = np.linalg.lstsq(A, np.abs(res) * 1.2533, rcond=None)
            self.sig_pv.append((max(coef[0], 0.05), max(coef[1], 0.0)))
        v = O[:, 10]
        self.vb = np.arange(0, 21, 0.5)
        wn = (O[:, 3] / self.rat[3] + O[:, 4] / self.rat[4]) / 2
        idx = np.clip(np.digitize(v, self.vb) - 1, 0, len(self.vb) - 1)
        f = np.array([wn[idx == b].mean() if (idx == b).sum() > 20 else np.nan for b in range(len(self.vb))])
        sd = np.array([1.4826 * np.median(np.abs(wn[idx == b] - np.median(wn[idx == b]))) if (idx == b).sum() > 20 else np.nan
                       for b in range(len(self.vb))])
        ok = ~np.isnan(f)
        self.f = np.maximum.accumulate(np.interp(self.vb, self.vb[ok], f[ok]))
        self.sd_w = np.maximum(np.interp(self.vb, self.vb[ok], sd[ok]), 0.02)
        pvimp = np.mean([O[:, i] / self.c[k] for k, i in enumerate(PV)], axis=0)
        rG = G - pvimp
        self.sig_G = max(1.4826 * np.median(np.abs(rG[day] - np.median(rG[day]))), 8.0)
        self.ghi_cs = ghi_cs
        # innovation scale (EWMA alpha = 0.3) and peer scale
        self.alpha = 0.3
        E = self._ewma(O)
        innov = O - np.vstack([O[:1], E[:-1]])
        # level-dependent (heteroscedastic) robust scales: sigma = s0 + s1*|level|, fitted by least squares on |.|*1.2533
        self.sin0 = np.zeros(N_SENS)
        self.sin1 = np.zeros(N_SENS)
        for i in range(N_SENS):
            A = np.stack([np.ones(len(O)), np.abs(E[:, i])], 1)
            cf, *_ = np.linalg.lstsq(A, np.abs(innov[:, i]) * 1.2533, rcond=None)
            self.sin0[i], self.sin1[i] = max(cf[0], 1e-3 * self.rat[i]), max(cf[1], 0.0)
        pr = np.stack([O[:, i] / self.rat[i] for i in PV], 1)
        med = np.median(pr, 1)
        rr = np.abs(pr - med[:, None]).ravel() * 1.2533
        A = np.stack([np.ones(rr.size), np.repeat(med, 3)], 1)
        cf, *_ = np.linalg.lstsq(A, rr, rcond=None)
        self.sp0, self.sp1 = max(cf[0], 0.003), max(cf[1], 0.0)
        rs = self._rolling_std(O, 12)
        self.rs0 = np.maximum(np.median(rs, 0), 1e-3 * self.rat)
        return self

    def _ewma(self, O):
        from scipy.signal import lfilter
        a = self.alpha
        y, _ = lfilter([a], [1, -(1 - a)], O, axis=0, zi=((1 - a) * O[:1]))
        return y

    @staticmethod
    def _rolling_std(O, w):
        c1 = np.cumsum(np.vstack([np.zeros((1, O.shape[1])), O]), 0)
        c2 = np.cumsum(np.vstack([np.zeros((1, O.shape[1])), O ** 2]), 0)
        n = O.shape[0]
        idx = np.arange(n)
        lo = np.maximum(idx - w + 1, 0)
        cnt = (idx - lo + 1)[:, None]
        s1 = c1[idx + 1] - c1[lo]
        s2 = c2[idx + 1] - c2[lo]
        return np.sqrt(np.maximum(s2 / cnt - (s1 / cnt) ** 2, 0))

    def residuals(self, O):
        n = O.shape[0]
        rho = np.zeros_like(O)
        X = O[:, :7].sum(1)
        L = O[:, 7] + O[:, 8]
        rbal = (X - self.a * L) / self.sig_bal
        G = O[:, 9]
        for k, i in enumerate(PV):
            pred = self.c[k] * G
            s = self.sig_pv[k][0] + self.sig_pv[k][1] * np.abs(pred)
            rho[:, i] = (O[:, i] - pred) / s
        v = O[:, 10]
        fv = np.interp(v, self.vb, self.f)
        sdv = np.interp(v, self.vb, self.sd_w)
        for j, i in enumerate(WI):
            rho[:, i] = (O[:, i] / self.rat[i] - fv) / sdv
        for i in (5, 6, 7, 8):
            rho[:, i] = rbal
        pvimp = np.mean([O[:, i] / self.c[k] for k, i in enumerate(PV)], axis=0)
        bound = np.maximum(G - 1.15 * self.ghi_cs, 0) / self.sig_G
        rho[:, 9] = np.maximum(np.abs((G - pvimp) / self.sig_G), bound)
        wmean = (O[:, 3] / self.rat[3] + O[:, 4] / self.rat[4]) / 2
        rho[:, 10] = (fv - wmean) / sdv
        E = self._ewma(O)
        Em = np.vstack([E[:1], E[:-1]])
        innov = (O - np.vstack([O[:1], E[:-1]])) / (self.sin0[None, :] + self.sin1[None, :] * np.abs(Em))
        pr = np.stack([O[:, i] / self.rat[i] for i in PV], 1)
        medp = np.median(pr, 1, keepdims=True)
        peer_s = (pr - medp) / (self.sp0 + self.sp1 * medp)
        peer = np.abs(peer_s)
        flat = self._rolling_std(O, 12) / self.rs0
        a = np.abs(rho)
        cus = np.zeros_like(a)
        s = np.zeros(O.shape[1])
        for t_ in range(n):
            s = np.maximum(0.0, s + a[t_] - 3.0)
            cus[t_] = s
        return dict(rbal=rbal, rho=a, rho_s=rho, innov=np.abs(innov), peer=peer, peer_s=peer_s, flat=flat, cusum=np.minimum(cus, 100.0))

    def window_features(self, O, tod_hours=None):
        R = self.residuals(O)
        n = O.shape[0]
        nw = n // STEPS_PER_WIN
        sl = slice(0, nw * STEPS_PER_WIN)

        def win(x, fn):
            return fn(x[sl].reshape(nw, STEPS_PER_WIN), axis=1)

        F = {}
        F["e_bal_mean"] = win(np.abs(R["rbal"]), np.mean)
        F["e_bal_max"] = win(np.abs(R["rbal"]), np.max)
        for nm, S in (("m", MQ_S), ("c", CO_S)):
            F[f"e_phys_{nm}"] = win(R["rho"][:, S].max(1), np.mean)
            F[f"e_physx_{nm}"] = win(R["rho"][:, S].max(1), np.max)
            F[f"e_innov_{nm}"] = win(R["innov"][:, S].max(1), np.mean)
            F[f"e_cusum_{nm}"] = win(R["cusum"][:, S].max(1), np.max)
            F[f"e_flat_{nm}"] = win(R["flat"][:, S].min(1), np.mean)
        F["e_peer"] = win(R["peer"].max(1), np.mean)
        # --- multi-scale signed-residual aggregation (bias detectors)
        rb = win(R["rbal"], np.mean)                                   # window mean of signed balance residual
        F["e_bal_bias"] = np.abs(rb)
        F["e_bal_bias10"] = np.abs(pd.Series(rb).rolling(10, min_periods=1).mean().values)
        wp = np.stack([win(R["peer_s"][:, k], np.mean) for k in range(3)], 1)
        F["e_peer_bias"] = np.abs(wp).max(1)
        F["e_peer_bias10"] = np.abs(pd.DataFrame(wp).rolling(10, min_periods=1).mean().values).max(1)
        for nm, S in (("m", MQ_S), ("c", CO_S)):
            wm = np.stack([win(R["rho_s"][:, i], np.mean) for i in S], 1)   # (nw, |S|)
            F[f"e_bias_{nm}"] = np.abs(wm).max(1)
            wm10 = pd.DataFrame(wm).rolling(10, min_periods=1).mean().values
            F[f"e_bias10_{nm}"] = np.abs(wm10).max(1)
        return F


NET_COLS = None
