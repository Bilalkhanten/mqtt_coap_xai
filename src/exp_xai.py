"""XAI experiments: TreeSHAP / KernelSHAP / LIME / Integrated Gradients, faithfulness, stability, agreement."""
import sys, time
import numpy as np
import shap
from scipy.stats import spearmanr, kendalltau
from sklearn.linear_model import Ridge
from common import *

D = Data(DATA + "/siteA.pkl")
tr, va, te = D.chrono()
tuned = jload("tuning")
F = len(FEATURES)
rng = np.random.default_rng(0)
N_PER = int(sys.argv[1]) if len(sys.argv) > 1 else 150
sel = []
for c in range(K):
    ii = te[D.y[te] == c]
    sel += list(rng.choice(ii, min(N_PER, len(ii)), replace=False))
sel = np.array(sorted(sel))
Xs = D.X[sel]
comm = D.day < 3
base = np.median(D.X[comm], 0).astype(np.float32)              # "nominal operating point" background
sd = D.X[comm].std(0) + 1e-6
nonneg = np.array([not f.startswith("tod_") for f in FEATURES])

print("training models", flush=True)
xgb = make("xgb", tuned["xgb"]["best"], 0).fit(D, tr, va)
rf = make("rf", tuned["rf"]["best"], 0).fit(D, tr, va)
mlp = make("mlp", tuned["mlp"]["best"], 0).fit(D, tr, va)
cnn = make("cnnlstm", tuned["cnnlstm"]["best"], 0).fit(D, tr, va)
Pxgb = xgb.predict_proba(D, sel)
pc = Pxgb.argmax(1)
ysel = D.y[sel]
out = dict(n_sel=int(len(sel)), acc_sel=float((pc == ysel).mean()))
arr = dict(sel=sel, pc=pc, ysel=ysel)


def sv_pred(sv, pc):
    sv = np.asarray(sv)
    if sv.ndim == 3 and sv.shape[2] == K:                      # (n, F, K)
        return sv[np.arange(len(pc)), :, pc]
    if sv.ndim == 3 and sv.shape[0] == K:                      # (K, n, F)
        return sv[pc, np.arange(len(pc)), :]
    raise ValueError(sv.shape)


# ----------------------------------------------------------------- TreeSHAP
t0 = time.time()
ex_x = shap.TreeExplainer(xgb.m)
sv_x = np.asarray(ex_x.shap_values(Xs))
out["time_treeshap_xgb_per_inst_ms"] = (time.time() - t0) / len(sel) * 1000
phi_x = sv_pred(sv_x, pc)
arr["sv_xgb_full"] = sv_x.astype(np.float32)
print("xgb treeshap done", sv_x.shape, f"{time.time()-t0:.0f}s", flush=True)
t0 = time.time()
Prf = rf.predict_proba(D, sel)
pc_rf = Prf.argmax(1)
ex_r = shap.TreeExplainer(rf.m)
sv_r = np.asarray(ex_r.shap_values(Xs, check_additivity=False))
out["time_treeshap_rf_per_inst_ms"] = (time.time() - t0) / len(sel) * 1000
phi_r = sv_pred(sv_r, pc_rf)
print("rf treeshap done", f"{time.time()-t0:.0f}s", flush=True)
# additivity check for XGB (log-odds space)
marg = xgb.m.predict(Xs, output_margin=True)
ev = np.atleast_1d(np.asarray(ex_x.expected_value, dtype=float))
recon = (ev[pc] if len(ev) > 1 else ev[0]) + phi_x.sum(1)
out["additivity_max_abs_err_xgb"] = float(np.abs(recon - marg[np.arange(len(pc)), pc]).max())


# ----------------------------------------------------------------- LIME (re-implemented tabular LIME)
def lime_tab(pred_fn, x, label, seed, n=1000, k=8):
    r = np.random.default_rng(seed)
    Z = r.standard_normal((n, F))
    Z[0] = 0
    Xp = x + Z * sd
    Xp[:, nonneg] = np.maximum(Xp[:, nonneg], 0)
    Zr = (Xp - x) / sd
    d2 = (Zr ** 2).sum(1)
    w = np.exp(-d2 / (0.75 * np.sqrt(F)) ** 2 * 1.0)
    y = pred_fn(Xp.astype(np.float32))[:, label]
    m = Ridge(alpha=1.0).fit(Zr, y, sample_weight=w)
    idx = np.argsort(-np.abs(m.coef_))[:k]
    m2 = Ridge(alpha=1.0).fit(Zr[:, idx], y, sample_weight=w)
    c = np.zeros(F)
    c[idx] = m2.coef_
    return c, float(m2.score(Zr[:, idx], y, sample_weight=w))


t0 = time.time()
pf = lambda X: xgb.m.predict_proba(X)
lime_c, lime_r2 = [], []
for i in range(len(sel)):
    c, r2 = lime_tab(pf, Xs[i], pc[i], seed=i)
    lime_c.append(c); lime_r2.append(r2)
lime_c = np.array(lime_c)
out["time_lime_per_inst_ms"] = (time.time() - t0) / len(sel) * 1000
out["lime_local_r2_mean"] = float(np.mean(lime_r2))
print("lime done", f"{time.time()-t0:.0f}s", flush=True)

# ----------------------------------------------------------------- global importances
gain = xgb.m.get_booster().get_score(importance_type="total_gain")
gain_v = np.array([gain.get(f"f{i}", 0.0) for i in range(F)])
# permutation importance on the validation block (macro-F1 drop, 5 repeats)
yv = D.y[va]
base_f1 = f1_score(yv, xgb.predict_proba(D, va).argmax(1), average="macro")
perm = np.zeros(F)
prr = np.random.default_rng(1)
Xv = D.X[va].copy()
for j in range(F):
    dr = []
    for _ in range(5):
        Xp = Xv.copy()
        Xp[:, j] = prr.permutation(Xp[:, j])
        dr.append(base_f1 - f1_score(yv, xgb.m.predict_proba(Xp).argmax(1), average="macro"))
    perm[j] = np.mean(dr)
glob_shap_x = np.abs(phi_x).mean(0)
glob_shap_r = np.abs(phi_r).mean(0)
glob_lime = np.abs(lime_c).mean(0)
out["spearman_global"] = dict(
    shap_xgb_vs_shap_rf=float(spearmanr(glob_shap_x, glob_shap_r)[0]),
    shap_xgb_vs_perm=float(spearmanr(glob_shap_x, perm)[0]),
    shap_xgb_vs_gain=float(spearmanr(glob_shap_x, gain_v)[0]),
    shap_xgb_vs_lime=float(spearmanr(glob_shap_x, glob_lime)[0]),
    perm_vs_gain=float(spearmanr(perm, gain_v)[0]))

# ----------------------------------------------------------------- per-instance agreement (SHAP vs LIME)
tau, jac5 = [], []
for i in range(len(sel)):
    a, b = np.abs(phi_x[i]), np.abs(lime_c[i])
    t_ = kendalltau(a, b)[0]
    tau.append(0.0 if np.isnan(t_) else t_)
    jac5.append(len(set(np.argsort(-a)[:5]) & set(np.argsort(-b)[:5])) / len(set(np.argsort(-a)[:5]) | set(np.argsort(-b)[:5])))
out["agree_shap_lime"] = dict(kendall_mean=float(np.mean(tau)), kendall_sd=float(np.std(tau)), jaccard5_mean=float(np.mean(jac5)),
                              by_class={CLASSES[c]: dict(kendall=float(np.mean(np.array(tau)[pc == c])), jaccard5=float(np.mean(np.array(jac5)[pc == c])))
                                        for c in range(K) if (pc == c).any()})
arr["tau"], arr["jac5"] = np.array(tau), np.array(jac5)


# ----------------------------------------------------------------- KernelSHAP vs TreeSHAP (interventional) on a small subset
t0 = time.time()
sub = np.concatenate([np.where(pc == c)[0][:10] for c in range(K)])
bg = D.X[comm][rng.choice(comm.sum(), 50, replace=False)]
ks = []
marg_fn = lambda X, c: xgb.m.predict(np.asarray(X, dtype=np.float32), output_margin=True)[:, c]
for i in sub:
    kex = shap.KernelExplainer(lambda X, c=pc[i]: marg_fn(X, c), bg)
    ks.append(kex.shap_values(Xs[i:i + 1], nsamples=1500, silent=True)[0])
ks = np.array(ks)
# TreeSHAP interventional (same background), log-odds scale, for the same class
ex_int = shap.TreeExplainer(xgb.m, data=bg, feature_perturbation="interventional")
ti = []
for i in sub:
    v = np.asarray(ex_int.shap_values(Xs[i:i + 1], check_additivity=False))
    ti.append(v[0][:, pc[i]] if v.ndim == 3 and v.shape[2] == K else (v[pc[i], 0, :] if v.ndim == 3 else v[0]))
ti = np.array(ti)
rho_k = [spearmanr(np.abs(ks[j]), np.abs(ti[j]))[0] for j in range(len(sub))]
rho_p = [spearmanr(np.abs(phi_x[i]), np.abs(ti[j]))[0] for j, i in enumerate(sub)]
out["kernel_vs_tree"] = dict(n=int(len(sub)), spearman_kernel_vs_interventional=float(np.nanmean(rho_k)),
                             spearman_pathdep_vs_interventional=float(np.nanmean(rho_p)),
                             l1_rel=float(np.mean(np.abs(ks - ti).sum(1) / np.maximum(np.abs(ti).sum(1), 1e-9))),
                             time_kernel_per_inst_s=float((time.time() - t0) / len(sub)))
print("kernelshap done", f"{time.time()-t0:.0f}s", out["kernel_vs_tree"], flush=True)


# ----------------------------------------------------------------- faithfulness: deletion / insertion / comprehensiveness / sufficiency
def curves(pred_fn, X, pcls, attr, baseline, rank="signed"):
    n = len(X)
    key = -attr if rank == "signed" else -np.abs(attr)
    order = np.argsort(key, axis=1)
    p0 = pred_fn(X)[np.arange(n), pcls]
    pb = pred_fn(np.tile(baseline, (n, 1)))[np.arange(n), pcls]
    dele = np.zeros((n, F + 1))
    ins = np.zeros((n, F + 1))
    dele[:, 0], ins[:, 0] = p0, pb
    Xd, Xi = X.copy(), np.tile(baseline, (n, 1)).astype(X.dtype)
    for k in range(1, F + 1):
        cols = order[:, k - 1]
        Xd[np.arange(n), cols] = baseline[cols]
        Xi[np.arange(n), cols] = X[np.arange(n), cols]
        dele[:, k] = pred_fn(Xd)[np.arange(n), pcls]
        ins[:, k] = pred_fn(Xi)[np.arange(n), pcls]
    return dele, ins, p0, pb


def summarise(dele, ins, p0):
    x = np.linspace(0, 1, F + 1)
    return dict(del_auc=float(np.trapezoid(dele, x, axis=1).mean()), ins_auc=float(np.trapezoid(ins, x, axis=1).mean()),
                comp5=float((p0 - dele[:, 5]).mean()), suff5=float((p0 - ins[:, 5]).mean()),
                comp3=float((p0 - dele[:, 3]).mean()), suff3=float((p0 - ins[:, 3]).mean()),
                del_auc20=float(np.trapezoid(dele[:, :21], np.linspace(0, 1, 21), axis=1).mean()))


# restrict probability evaluation to attack-predicted instances for clarity as well as all
faith = {}
pf_x = lambda X: xgb.m.predict_proba(X)
rr = np.random.default_rng(5)
rand_attr = rr.standard_normal(phi_x.shape)
glob_rank = np.tile(glob_shap_x, (len(sel), 1))
methods = {"TreeSHAP": phi_x, "LIME": lime_c, "Gain (global)": np.tile(gain_v, (len(sel), 1)),
           "Permutation (global)": np.tile(perm, (len(sel), 1)), "Random": rand_attr}
curve_store = {}
for nm, A in methods.items():
    d_, i_, p0, pb = curves(pf_x, Xs, pc, A, base)
    faith[nm] = summarise(d_, i_, p0)
    curve_store[nm] = (d_.mean(0), i_.mean(0))
    print("faith", nm, faith[nm], flush=True)
out["faith_xgb"] = faith
arr["curve_del"] = np.stack([curve_store[k][0] for k in methods])
arr["curve_ins"] = np.stack([curve_store[k][1] for k in methods])
out["faith_methods"] = list(methods)
# RF TreeSHAP faithfulness
pf_r = lambda X: rf.m.predict_proba(X)
d_, i_, p0, pb = curves(pf_r, Xs, pc_rf, phi_r, base)
out["faith_rf"] = {"TreeSHAP": summarise(d_, i_, p0)}
d2_, i2_, p02, _ = curves(pf_r, Xs, pc_rf, np.random.default_rng(6).standard_normal(phi_r.shape), base)
out["faith_rf"]["Random"] = summarise(d2_, i2_, p02)
print("faith rf", out["faith_rf"], flush=True)

# ----------------------------------------------------------------- stability under input noise
n_st = min(len(sel), 420)
st_idx = np.sort(np.random.default_rng(2).choice(len(sel), n_st, replace=False))
eps = 0.05
sh_rel, li_j, sh_j = [], [], []
rn = np.random.default_rng(3)
for i in st_idx:
    x = Xs[i]
    for rep in range(3):
        xp = x + eps * sd * rn.standard_normal(F)
        xp[nonneg] = np.maximum(xp[nonneg], 0)
        xp = xp.astype(np.float32)
        if xgb.m.predict_proba(xp[None])[0].argmax() != pc[i]:
            continue
        ph = sv_pred(np.asarray(ex_x.shap_values(xp[None])), np.array([pc[i]]))[0]
        sh_rel.append(np.linalg.norm(ph - phi_x[i]) / (np.linalg.norm(phi_x[i]) + 1e-9))
        a, b = set(np.argsort(-np.abs(phi_x[i]))[:5]), set(np.argsort(-np.abs(ph))[:5])
        sh_j.append(len(a & b) / len(a | b))
        cl, _ = lime_tab(pf, xp, pc[i], seed=1000 + i * 3 + rep)
        a2, b2 = set(np.argsort(-np.abs(lime_c[i]))[:5]), set(np.argsort(-np.abs(cl))[:5])
        li_j.append(len(a2 & b2) / len(a2 | b2))
out["stability"] = dict(eps=eps, treeshap_rel_l2=float(np.mean(sh_rel)), treeshap_jaccard5=float(np.mean(sh_j)),
                        lime_jaccard5=float(np.mean(li_j)), n=int(len(sh_j)))
print("stability", out["stability"], flush=True)

# ----------------------------------------------------------------- neural networks: Integrated Gradients
import torch
torch.set_num_threads(2)


def ig_attr(net, x, pcls, steps=32):
    x = torch.from_numpy(x)
    bl = torch.zeros_like(x)
    tot = torch.zeros_like(x)
    for a in torch.linspace(1 / steps, 1, steps):
        xi = (bl + a * (x - bl)).requires_grad_(True)
        o = net(xi)
        o[torch.arange(len(x)), torch.from_numpy(pcls)].sum().backward()
        tot += xi.grad
    return ((x - bl) * tot / steps).detach().numpy()


nn_sel = np.concatenate([np.where(ysel == c)[0][:50] for c in range(K)])
nn = {}
for nm, mdl in (("MLP", mlp), ("CNN-LSTM", cnn)):
    t0 = time.time()
    Xn = mdl._x(D, sel[nn_sel])
    P = mdl.predict_proba(D, sel[nn_sel])
    pcn = P.argmax(1)
    ig = ig_attr(mdl.net, Xn, pcn)
    phi = ig.sum(1) if ig.ndim == 3 else ig                  # aggregate over time steps
    tim = (time.time() - t0) / len(nn_sel) * 1000

    def pf_n(Z, mdl=mdl):
        # Z is (n,F) in standardised space; for CNN-LSTM the same value is broadcast over the sequence
        if mdl.kind == "cnnlstm":
            return pf_cache["seq_fn"](Z)
        torch_ = torch
        with torch_.no_grad():
            z = mdl.net(torch_.from_numpy(Z.astype(np.float32))).numpy()
        z = z - z.max(1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(1, keepdims=True)

    # faithfulness in standardised space, baseline = commissioning mean (zeros)
    Zs = D.Z[sel[nn_sel]]
    if mdl.kind == "cnnlstm":
        Zseq = D.seq(sel[nn_sel], 5)

        def curves_seq(attr, rank="signed"):
            n = len(Zseq)
            order = np.argsort(-attr, axis=1)
            dele = np.zeros((n, F + 1)); ins = np.zeros((n, F + 1))

            def pr(Zq):
                with torch.no_grad():
                    z = mdl.net(torch.from_numpy(Zq.astype(np.float32))).numpy()
                z = z - z.max(1, keepdims=True); e = np.exp(z); return e / e.sum(1, keepdims=True)
            dele[:, 0] = pr(Zseq)[np.arange(n), pcn]
            ins[:, 0] = pr(np.zeros_like(Zseq))[np.arange(n), pcn]
            Zd, Zi = Zseq.copy(), np.zeros_like(Zseq)
            for k in range(1, F + 1):
                cols = order[:, k - 1]
                Zd[np.arange(n), :, cols] = 0.0
                Zi[np.arange(n), :, cols] = Zseq[np.arange(n), :, cols]
                dele[:, k] = pr(Zd)[np.arange(n), pcn]
                ins[:, k] = pr(Zi)[np.arange(n), pcn]
            return dele, ins, dele[:, 0]
        d_, i_, p0 = curves_seq(phi)
        dr, ir, p0r = curves_seq(np.random.default_rng(8).standard_normal(phi.shape))
    else:
        d_, i_, p0, _ = curves(pf_n, Zs, pcn, phi, np.zeros(F, np.float32))
        dr, ir, p0r, _ = curves(pf_n, Zs, pcn, np.random.default_rng(8).standard_normal(phi.shape), np.zeros(F, np.float32))
    nn[nm] = dict(IG=summarise(d_, i_, p0), Random=summarise(dr, ir, p0r), ig_time_ms=float(tim), n=int(len(nn_sel)))
    arr["ig_" + nm.replace("-", "").lower()] = phi.astype(np.float32)
    arr["ig_idx_" + nm.replace("-", "").lower()] = nn_sel
    # agreement between IG and TreeSHAP (mean Spearman of |attr| per instance)
    rho = [spearmanr(np.abs(phi[j]), np.abs(phi_x[nn_sel[j]]))[0] for j in range(len(nn_sel))]
    nn[nm]["spearman_vs_treeshap"] = float(np.nanmean(rho))
    print("NN", nm, nn[nm], flush=True)
# TreeSHAP faithfulness on the same NN subset for a like-for-like comparison
d_, i_, p0, _ = curves(pf_x, Xs[nn_sel], pc[nn_sel], phi_x[nn_sel], base)
out["faith_xgb_nnsubset"] = summarise(d_, i_, p0)
out["nn"] = nn

# ----------------------------------------------------------------- group-level SHAP (XGBoost, predicted class)
gi = {g: [FEATURES.index(f) for f in fs] for g, fs in GROUPS.items()}
G = np.stack([np.abs(phi_x[:, cols]).sum(1) for cols in gi.values()], 1)
Gs = G / G.sum(1, keepdims=True)
out["group_share_by_class"] = {CLASSES[c]: {g: float(Gs[pc == c, j].mean()) for j, g in enumerate(gi)} for c in range(K) if (pc == c).any()}
arr["phi_xgb"], arr["phi_rf"], arr["lime"] = phi_x.astype(np.float32), phi_r.astype(np.float32), lime_c.astype(np.float32)
arr["gain"], arr["perm"], arr["Xs"] = gain_v, perm, Xs
arr["expected"] = np.atleast_1d(ex_x.expected_value)
# case studies
cases = {}
v = D.df.variant.values
sc = D.df.scope.values
pick = {"FDIA (V1 bias)": lambda i: ysel[i] == 1 and v[sel[i]] == "V1-bias",
        "MITM (MQTT)": lambda i: ysel[i] == 4 and sc[sel[i]] == "mqtt",
        "Recon (CoAP)": lambda i: ysel[i] == 6 and sc[sel[i]] == "coap",
        "Flood (MQTT)": lambda i: ysel[i] == 2 and sc[sel[i]] == "mqtt"}
for nm, fn in pick.items():
    cand = [i for i in range(len(sel)) if fn(i) and pc[i] == ysel[i]]
    if cand:
        i = max(cand, key=lambda j: Pxgb[j, pc[j]])
        cases[nm] = int(i)
out["cases"] = cases
out["case_conf"] = {k: float(Pxgb[i, pc[i]]) for k, i in cases.items()}
jdump(out, "xai")
np.savez_compressed(f"{RES}/xai_arrays.npz", **{k: np.asarray(v) for k, v in arr.items()})
print("XAI DONE", flush=True)
