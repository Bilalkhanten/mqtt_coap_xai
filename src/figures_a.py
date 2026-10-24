import os
"""Figures that depend only on main / stats / protocols / ablation / imbalance results."""
import sys, json
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from common import *
import figstyle as fs
fs.apply()
FIG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figs")
res, st = jload("main"), jload("stats")
CL = fs.CLASSES
MN = ["lr", "rf", "xgb", "lgbm", "mlp", "cnnlstm", "xgblag"]
LAB = {k: MODEL_NAMES[k] for k in MN}


def save(fig, name):
    fig.savefig(f"{FIG}/{name}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("saved", name, flush=True)


# ---------- main comparison: macro-F1 with block-bootstrap CI + per-seed dots; right panel MCC
def fig_models():
    fig, ax = plt.subplots(1, 2, figsize=(6.6, 3.25), gridspec_kw=dict(width_ratios=[1.35, 1]))
    order = sorted(MN, key=lambda k: res[k]["agg"]["macro_f1"]["mean"])
    for i, k in enumerate(order):
        m = res[k]["agg"]["macro_f1"]["mean"]
        lo, hi = res[k]["ci_macro_f1"]
        ax[0].plot([lo, hi], [i, i], color=fs.MODEL_COL[k], lw=1.6, solid_capstyle="round")
        ax[0].plot(m, i, fs.MARK[k], color=fs.MODEL_COL[k], ms=6, mec=fs.SURF, mew=1)
        ax[0].text(hi + 0.006, i, f"{m:.3f}", va="center", fontsize=8.3, color=fs.INK2)
    ax[0].set_yticks(range(len(order)), [LAB[k] for k in order])
    ax[0].set_xlim(0.70, 0.99); ax[0].grid(axis="y", visible=False); ax[0].grid(axis="x")
    ax[0].set_xlabel("Macro-F1 (95 % block-bootstrap CI)")
    ax[0].set_title("Macro-F1")
    fs.panel(ax[0], "a")
    for i, k in enumerate(order):
        a = res[k]["agg"]
        for j, key in enumerate(["mcc", "bal_acc"]):
            pass
        ax[1].barh(i, a["mcc"]["mean"], color=fs.MODEL_COL[k], height=0.55)
        ax[1].text(a["mcc"]["mean"] + 0.008, i, f"{a['mcc']['mean']:.3f}", va="center", fontsize=8.3, color=fs.INK2)
    ax[1].set_yticks(range(len(order)), [""] * len(order)); ax[1].grid(axis="y", visible=False); ax[1].grid(axis="x")
    ax[1].set_xlim(0.5, 1.0); ax[1].set_xlabel("MCC")
    ax[1].set_title("MCC"); fs.panel(ax[1], "b")
    fig.tight_layout(); save(fig, "fig_models")


# ---------- confusion matrices: best model + logistic baseline (row-normalised)
def _cm_panel(a, cm, title):
    from matplotlib.colors import PowerNorm
    n = cm.shape[0]
    rn = cm / cm.sum(1, keepdims=True)
    a.pcolormesh(np.arange(n + 1), np.arange(n + 1), rn, cmap=fs.SEQ, norm=PowerNorm(0.5, vmin=0, vmax=1), edgecolors="white", linewidth=1.8)
    a.set_xlim(0, n); a.set_ylim(n, 0); a.set_aspect("equal"); a.grid(False)
    for i in range(n):
        for j in range(n):
            col = "white" if rn[i, j] > 0.45 else fs.INK
            a.text(j + 0.5, i + 0.40, f"{int(cm[i, j]):,}".replace(",", " "), ha="center", va="center", fontsize=7.4,
                   fontweight="bold" if i == j else "normal", color=col)
            a.text(j + 0.5, i + 0.73, f"{rn[i, j]*100:.1f}%", ha="center", va="center", fontsize=6.4, color=col, alpha=0.92)
    for i in range(n):
        a.add_patch(plt.Rectangle((i, i), 1, 1, fill=False, ec=fs.INK, lw=1.4))
    a.set_xticks(np.arange(n) + 0.5, CL, rotation=30, ha="right", rotation_mode="anchor"); a.set_yticks(np.arange(n) + 0.5, CL)
    a.tick_params(length=0)
    for s_ in a.spines.values(): s_.set_visible(True); s_.set_color(fs.AXIS)
    a.set_xlabel("Predicted class", labelpad=6); a.set_ylabel("True class"); a.set_title(title, pad=24)
    rec = np.diag(cm) / cm.sum(1); prec = np.diag(cm) / np.maximum(cm.sum(0), 1); f1 = 2 * rec * prec / (rec + prec)
    sup = cm.sum(1)
    xs = [n + 0.3, n + 1.65, n + 3.0, n + 4.55]
    for x, h in zip(xs, ["Support", "Recall", "Precision", "F1"]):
        a.text(x, -0.18, h, ha="left", va="bottom", fontsize=7.4, color=fs.INK, fontweight="bold", clip_on=False)
    for i in range(n):
        a.text(xs[0], i + 0.5, f"{int(sup[i]):,}".replace(",", " "), ha="left", va="center", fontsize=7.4, color=fs.INK2, clip_on=False)
        for x, v in zip(xs[1:], [rec[i], prec[i], f1[i]]):
            a.text(x, i + 0.5, f"{v:.3f}", ha="left", va="center", fontsize=7.4, color=fs.INK2, clip_on=False)


def fig_confusion():
    fig, a = plt.subplots(figsize=(6.6, 4.5))
    k = st["best_model"]
    _cm_panel(a, np.array(st["cm"][k], float), f"{LAB[k]}: test days 23–29, n = 20 160 windows")
    a.title.set_x(-0.05); a.title.set_ha("left")
    fig.subplots_adjust(left=0.13, right=0.70, bottom=0.2, top=0.88)
    save(fig, "fig_confusion")


def fig_confusion_lr():
    fig, a = plt.subplots(figsize=(6.6, 4.5))
    _cm_panel(a, np.array(st["cm"]["lr"], float), f"{LAB['lr']}: test days 23–29, n = 20 160 windows")
    a.title.set_x(-0.05); a.title.set_ha("left")
    fig.subplots_adjust(left=0.13, right=0.70, bottom=0.2, top=0.88)
    save(fig, "fig_confusion_lr")


# ---------- per-class F1 by model (dot plot)
def fig_perclass():
    fig, ax = plt.subplots(figsize=(6.6, 3.25))
    classes = CL[1:]
    for k in MN:
        v = np.array(res[k]["agg"]["per_f1"]["mean"])[1:]
        ax.plot(range(len(classes)), v, fs.MARK[k] + "-", color=fs.MODEL_COL[k], lw=0.0, ms=5.5, mec=fs.SURF, mew=0.8, label=LAB[k])
    ax.set_xticks(range(len(classes)), classes); ax.set_ylim(0.0, 1.03); ax.set_ylabel("F1 (mean of 5 seeds)")
    ax.grid(axis="x", visible=False)
    ax.legend(ncol=4, loc="lower center", bbox_to_anchor=(0.5, 1.0), fontsize=7.8)
    save(fig, "fig_perclass")


# ---------- critical-difference diagram
def fig_cd():
    f = st["friedman"]; r = f["avg_rank"]; cd = f["cd"]
    names = sorted(r, key=r.get)
    fig, ax = plt.subplots(figsize=(6.6, 2.13))
    ax.set_xlim(0.7, 7.3); ax.set_ylim(-1.6, 1.25); ax.invert_xaxis(); ax.axis("off")
    ax.plot([1, 7], [0.55, 0.55], color=fs.INK2, lw=0.9)
    for t in range(1, 8):
        ax.plot([t, t], [0.55, 0.48], color=fs.INK2, lw=0.9)
        ax.text(t, 0.68, str(t), ha="center", fontsize=8.3, color=fs.INK2)
    # CD bar
    ax.plot([7, 7 - cd], [1.02, 1.02], color=fs.INK, lw=1.8); ax.text(7 - cd / 2, 1.1, f"CD = {cd:.2f}", ha="center", fontsize=8.3)
    for i, n in enumerate(names):
        x = r[n]; left = i < 4
        y = 0.35 - 0.34 * (i if left else i - 4) if left else -0.35 - 0.0
        yy = 0.2 - 0.3 * i if left else None
    half = [names[:4], names[4:]]
    ys = {}
    for i, n in enumerate(half[0]): ys[n] = (0.25 - 0.33 * i, "right")
    for i, n in enumerate(half[1]): ys[n] = (0.25 - 0.33 * (len(half[1]) - 1 - i), "left")
    for n, (y, side) in ys.items():
        x = r[n]
        ax.plot([x, x], [0.55, y], color=fs.MODEL_COL[n], lw=1.0)
        xe = 7.3 if side == "right" else 0.7
        xe = 6.3 if False else None
    # place labels on left (low ranks → right side because axis inverted)
    for n, (y, side) in ys.items():
        x = r[n]
        if side == "right":
            ax.plot([x, 0.85], [y, y], color=fs.MODEL_COL[n], lw=1.0); ax.text(0.8, y, f"{LAB[n]}  ({x:.2f})", ha="left", va="center", fontsize=8.3)
        else:
            ax.plot([x, 7.15], [y, y], color=fs.MODEL_COL[n], lw=1.0); ax.text(7.2, y, f"({x:.2f})  {LAB[n]}", ha="right", va="center", fontsize=8.3)
        ax.plot(x, 0.55, fs.MARK[n], color=fs.MODEL_COL[n], ms=5, mec=fs.SURF, mew=0.8)
    # groups not separable by Nemenyi
    y0 = -1.15
    srt = sorted(r.values()); k = 0
    for i, n in enumerate(names):
        j = i
        while j + 1 < len(names) and r[names[j + 1]] - r[n] < cd: j += 1
        if j > i and not (i > 0 and r[names[min(j, len(names) - 1)]] - r[names[i - 1]] < cd and False):
            ax.plot([r[n], r[names[j]]], [y0 - 0.1 * k, y0 - 0.1 * k], color=fs.INK, lw=2.2, solid_capstyle="round"); k += 1
    save(fig, "fig_cd")


# ---------- FDIA anatomy: variants, delay, FPR by benign type
def fig_anatomy():
    fig, ax = plt.subplots(1, 3, figsize=(6.6, 3.25), gridspec_kw=dict(width_ratios=[1, 1.25, 1.1]))
    v = st["fdia_by_variant"]; names = ["V1-bias", "V2-ramp", "V3-balanced"]
    cols = [fs.CAT[0], fs.CAT[1], fs.CAT[3]]
    rec = [v[n]["recall"] for n in names]
    ax[0].bar(range(3), rec, color=cols, width=0.55)
    for i, n in enumerate(names): ax[0].text(i, rec[i] + 0.02, f"{rec[i]*100:.0f}%\n(n={v[n]['n']})", ha="center", fontsize=7.8, color=fs.INK2)
    ax[0].set_xticks(range(3), ["V1\nbias", "V2\nramp", "V3\nbalanced"]); ax[0].set_ylim(0, 1.18); ax[0].set_yticks([0,0.2,0.4,0.6,0.8,1.0]); ax[0].set_ylabel("Recall")
    ax[0].set_title("FDIA recall by variant"); fs.panel(ax[0], "a")
    rs = st["recall_by_scope"]
    xs, labs, vals, cc = [], [], [], []
    i = 0
    for c in CL[1:]:
        for s in rs[c]:
            if c == "FDIA": continue
            xs.append(i); vals.append(rs[c][s]["recall"]); cc.append(fs.CLS_COL[c]); labs.append(f"{c[:5]}.-{s.upper()[:2]}" if False else f"{c}\n{s}"); i += 1
    ax[1].bar(xs, vals, color=cc, width=0.65)
    ax[1].set_xticks(xs, [l.replace("\n", " ").replace("Amplification", "Ampl.") for l in labs], fontsize=7.1, rotation=60, ha="right")
    for x_, v_ in zip(xs, vals): ax[1].text(x_, v_ + 0.015, f"{v_*100:.0f}", ha="center", fontsize=7.3, color=fs.INK2)
    ax[1].set_ylim(0.6, 1.06); ax[1].set_ylabel("Recall"); ax[1].set_title("Recall by transport"); fs.panel(ax[1], "b")
    fp = st["fpr_by_benign"]
    nm = ["none", "route", "outage", "fault"]
    nlab = [f"No event\nn={fp['none']['n']:,}", f"Route\nn={fp['route']['n']:,}", f"Outage\nn={fp['outage']['n']:,}", f"Fault\nn={fp['fault']['n']:,}"]
    vals = [fp[n]["fpr"] * 100 for n in nm]
    ax[2].bar(range(4), vals, color=[fs.GRAY, fs.GRAY, fs.GRAY, fs.CAT[7]], width=0.6)
    for i, n in enumerate(nm): ax[2].text(i, vals[i] + 0.4, f"{vals[i]:.1f}%", ha="center", fontsize=7.8, color=fs.INK2)
    ax[2].set_xticks(range(4), nlab, fontsize=7.1); ax[2].set_ylim(0, 22); ax[2].set_ylabel("False-alarm rate (%)")
    ax[2].set_title("Normal windows flagged"); fs.panel(ax[2], "c")
    fig.tight_layout(); save(fig, "fig_anatomy")


# ---------- evaluation protocol / leakage
def fig_protocols():
    p = jload("protocols")
    fig, ax = plt.subplots(1, 2, figsize=(6.6, 3.02), gridspec_kw=dict(width_ratios=[1.2, 1]))
    prots = ["random", "grouped", "chrono"]; plab = ["Random\nwindows", "Burst-\ngrouped\n5-fold", "Chrono-\nlogical\n+ purge"]
    w = 0.34
    for j, (m, c) in enumerate([("rf", fs.MODEL_COL["rf"]), ("xgb", fs.MODEL_COL["xgb"])]):
        for i, pr in enumerate(prots):
            v = np.array(p[m][pr]["macro_f1"])
            ax[0].bar(i + (j - 0.5) * w, v.mean(), w * 0.92, color=c, label=MODEL_NAMES[m] if i == 0 else None)
            ax[0].plot([i + (j - 0.5) * w] * len(v), v, "o", color=fs.INK, ms=2.2, mec=fs.SURF, mew=0.4)
            ax[0].text(i + (j - 0.5) * w, v.mean() + 0.006, f"{v.mean():.3f}", ha="center", fontsize=7.6, color=fs.INK2)
    ax[0].set_xticks(range(3), plab); ax[0].set_ylim(0.84, 0.95); ax[0].set_ylabel("Macro-F1"); ax[0].legend(loc="upper right", fontsize=7.8)
    ax[0].set_title("Same models, three protocols"); fs.panel(ax[0], "a")
    for i, pr in enumerate(prots):
        v = p["xgb"][pr]["nn_dist_median"]
        ax[1].barh(i, max(v, 0.7), color=fs.CAT[0] if pr != "random" else fs.CAT[7], height=0.5)
        ax[1].text(max(v, 0.7) * 1.3, i, f"{v:,.0f} windows" if v > 1 else "1 window", va="center", fontsize=8.3)
    ax[1].set_xscale("log"); ax[1].set_xlim(0.6, 2e6); ax[1].set_yticks(range(3), ["Random", "Grouped", "Chrono."]); ax[1].invert_yaxis()
    ax[1].grid(axis="y", visible=False); ax[1].grid(axis="x")
    ax[1].set_xlabel("Median distance to nearest\ntraining window (windows)"); ax[1].set_title("Temporal proximity"); fs.panel(ax[1], "b")
    fig.tight_layout(); save(fig, "fig_protocols")


# ---------- ablation heatmap: change in per-class F1 when a group is removed
def fig_ablation():
    a = jload("ablation")
    full = np.array(a["full"]["per_f1"])
    gs = list(a["drop"])
    M = np.array([np.array(a["drop"][g]["per_f1"]) - full for g in gs])[:, :]
    fig, ax = plt.subplots(1, 2, figsize=(6.6, 3.47), gridspec_kw=dict(width_ratios=[1.45, 1]))
    lim = np.abs(M).max()
    im = ax[0].imshow(M, cmap=fs.DIV, vmin=-lim, vmax=lim, aspect="auto")
    for i in range(len(gs)):
        for j in range(K):
            ax[0].text(j, i, (f"{M[i, j]*100:+.1f}" if abs(M[i, j]*100) >= 0.05 else "0.0"), ha="center", va="center", fontsize=7.1, color="white" if abs(M[i, j]) > lim * 0.55 else fs.INK)
    ax[0].set_xticks(range(K), CL, rotation=40, ha="right"); ax[0].set_yticks(range(len(gs)), gs); ax[0].grid(False)
    for s in ax[0].spines.values(): s.set_visible(False)
    ax[0].set_title("Δ F1 (points), group removed"); fs.panel(ax[0], "a")
    fm = np.mean(a["full"]["macro_f1"])
    drop = np.array([fm - np.mean(a["drop"][g]["macro_f1"]) for g in gs]) * 100
    only = np.array([np.mean(a["only"][g]["macro_f1"]) for g in gs])
    yy = np.arange(len(gs))
    ax[1].barh(yy, only, color=[fs.CAT[i] for i in range(len(gs))], height=0.6)
    for i in range(len(gs)): ax[1].text(only[i] + 0.012, i, f"{only[i]:.2f}", va="center", fontsize=7.8, color=fs.INK2)
    ax[1].set_yticks(yy, [""] * len(gs)); ax[1].invert_yaxis(); ax[1].grid(axis="y", visible=False); ax[1].grid(axis="x")
    ax[1].set_xlim(0, 0.85); ax[1].set_xlabel("Macro-F1 using this group alone"); ax[1].set_title("Stand-alone power"); fs.panel(ax[1], "b")
    ax[0].invert_yaxis() if False else None
    fig.tight_layout(); save(fig, "fig_ablation")


# ---------- imbalance handling
def fig_imbalance():
    d = jload("imbalance")
    fig, ax = plt.subplots(1, 2, figsize=(6.6, 3.02), gridspec_kw=dict(width_ratios=[1, 1.2]))
    strat = list(d["xgb"].keys())
    lab = {"none": "none", "class-weight": "class\nweight", "SMOTE": "SMOTE", "undersample": "under-\nsample", "SMOTE+class-weight": "SMOTE+\nweight"}
    for j, m in enumerate(d):
        ss = [s for s in strat if s in d[m]]
        xs = np.array([strat.index(s) for s in ss]) + (j - 0.5) * 0.2
        v = [np.mean(d[m][s]["macro_f1"]) for s in ss]
        sd = [np.std(d[m][s]["macro_f1"], ddof=1) for s in ss]
        ax[0].errorbar(xs, v, sd, fmt=("s" if m == "rf" else "^"), color=fs.MODEL_COL[m], ms=5.5, lw=1, capsize=2, label=MODEL_NAMES[m])
    ax[0].set_xticks(range(len(strat)), [lab[s] for s in strat], fontsize=7.6)
    ax[0].set_ylabel("Macro-F1 (5 seeds)"); ax[0].legend(fontsize=7.8, loc="lower left"); ax[0].set_title("Overall"); fs.panel(ax[0], "a")
    M = np.array([np.array(d["xgb"][s]["per_f1"]) - np.array(d["xgb"]["none"]["per_f1"]) for s in strat])[1:, 1:]
    im = ax[1].imshow(M, cmap=fs.DIV, vmin=-np.abs(M).max(), vmax=np.abs(M).max(), aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax[1].text(j, i, (f"{M[i, j]*100:+.1f}" if abs(M[i, j]*100) >= 0.05 else "0.0"), ha="center", va="center", fontsize=7.3, color="white" if abs(M[i, j]) > np.abs(M).max() * 0.6 else fs.INK)
    ax[1].set_xticks(range(6), CL[1:], rotation=40, ha="right"); ax[1].set_yticks(range(M.shape[0]), [lab[s].replace("\n", " ").replace("- ", "") for s in strat[1:]], fontsize=7.6); ax[1].grid(False)
    for s_ in ax[1].spines.values(): s_.set_visible(False)
    ax[1].set_title("XGBoost: Δ F1 (points) vs. no handling"); fs.panel(ax[1], "b")
    fig.tight_layout(); save(fig, "fig_imbalance")


# ---------- PR curves (best model) and reliability diagram
def fig_calibration():
    from sklearn.metrics import precision_recall_curve, average_precision_score
    P = np.load(f"{RES}/main_probs.npz")
    te = np.load(f"{RES}/test_idx.npy")
    D = Data(DATA + "/siteA.pkl")
    y = D.y[te]
    best = st["best_model"]
    fig, ax = plt.subplots(1, 2, figsize=(6.6, 3.36))
    p = P[best][0]
    for c in range(1, K):
        pr, rc, _ = precision_recall_curve(y == c, p[:, c])
        ap = average_precision_score(y == c, p[:, c])
        ax[0].plot(rc, pr, color=fs.CLS_COL[CL[c]], lw=1.4, label=f"{CL[c]} (AP {ap:.2f})")
    ax[0].set_xlabel("Recall"); ax[0].set_ylabel("Precision"); ax[0].set_xlim(0, 1.0); ax[0].set_ylim(0, 1.03)
    ax[0].legend(fontsize=7.4, loc="lower left"); ax[0].set_title("Precision–recall (one-vs-rest)"); ax[0].grid(axis="x"); fs.panel(ax[0], "a")
    bins = np.linspace(0.0, 1.0, 11)
    ax[1].plot([0, 1], [0, 1], color=fs.AXIS, lw=1, ls="--")
    for k in ["lr", "xgblag", "cnnlstm"]:
        pk = P[k][0]
        conf, pred = pk.max(1), pk.argmax(1)
        acc = (pred == y)
        xs, ys = [], []
        for lo, hi in zip(bins[:-1], bins[1:]):
            m = (conf > lo) & (conf <= hi)
            if m.sum() >= 30:
                xs.append(conf[m].mean()); ys.append(acc[m].mean())
        ece = metrics(y, pk)["ece"]
        ax[1].plot(xs, ys, fs.MARK[k] + "-", color=fs.MODEL_COL[k], ms=4.2, lw=1.1, mec=fs.SURF, mew=0.6, label=f"{LAB[k]} (ECE {ece:.3f})")
    ax[1].set_xlabel("Confidence"); ax[1].set_ylabel("Accuracy"); ax[1].set_xlim(0.2, 1.0); ax[1].set_ylim(0.2, 1.02)
    ax[1].legend(fontsize=7.4, loc="lower right"); ax[1].set_title("Reliability (top-label)"); fs.panel(ax[1], "b")
    fig.tight_layout(); save(fig, "fig_calibration")


# ---------- univariate FDIA detectability
def fig_fdia_auc():
    a = jload("fdia_auc")
    vs = ["V1-bias", "V2-ramp", "V3-balanced"]
    f1 = sorted(a["V1-bias"]["all"], key=lambda f: -a["V1-bias"]["all"][f])[:7]
    f3 = [f for f in sorted(a["V3-balanced"]["all"], key=lambda f: -a["V3-balanced"]["all"][f]) if f not in f1][:5]
    feats = f1 + f3
    M = np.array([[a[v]["all"][f] for v in vs] for f in feats])
    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    im = ax.imshow(M, cmap=fs.SEQ, vmin=0.5, vmax=1.0, aspect="auto")
    for i in range(len(feats)):
        for j in range(3):
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=7.6, color="white" if M[i, j] > 0.8 else fs.INK)
    ax.set_xticks(range(3), ["V1 bias", "V2 ramp", "V3 balanced"]); ax.set_yticks(range(len(feats)), feats, fontsize=7.6); ax.grid(False)
    for s_ in ax.spines.values(): s_.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.03); cb.set_label("ROC-AUC vs. Normal", fontsize=7.8); cb.outline.set_visible(False)
    fig.tight_layout(); save(fig, "fig_fdia_auc")


if __name__ == "__main__":
    which = sys.argv[1:] or ["models", "confusion", "perclass", "cd", "anatomy", "protocols", "ablation", "imbalance"]
    for w in which:
        globals()["fig_" + w]()


