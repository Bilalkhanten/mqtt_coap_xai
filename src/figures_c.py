import os
"""Figures for robustness (E7/E8/E10), explanations and latency."""
import sys, json, os
import numpy as np
import matplotlib.pyplot as plt
from common import *
import figstyle as fs
fs.apply()
FIG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figs")
CL = fs.CLASSES
GCOL = {g: fs.CAT[i] for i, g in enumerate(GROUPS)}


def save(fig, name):
    fig.savefig(f"{FIG}/{name}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("saved", name, flush=True)


def short(f):
    return f.replace("_m", "·M").replace("_c", "·C") if f.startswith("e_") and (f.endswith("_m") or f.endswith("_c")) else f


# ------------------------------------------------------------------------------------------- XAI: global
def fig_xai_global():
    X = jload("xai")
    A = np.load(f"{RES}/xai_arrays.npz")
    phi = A["phi_xgb"]
    g = np.abs(phi).mean(0)
    order = np.argsort(-g)[:15][::-1]
    fig, ax = plt.subplots(1, 2, figsize=(6.6, 3.70), gridspec_kw=dict(width_ratios=[1.05, 1]))
    cols = [GCOL[GROUP_OF[FEATURES[i]]] for i in order]
    ax[0].barh(range(len(order)), g[order], color=cols, height=0.66)
    ax[0].set_yticks(range(len(order)), [FEATURES[i] for i in order], fontsize=7.6)
    ax[0].grid(axis="y", visible=False); ax[0].grid(axis="x")
    ax[0].set_xlabel("Mean |SHAP| of the predicted class (log-odds)"); ax[0].set_title("Top 15 features"); fs.panel(ax[0], "a")
    seen = []
    for i in order[::-1]:
        gg = GROUP_OF[FEATURES[i]]
        if gg not in seen:
            seen.append(gg)
    ax[0].legend([plt.Rectangle((0, 0), 1, 1, color=GCOL[s]) for s in seen], seen, fontsize=7.1, loc="lower right", handlelength=1)
    gs = X["group_share_by_class"]
    classes = [c for c in CL if c in gs]
    M = np.array([[gs[c][gname] for gname in GROUPS] for c in classes])
    im = ax[1].imshow(M, cmap=fs.SEQ, vmin=0, vmax=max(0.6, M.max()), aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax[1].text(j, i, f"{100*M[i, j]:.0f}", ha="center", va="center", fontsize=7.3, color="white" if M[i, j] > 0.35 else fs.INK)
    short_g = ["Traffic", "Source", "Temporal", "MQTT", "CoAP", "Residual", "Bias", "Context"]
    ax[1].set_xticks(range(len(GROUPS)), short_g, rotation=40, ha="right"); ax[1].set_yticks(range(len(classes)), classes)
    ax[1].grid(False)
    for s_ in ax[1].spines.values(): s_.set_visible(False)
    ax[1].set_title("Share of |SHAP| by feature group (%)"); fs.panel(ax[1], "b")
    fig.tight_layout(); save(fig, "fig_xai_global")


# ------------------------------------------------------------------------------------------- XAI: faithfulness
def fig_xai_faith():
    X = jload("xai")
    A = np.load(f"{RES}/xai_arrays.npz")
    names = X["faith_methods"]
    lab = {"TreeSHAP": "TreeSHAP", "LIME": "LIME", "Gain (global)": "Gain (global)", "Permutation (global)": "Permutation (global)", "Random": "Random"}
    col = {"TreeSHAP": fs.CAT[0], "LIME": fs.CAT[1], "Gain (global)": fs.CAT[2], "Permutation (global)": fs.CAT[3], "Random": fs.GRAY}
    ls = {"Random": "--"}
    fig, ax = plt.subplots(1, 3, figsize=(6.6, 3.02), gridspec_kw=dict(width_ratios=[1, 1, 1.1]))
    K_ = 25
    xs = np.arange(K_ + 1)
    for j, n in enumerate(names):
        ax[0].plot(xs, A["curve_del"][j][:K_ + 1], color=col[n], ls=ls.get(n, "-"), lw=1.4, label=lab[n])
        ax[1].plot(xs, A["curve_ins"][j][:K_ + 1], color=col[n], ls=ls.get(n, "-"), lw=1.4)
    ax[0].set_xlabel("Features removed"); ax[0].set_ylabel("Predicted-class probability"); ax[0].set_title("Deletion"); fs.panel(ax[0], "a")
    ax[1].set_xlabel("Features inserted"); ax[1].set_title("Insertion"); fs.panel(ax[1], "b")
    for a in ax[:2]:
        a.set_ylim(-0.02, 1.02)
    st = X["stability"]; ag = X["agree_shap_lime"]
    items = [("TreeSHAP\n(noise)", st["treeshap_jaccard5"], fs.CAT[0]), ("LIME\n(noise)", st["lime_jaccard5"], fs.CAT[1]),
             ("SHAP vs.\nLIME", ag["jaccard5_mean"], fs.CAT[6])]
    ax[2].bar(range(3), [v for _, v, _ in items], color=[c for _, _, c in items], width=0.55)
    for i, (_, v, _) in enumerate(items):
        ax[2].text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=7.8, color=fs.INK2)
    ax[2].set_xticks(range(3), [n for n, _, _ in items], fontsize=7.6); ax[2].set_ylim(0, 1.1); ax[2].set_ylabel("Top-5 Jaccard overlap")
    ax[2].set_title("Stability and agreement"); fs.panel(ax[2], "c")
    fig.tight_layout()
    h_, l_ = ax[0].get_legend_handles_labels()
    fig.legend(h_, l_, ncol=5, fontsize=7.8, loc="lower center", bbox_to_anchor=(0.36, -0.07), handlelength=1.6, columnspacing=1.2)
    save(fig, "fig_xai_faith")


# ------------------------------------------------------------------------------------------- XAI: case studies
def fig_xai_cases():
    X = jload("xai")
    A = np.load(f"{RES}/xai_arrays.npz")
    phi, Xs = A["phi_xgb"], A["Xs"]
    cases = X["cases"]
    names = list(cases)
    fig, axs = plt.subplots(1, len(names), figsize=(6.6, 3.02), sharex=False)
    axs = np.atleast_1d(axs)
    for ax, nm in zip(axs, names):
        i = cases[nm]
        top = np.argsort(-np.abs(phi[i]))[:7][::-1]
        v = phi[i][top]
        ax.barh(range(len(top)), v, color=[GCOL[GROUP_OF[FEATURES[j]]] for j in top], height=0.62)
        ax.set_yticks(range(len(top)), [FEATURES[j] for j in top], fontsize=7.0)
        ax.axvline(0, color=fs.AXIS, lw=0.8)
        ax.grid(axis="y", visible=False); ax.grid(axis="x")
        ax.set_title(f"{nm}\n(p = {X['case_conf'][nm]:.2f})", fontsize=8.0)
        ax.set_xlabel("SHAP (log-odds)", fontsize=7.8)
        ax.tick_params(axis="x", labelsize=6.5)
    fs.panel(axs[0], "a")
    h = [plt.Rectangle((0, 0), 1, 1, color=GCOL[g]) for g in GROUPS]
    fig.legend(h, list(GROUPS), ncol=4, fontsize=7.0, loc="lower center", bbox_to_anchor=(0.5, -0.12), handlelength=1)
    fig.tight_layout(); save(fig, "fig_xai_cases")


# ------------------------------------------------------------------------------------------- latency
def fig_latency():
    L = jload("latency")
    names = {"lr": "Logistic regression", "rf": "Random forest", "xgb": "XGBoost", "lgbm": "LightGBM", "mlp": "MLP", "cnnlstm": "CNN-LSTM"}
    fig, ax = plt.subplots(1, 2, figsize=(6.6, 3.02), gridspec_kw=dict(width_ratios=[1.15, 1]))
    for k, n in names.items():
        s = np.sort(np.array(L["predict_ms"][k]["samples"]))
        ax[0].plot(s, np.arange(1, len(s) + 1) / len(s), color=fs.MODEL_COL[k], lw=1.3, label=n)
    s = np.sort(np.array(L["treeshap_xgb_ms"]["samples"]))
    ax[0].plot(s, np.arange(1, len(s) + 1) / len(s), color=fs.INK, lw=1.3, ls="--", label="TreeSHAP (XGBoost)")
    ax[0].set_xscale("log"); ax[0].set_xlabel("Per-window latency (ms, single window)"); ax[0].set_ylabel("Empirical CDF"); ax[0].legend(fontsize=7.4, loc="center", bbox_to_anchor=(0.72, 0.38))
    ax[0].grid(axis="x"); ax[0].set_title("Inference and explanation"); fs.panel(ax[0], "a")
    parts = [("Network features", L["feat_network_ms_per_window"], fs.CAT[0]), ("Energy features", L["feat_energy_ms_per_window"], fs.CAT[2]),
             ("XGBoost p95", L["predict_ms"]["xgb"]["p95"], fs.CAT[1]), ("TreeSHAP p95", L["treeshap_xgb_ms"]["p95"], fs.CAT[6])]
    left = 0
    for n, v, c in parts:
        ax[1].barh(0, v, left=left, color=c, height=0.45, label=f"{n}: {v:.2f} ms")
        left += v
    ax[1].set_yticks([]); ax[1].set_xlabel("Time per 30 s window (ms)"); ax[1].legend(fontsize=7.1, loc="upper left", bbox_to_anchor=(0, 0.92))
    ax[1].grid(axis="y", visible=False); ax[1].grid(axis="x"); ax[1].set_ylim(-0.6, 1.2)
    ax[1].text(left, 0.33, f"{left:.1f} ms", ha="right", fontsize=7.8, color=fs.INK2)
    ax[1].set_title("End-to-end budget (p95)"); fs.panel(ax[1], "b")
    fig.tight_layout(); save(fig, "fig_latency")




# ------------------------------------------------------------------------------------------- robustness
def fig_evasion():
    R = jload("robust")
    sc = list(R["e8"].keys())
    xs = [float(s) for s in sc]
    fig, ax = plt.subplots(1, 2, figsize=(6.6, 3.14))
    for nm, lab in (("xgb", "XGBoost"), ("rf", "Random forest"), ("cnnlstm", "CNN-LSTM"), ("lr", "Logistic regression")):
        ax[0].plot(xs, [R["e8"][s][nm]["macro_f1"] for s in sc], fs.MARK[nm] + "-", color=fs.MODEL_COL[nm], ms=4.5, mec=fs.SURF, mew=0.6, lw=1.3, label=lab)
    ax[0].set_xscale("log"); ax[0].invert_xaxis(); ax[0].set_xticks(xs, [f"{x:g}" for x in xs]); ax[0].minorticks_off()
    ax[0].set_xlabel("Attack-intensity scale (1 = as trained)"); ax[0].set_ylabel("Macro-F1"); ax[0].set_ylim(0.35, 1.0)
    ax[0].legend(fontsize=7.4, loc="lower left"); ax[0].set_title("Macro-F1 under weakened attacks"); ax[0].grid(axis="x"); fs.panel(ax[0], "a")
    for c in range(1, 7):
        ax[1].plot(xs, [R["e8"][s]["xgb"]["per_r"][c] for s in sc], "o-", color=fs.CLS_COL[CL[c]], ms=3.8, mec=fs.SURF, mew=0.5, lw=1.2, label=CL[c])
    ax[1].set_xscale("log"); ax[1].invert_xaxis(); ax[1].set_xticks(xs, [f"{x:g}" for x in xs]); ax[1].minorticks_off()
    ax[1].set_xlabel("Attack-intensity scale"); ax[1].set_ylabel("Recall"); ax[1].set_ylim(-0.02, 1.28); ax[1].set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax[1].legend(fontsize=7.1, ncol=3, loc="upper center"); ax[1].set_title("XGBoost recall by class"); ax[1].grid(axis="x"); fs.panel(ax[1], "b")
    fig.tight_layout(); save(fig, "fig_evasion")


def fig_transfer():
    R = jload("robust")
    fig, ax = plt.subplots(1, 3, figsize=(6.6, 3.02), gridspec_kw=dict(width_ratios=[1, 1.1, 0.9]))
    vs = ["V1-bias", "V2-ramp"]
    w = 0.18
    for j, (nm, lab) in enumerate((("xgb", "XGBoost"), ("rf", "Random forest"))):
        e = R["e7"][nm]
        seen = [np.mean(e["none"]["recall"][v]) for v in vs]
        held = [np.mean(e[v]["recall"][v]) for v in vs]
        ax[0].bar(np.arange(2) + (2 * j - 1.5) * w, seen, w, color=fs.MODEL_COL[nm], label=f"{lab}, seen")
        ax[0].bar(np.arange(2) + (2 * j - 0.5) * w, held, w, color=fs.MODEL_COL[nm], alpha=0.45, label=f"{lab}, held out")
    ax[0].set_xticks(range(2), ["V1 bias", "V2 ramp"]); ax[0].set_ylabel("FDIA recall of the variant"); ax[0].set_ylim(0, 1.45)
    ax[0].set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0]); ax[0].legend(fontsize=6.6, loc="upper center", ncol=1); ax[0].set_title("Unseen FDIA variant"); fs.panel(ax[0], "a")
    z = R["e10"]["zero_shot"]
    mods = ["xgb", "rf", "cnnlstm", "lr"]
    cls = list(range(1, 7))
    for j, nm in enumerate(mods):
        ax[1].plot(np.arange(6) + (j - 1.5) * 0.12, [z[nm]["per_f1"][c] for c in cls], fs.MARK[nm], color=fs.MODEL_COL[nm], ms=4.6, mec=fs.SURF, mew=0.5,
                   label={"xgb": "XGBoost", "rf": "RF", "cnnlstm": "CNN-LSTM", "lr": "LR"}[nm])
    ax[1].set_xticks(range(6), [CL[c][:5] + "." if len(CL[c]) > 6 else CL[c] for c in cls], fontsize=7.6, rotation=35, ha="right", rotation_mode="anchor")
    ax[1].set_ylabel("F1 at site B (zero-shot)"); ax[1].set_ylim(0, 1.03); ax[1].grid(axis="x")
    ax[1].set_title("Cross-site transfer"); fs.panel(ax[1], "b")
    fsd = R["e10"].get("few_shot", {})
    if fsd:
        ks = sorted(fsd, key=int)
        for nm, lab in (("xgb", "XGBoost"), ("rf", "Random forest")):
            ax[2].plot([int(k) for k in ks], [fsd[k][nm]["macro_f1"] for k in ks], fs.MARK[nm] + "-", color=fs.MODEL_COL[nm], ms=4.5, mec=fs.SURF, mew=0.6, label=lab)
        ax[2].set_xticks([int(k) for k in ks])
        ax[2].set_xlabel("Site-B days added to training"); ax[2].set_ylabel("Macro-F1 at site B"); ax[2].legend(fontsize=7.2, loc="lower right")
    ax[2].set_title("Few-shot adaptation"); fs.panel(ax[2], "c")
    fig.tight_layout()
    h_, l_ = ax[1].get_legend_handles_labels()
    fig.legend(h_, l_, ncol=4, fontsize=7.8, loc="lower center", bbox_to_anchor=(0.5, -0.07), handlelength=1.2)
    save(fig, "fig_transfer")


if __name__ == "__main__":
    for n in sys.argv[1:]:
        globals()["fig_" + n]()
