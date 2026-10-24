"""Figures from the simulator itself: pipeline, plant, attack timeline, feature distributions."""
import sys, os, pickle
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from common import *
import figstyle as fs
fs.apply()
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(ROOT_DIR, "figs")
CL = fs.CLASSES


def save(fig, name):
    fig.savefig(f"{FIG}/{name}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("saved", name, flush=True)


def box(ax, x, y, w, h, title, lines=(), fc="#ffffff", ec=fs.AXIS, tc=fs.INK, lw=0.9, fs_t=8.4, fs_l=7.2):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.0,rounding_size=0.06", fc=fc, ec=ec, lw=lw))
    ax.text(x + w / 2, y + h - 0.13, title, ha="center", va="top", fontsize=fs_t, fontweight="bold", color=tc)
    for i, l in enumerate(lines):
        ax.text(x + w / 2, y + h - 0.40 - 0.2 * i, l, ha="center", va="top", fontsize=fs_l, color=fs.INK2)


def arrow(ax, x0, y0, x1, y1, color=fs.INK2):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=7, lw=1.0, color=color, shrinkA=0, shrinkB=0))


def fig_pipeline():
    fig, ax = plt.subplots(figsize=(6.6, 3.70))
    ax.set_xlim(0, 12); ax.set_ylim(0, 5.3); ax.axis("off"); ax.grid(False)
    W, H = 2.2, 1.6
    y1 = 3.55
    tops = [("Plant", ["PV x3, wind x2, load x2", "battery + diesel genset", "exact power balance"], "#eef4fc"),
            ("Field network", ["11 sensors, 5-10 s", "MQTT/TCP + CoAP/UDP", "RTT, loss, retry, skew"], "#eef4fc"),
            ("Event injection", ["6 attack families", "5 benign look-alikes", "FDIA: 3 variants"], "#fdf0ea"),
            ("Window features", ["30 s windows, 68 feats", "8 groups, 2 transports", "calibrated on days 0-2"], "#eef4fc"),
            ("Classifier", ["7 models, tuned on", "validation days only", "chronological split"], "#eef4fc")]
    xs = [0.05 + i * 2.38 for i in range(5)]
    for x, (t, l, c) in zip(xs, tops):
        box(ax, x, y1, W, H, t, l, fc=c, fs_t=8.0, fs_l=6.5)
    for x in xs[:-1]:
        arrow(ax, x + W + 0.01, y1 + H / 2, x + 2.38 - 0.01, y1 + H / 2)
    y2 = 1.65
    box(ax, 0.05, y2, 6.9, 1.5, "Evaluation protocol", ["chronological split with purge gaps; block bootstrap",
        "Friedman-Nemenyi, Holm; leakage, ablation, imbalance",
        "unseen FDIA variants; evasion; cross-site transfer"], fc="#f7f7f4", fs_t=8.2, fs_l=6.8)
    box(ax, 7.1, y2, 4.85, 1.5, "Explanation layer", ["TreeSHAP, KernelSHAP, LIME, Integrated Gradients", "faithfulness, stability, agreement, group SHAP"], fc="#eef7f2", fs_t=8.2, fs_l=6.8)
    arrow(ax, xs[4] + W / 2, y1, xs[4] + W / 2 - 0.0, y2 + 1.5)
    y3 = 0.3
    segs = [("calib.", 0, 3, fs.GRAY, "white"), ("train", 3, 19, fs.CAT[0], "white"), ("val", 20, 23, fs.CAT[3], fs.INK), ("test", 23, 30, fs.CAT[1], "white")]
    x0, sc = 1.2, 10.7 / 30
    ax.text(0.05, y3 + 0.25, "30 days", fontsize=7.6, color=fs.INK2, va="center")
    for nm, a_, b_, c, tc in segs:
        ax.add_patch(plt.Rectangle((x0 + a_ * sc + 0.02, y3), (b_ - a_) * sc - 0.04, 0.5, fc=c, ec="none"))
        ax.text(x0 + (a_ + b_) / 2 * sc, y3 + 0.25, nm, ha="center", va="center", fontsize=7.2, color=tc)
    ax.add_patch(plt.Rectangle((x0 + 19 * sc + 0.02, y3), 1 * sc - 0.04, 0.5, fc="none", ec=fs.INK2, lw=0.6, hatch="////"))
    ax.text(x0 + 19.5 * sc, y3 - 0.1, "purge gap", ha="center", va="top", fontsize=6.6, color=fs.INK2)
    save(fig, "fig_pipeline")


def fig_timeline():
    b = pickle.load(open(os.path.join(ROOT_DIR, "data", "bursts.pkl"), "rb"))["bursts_A"]
    fig, ax = plt.subplots(1, 2, figsize=(6.6, 3.25), gridspec_kw=dict(width_ratios=[2.0, 1]))
    rows = CL[1:]
    for x0, x1, c, a in [(0, 3, fs.GRAY, 0.10), (3, 19, fs.CAT[0], 0.05), (20, 23, fs.CAT[3], 0.08), (23, 30, fs.CAT[1], 0.08)]:
        ax[0].axvspan(x0, x1, color=c, alpha=a, lw=0)
    for x, t in [(1.5, "calib."), (11, "train"), (21.5, "val"), (26.5, "test")]:
        ax[0].text(x, len(rows) - 0.35, t, ha="center", fontsize=7.8, color=fs.INK2)
    for r in b:
        j = rows.index(r["cls"])
        ax[0].plot([r["t0"] / 86400, r["t1"] / 86400 + 0.02], [j, j], color=fs.CLS_COL[r["cls"]], lw=3.4, solid_capstyle="butt", alpha=0.9)
    ax[0].set_yticks(range(len(rows)), rows); ax[0].set_xlim(0, 30); ax[0].set_ylim(-0.6, len(rows) - 0.1); ax[0].invert_yaxis()
    ax[0].set_xlabel("Day of simulation"); ax[0].grid(axis="y", visible=False); ax[0].grid(axis="x"); ax[0].set_title("Attack bursts, site A"); fs.panel(ax[0], "a")
    df = pd.read_pickle(os.path.join(ROOT_DIR, "data", "siteA.pkl"))
    cnt = df.label.value_counts().sort_index()
    ax[1].barh(range(7), cnt.values, color=[fs.CLS_COL[c] for c in CL], height=0.6)
    for i, v in enumerate(cnt.values): ax[1].text(v * 1.15, i, f"{v:,}", va="center", fontsize=7.8, color=fs.INK2)
    ax[1].set_xscale("log"); ax[1].set_xlim(300, 4e6); ax[1].set_yticks(range(7), CL); ax[1].invert_yaxis()
    ax[1].grid(axis="y", visible=False); ax[1].grid(axis="x"); ax[1].set_xlabel("30 s windows (log scale)"); ax[1].set_title("Class balance"); fs.panel(ax[1], "b")
    fig.tight_layout(); save(fig, "fig_timeline")


def sim_cache():
    p = os.path.join(ROOT_DIR, "data", "sim_cache.pkl")
    if os.path.exists(p):
        return pickle.load(open(p, "rb"))
    from simulate import simulate
    from config import SITE_A
    df, meta, (truth, meas, O, aux) = simulate(SITE_A, verbose=False)
    keep = dict(truth=truth.astype(np.float32), meas=meas.astype(np.float32), O=O.astype(np.float32),
                soc=np.asarray(aux["soc"]), ghi_cs=np.asarray(aux["ghi_cs"]), bursts=meta["bursts"])
    pickle.dump(keep, open(p, "wb"))
    return keep


def fig_plant():
    S = sim_cache()
    truth, O, soc, ghi = S["truth"], S["O"], S["soc"], S["ghi_cs"]
    SPD = 17280
    d = 12
    sl = slice(d * SPD, (d + 1) * SPD)
    h = np.arange(SPD) * 5 / 3600
    agg = lambda a: np.convolve(a, np.ones(12) / 12, mode="same")
    pv = truth[sl, 0:3].sum(1); wd = truth[sl, 3:5].sum(1); ld = truth[sl, 7:9].sum(1)
    dg = truth[sl, 5]; bt = truth[sl, 6]
    fig, ax = plt.subplots(2, 2, figsize=(6.6, 5.15))
    a = ax[0, 0]
    for y, c, n in [(pv, fs.CAT[0], "PV"), (wd, fs.CAT[2], "Wind"), (ld, fs.INK, "Load")]:
        a.plot(h, agg(y), color=c, lw=1.4)
    a.set_ylim(0, 84)
    a.text(9.6, 76, "PV", color=fs.CAT[0], fontsize=8.3, fontweight="bold")
    a.text(21.3, 66, "Wind", color=fs.CAT[2], fontsize=8.3, fontweight="bold")
    a.text(17.2, 60, "Load", color=fs.INK, fontsize=8.3, fontweight="bold")
    a.annotate("PV curtailed\n(battery full)", xy=(11.7, 8), xytext=(12.3, 46), fontsize=7.1, color=fs.INK2, arrowprops=dict(arrowstyle="-", color=fs.MUTED, lw=0.6))
    a.set_xlim(0, 24); a.set_xticks(range(0, 25, 6)); a.set_xlabel("Hour of day 12"); a.set_ylabel("Power (kW)"); a.set_title("Generation and demand"); fs.panel(a, "a")
    a = ax[0, 1]
    a.plot(h, agg(bt), color=fs.CAT[4], lw=1.2); a.plot(h, agg(dg), color=fs.CAT[3], lw=1.2); a.axhline(0, color=fs.AXIS, lw=0.8)
    a.text(0.5, agg(bt).max() * 0.85, "Battery (+ discharge)", color=fs.CAT[4], fontsize=8.0, fontweight="bold")
    a.text(0.5, agg(bt).min() * 0.9 if agg(bt).min() < 0 else -5, "Diesel", color=fs.CAT[3], fontsize=8.0, fontweight="bold") if False else None
    a.text(15.2, 52, "Diesel", color="#8a6000", fontsize=8.0, fontweight="bold")
    a.set_xlim(0, 24); a.set_xticks(range(0, 25, 6)); a.set_xlabel("Hour of day 12"); a.set_ylabel("Power (kW)"); a.set_title("Dispatch"); fs.panel(a, "b")
    a = ax[1, 0]
    ax2 = a.twinx()
    a.plot(h, soc[sl] * 100, color=fs.CAT[4], lw=1.3); a.set_ylabel("Battery SOC (%)", color=fs.INK2)
    pyr = truth[sl, 9]
    ax2.plot(h, agg(pyr), color=fs.CAT[1], lw=1.0, alpha=0.9); ax2.grid(False); ax2.spines["right"].set_visible(True); ax2.set_ylabel("Irradiance (W m$^{-2}$)", color=fs.INK2)
    a.set_xlim(0, 24); a.set_xticks(range(0, 25, 6)); a.set_xlabel("Hour of day 12"); a.set_title("State of charge and irradiance"); fs.panel(a, "c")
    a.yaxis.label.set_color(fs.CAT[4]); ax2.yaxis.label.set_color("#b84a18")
    # FDIA examples
    a = ax[1, 1]
    cand = [r for r in S["bursts"] if r["cls"] == "FDIA" and len(r["params"]["sensors"]) == 1 and r["dur"] > 240 and r["t0"] > 5 * 86400]
    v1 = next(r for r in cand if r["params"]["variant"] == "V1-bias"); v2 = next(r for r in cand if r["params"]["variant"] == "V2-ramp")
    for r, off, c in [(v1, 0, fs.CAT[0]), (v2, 0, fs.CAT[1])]:
        s = r["params"]["sensors"][0]
        t0 = int(r["t0"] / 5); t1 = int(r["t1"] / 5)
        lo, hi = max(t0 - 90, 0), t1 + 90
        tt = (np.arange(lo, hi) - t0) * 5 / 60
        a.plot(tt, S["truth"][lo:hi, s], color=c, lw=0.8, alpha=0.55)
        a.plot(tt, S["O"][lo:hi, s], color=c, lw=1.5)
    a.axvline(0, color=fs.AXIS, lw=0.8, ls="--")
    a.set_xlabel("Minutes from attack onset"); a.set_ylabel("Sensor reading (kW)")
    a.set_title("False-data injection: observed vs. true"); fs.panel(a, "d")
    a.text(-7.4, 1.0, "bias (V1)", color=fs.CAT[0], fontsize=8.0, fontweight="bold")
    a.text(-7.4, 0.55, "ramp (V2)", color="#b84a18", fontsize=8.0, fontweight="bold")
    a.text(0.98, 0.97, "thin line = true value", color=fs.INK2, transform=a.transAxes, fontsize=7.9, ha="right", va="top")
    fig.tight_layout(); save(fig, "fig_plant")


def fig_features():
    df = pd.read_pickle(os.path.join(ROOT_DIR, "data", "siteA.pkl"))
    df = df[df.day >= 3]
    feats = [("e_bal_bias10", "Balance bias (5-min mean)", "signed"), ("e_bias10_m", "Multi-scale bias (MQTT)", "signed"),
             ("m_rate", "MQTT message rate (s$^{-1}$)", "pos"), ("c_out_in", "CoAP response / request bytes", "pos"),
             ("m_ttl_dev", "MQTT TTL deviation", "signed"), ("m_wildcard", "MQTT wildcard subscriptions", "pos")]
    fig, ax = plt.subplots(2, 3, figsize=(6.6, 4.70))
    for a, (f, t, kind) in zip(ax.ravel(), feats):
        data, cols = [], []
        for ci, c in enumerate(CL):
            v = df.loc[df.label == ci, f].values.astype(float)
            v = v[np.isfinite(v)]
            if kind == "pos":
                v = np.log1p(np.maximum(v, 0))
            else:
                v = np.sign(v) * np.log1p(np.abs(v) * 1)
            data.append(v)
        bp = a.boxplot(data, vert=True, widths=0.65, showfliers=False, patch_artist=True, whis=(5, 95), medianprops=dict(color=fs.INK, lw=1.2),
                       whiskerprops=dict(color=fs.AXIS), capprops=dict(color=fs.AXIS), boxprops=dict(lw=0))
        for p, c in zip(bp["boxes"], CL): p.set_facecolor(fs.CLS_COL[c]); p.set_alpha(0.9)
        a.set_xticks(range(1, 8), ["N", "F", "Fl", "Am", "MI", "Re", "Rc"], fontsize=7.6); a.set_title(t, fontsize=8.2)
        a.grid(axis="x", visible=False)
    for a, l in zip(ax.ravel(), "abcdef"): fs.panel(a, l)
    fig.text(0.5, -0.01, "N Normal · F FDIA · Fl Flood · Am Amplification · MI MITM · Re Replay · Rc Recon  (values on log1p scale; box = IQR, whiskers = 5–95%)", ha="center", fontsize=7.3, color=fs.INK2)
    fig.tight_layout(); save(fig, "fig_features")


if __name__ == "__main__":
    which = sys.argv[1:] or ["pipeline", "timeline", "plant", "features"]
    for w in which:
        globals()["fig_" + w]()
