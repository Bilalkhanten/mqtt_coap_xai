import os
"""Conceptual figures: framework flowchart, system architecture, feature pipeline, evaluation protocol,
model design, training diagnostics and explanation workflow.  Same style module as the data figures."""
import sys, json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Polygon
from common import *
import figstyle as fs
fs.apply()
FIG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figs")
CL = fs.CLASSES

T_BLUE, T_AMBER, T_GREEN, T_RED, T_GRAY = "#eaf2fc", "#fff4dc", "#e8f5ee", "#fdeae4", "#f1f0ec"
E_BLUE, E_AMBER, E_GREEN, E_RED, E_GRAY = "#9dc0ee", "#e9bf62", "#8fd0b0", "#f0a68c", "#c3c2b7"
GCOL = dict(zip(["Traffic volume", "Source dynamics", "Temporal integrity", "MQTT semantics", "CoAP semantics",
                 "Energy residuals", "Energy bias (multi-scale)", "Context"], fs.CAT))


def save(fig, name):
    fig.savefig(f"{FIG}/{name}.png", dpi=300, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    print("saved", name, flush=True)


def canvas(W, H, wi=6.6):
    fig, ax = plt.subplots(figsize=(wi, wi * H / W))
    ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.axis("off"); ax.grid(False)
    fig.subplots_adjust(0, 0, 1, 1)
    return fig, ax


def box(ax, x, y, w, h, title, lines=(), fc="white", ec=fs.AXIS, tsz=8.2, lsz=7.2, tcol=fs.INK, lw=1.0, ls="-", lsp=0.30, talign="center", tdy=0.30):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.12", fc=fc, ec=ec, lw=lw, ls=ls, zorder=2))
    tx = x + w / 2 if talign == "center" else x + 0.15
    ax.text(tx, y + tdy, title, ha=talign, va="center", fontsize=tsz, fontweight="bold", color=tcol, zorder=3)
    for i, l in enumerate(lines):
        ax.text(tx, y + tdy + 0.42 + lsp * i, l, ha=talign, va="center", fontsize=lsz, color=fs.INK2, zorder=3)


def arrow(ax, p0, p1, color=fs.INK2, lw=1.1, style="-|>", rad=0.0, ls="-", ms=8, z=1):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle=style, mutation_scale=ms, lw=lw, color=color, shrinkA=0, shrinkB=0,
                                 connectionstyle=f"arc3,rad={rad}", ls=ls, zorder=z))


def poly(ax, pts, color=fs.INK2, lw=1.1, ls="-", arrow_end=True, ms=8, z=1):
    """Orthogonal connector through the given points, arrow head on the last segment."""
    for a, b in zip(pts[:-2], pts[1:-1]):
        ax.plot([a[0], b[0]], [a[1], b[1]], color=color, lw=lw, ls=ls, solid_capstyle="butt", zorder=z)
    arrow(ax, pts[-2], pts[-1], color=color, lw=lw, ls=ls, ms=ms, z=z) if arrow_end else ax.plot(
        [pts[-2][0], pts[-1][0]], [pts[-2][1], pts[-1][1]], color=color, lw=lw, ls=ls, zorder=z)


def band(ax, x, y, w, text, color, tcol=fs.INK):
    ax.add_patch(plt.Rectangle((x, y), w, 0.42, fc=color, ec="none", zorder=1))
    ax.text(x + 0.15, y + 0.21, text, ha="left", va="center", fontsize=8.0, fontweight="bold", color=tcol, zorder=3)


def badge(ax, x, y, n, color=fs.INK):
    ax.add_patch(plt.Circle((x, y), 0.21, fc=color, ec="none", zorder=4))
    ax.text(x, y + 0.01, str(n), ha="center", va="center", fontsize=7.4, fontweight="bold", color="white", zorder=5)


# ======================================================================================= 1. framework flowchart
def fig_framework():
    W, H = 12.0, 8.7
    fig, ax = canvas(W, H)
    bw, gap, x0 = 2.76, 0.32, 0.0
    X = [x0 + i * (bw + gap) for i in range(4)]
    bh = 1.62
    rows = [(0.0, T_BLUE, E_BLUE, "Stage 1  Simulate: plant, network, attacks", [
                ("Plant model", ["PV ×3, wind ×2, battery,", "diesel genset, two feeders", "exact power balance"]),
                ("Field network", ["11 sensors, 5–10 s period", "MQTT/TCP and CoAP/UDP", "RTT, loss, retry, skew"]),
                ("Event scheduler", ["878 attack bursts (6 families)", "FDIA: bias, ramp, balanced", "5 benign look-alikes"]),
                ("Labelled message log", ["0.52 M messages per day", "30 days, sites A and B", "burst identifier per message"])]),
            (3.0, T_AMBER, E_AMBER, "Stage 2  Learn: windows, features, models", [
                ("Windowing", ["30 s windows, 86 400", "label: burst covering ≥ 50 %", "benign-event bit mask"]),
                ("Feature extraction", ["68 features in 8 groups", "baselines from days 0–2", "log1p and z-score"]),
                ("Chronological split", ["train: days 3–19", "validation: days 20–22", "test: days 23–29, purged"]),
                ("Train and tune", ["7 models, 5 seeds", "random search on validation", "class-weighted loss"])]),
            (6.0, T_GREEN, E_GREEN, "Stage 3  Evaluate and explain", [
                ("Held-out test", ["macro-F1, MCC, κ, ECE", "block-bootstrap CIs", "Friedman–Nemenyi, Holm"]),
                ("Stress tests", ["unseen FDIA variants", "attack-intensity scaling", "cross-site transfer"]),
                ("Explainers", ["TreeSHAP, KernelSHAP", "LIME, Integrated Gradients", "group-level SHAP"]),
                ("Explanation audit", ["deletion and insertion", "stability and agreement", "latency budget"])])]
    centers = []
    for (y, tc, ec, head, items) in rows:
        band(ax, 0, y, W, head, tc)
        for i, (t, ls_) in enumerate(items):
            fc = T_RED if (t == "Event scheduler") else "white"
            e = E_RED if t == "Event scheduler" else ec
            box(ax, X[i], y + 0.62, bw, bh, t, ls_, fc=fc, ec=e, tsz=8.0, lsz=7.0)
        for i in range(3):
            arrow(ax, (X[i] + bw + 0.01, y + 0.62 + bh / 2), (X[i + 1] - 0.01, y + 0.62 + bh / 2))
        centers.append(y + 0.62 + bh)
    # stage hand-overs (orthogonal connectors in the gaps)
    for k in range(2):
        yb = centers[k]; yt = rows[k + 1][0]
        poly(ax, [(X[3] + bw / 2, yb), (X[3] + bw / 2, yb + 0.5), (X[0] + bw / 2, yb + 0.5), (X[0] + bw / 2, yt + 0.0)], z=1)
    # validation feedback loop: Train and tune -> Chronological split
    yb = rows[1][0] + 0.62 + bh
    poly(ax, [(X[3] + bw * 0.18, yb), (X[3] + bw * 0.18, yb + 0.16), (X[2] + bw * 0.5, yb + 0.16), (X[2] + bw * 0.5, yb + 0.01)], color="#b9831a", lw=1.2)
    ax.text(X[2] + bw * 0.5 - 0.1, yb + 0.16, "validation macro-F1 feeds back", ha="right", va="center", fontsize=6.6, color=fs.INK2, style="italic")
    save(fig, "fig_framework")


# ======================================================================================= 2. system architecture + threat model
def fig_architecture():
    W, H = 12.0, 8.5
    fig, ax = canvas(W, H)
    RED = "#a82a29"
    # field
    ax.add_patch(FancyBboxPatch((0.0, 0.2), 3.2, 5.3, boxstyle="round,pad=0,rounding_size=0.12", fc=T_GRAY, ec=fs.AXIS, lw=1.0, zorder=1))
    ax.text(1.6, 0.55, "Hybrid microgrid (field)", ha="center", va="center", fontsize=8.2, fontweight="bold")
    dev = [("PV inverter ×3", 0), ("Wind controller ×2", 0), ("Diesel genset", 0), ("Battery system", 0), ("Feeder meter ×2", 1), ("Pyranometer", 1), ("Anemometer", 1)]
    for i, (n, tr) in enumerate(dev):
        yy = 0.95 + i * 0.58
        c = fs.CAT[0] if tr == 0 else fs.CAT[2]
        ax.add_patch(FancyBboxPatch((0.25, yy), 2.7, 0.46, boxstyle="round,pad=0,rounding_size=0.08", fc="white", ec=c, lw=1.2, zorder=2))
        ax.text(0.45, yy + 0.23, n, ha="left", va="center", fontsize=7.2, color=fs.INK, zorder=3)
        ax.text(2.82, yy + 0.23, "MQTT" if tr == 0 else "CoAP", ha="right", va="center", fontsize=6.2, color=c, fontweight="bold", zorder=3)
    ax.text(1.6, 5.2, "7 MQTT devices, 4 CoAP devices", ha="center", va="center", fontsize=6.8, color=fs.INK2)
    # network
    ax.add_patch(FancyBboxPatch((4.0, 0.2), 4.7, 5.3, boxstyle="round,pad=0,rounding_size=0.12", fc="white", ec=fs.AXIS, lw=1.0, zorder=1))
    ax.text(6.35, 0.55, "Field network", ha="center", va="center", fontsize=8.2, fontweight="bold")
    box(ax, 4.2, 1.05, 2.6, 1.75, "MQTT over TCP", ["publish every 5 s", "broker, QoS 0/1", "TTL, RTT, keep-alive"], fc=T_BLUE, ec=E_BLUE, tsz=7.8, lsz=6.8, lsp=0.29)
    box(ax, 4.2, 3.05, 2.6, 1.75, "CoAP over UDP", ["observe every 10 s", "70 % CON, RFC 7252", "duplicates, MIDs, sizes"], fc=T_GREEN, ec=E_GREEN, tsz=7.8, lsz=6.8, lsp=0.29)
    box(ax, 7.15, 1.9, 1.4, 1.95, "Gateway", ["passive", "copy of all", "messages"], fc=T_GRAY, ec=E_GRAY, tsz=7.8, lsz=6.6, lsp=0.28)
    arrow(ax, (6.82, 1.92), (7.13, 2.35), ms=6); arrow(ax, (6.82, 3.92), (7.13, 3.4), ms=6)
    # control centre
    ax.add_patch(FancyBboxPatch((9.3, 0.2), 2.7, 5.3, boxstyle="round,pad=0,rounding_size=0.12", fc=T_AMBER, ec=E_AMBER, lw=1.0, zorder=1))
    ax.text(10.65, 0.55, "Control centre", ha="center", va="center", fontsize=8.2, fontweight="bold")
    cc = [("30 s windows", "68 features"), ("Classifier", "Normal + 6 attacks"), ("Explainer", "TreeSHAP"), ("Operator view", "class + top features")]
    for i, (a_, b_) in enumerate(cc):
        yy = 0.95 + i * 1.12
        box(ax, 9.5, yy, 2.3, 0.85, a_, [b_], fc="white", ec=E_AMBER, tsz=7.6, lsz=6.7, lsp=0.28, tdy=0.24)
        if i < 3:
            arrow(ax, (10.65, yy + 0.85), (10.65, yy + 1.12), ms=6)
    poly(ax, [(8.57, 2.85), (8.95, 2.85), (8.95, 1.38), (9.48, 1.38)], lw=1.4)
    # flows device -> lane
    arrow(ax, (3.25, 1.95), (4.15, 1.95), color=fs.CAT[0], lw=1.4); arrow(ax, (3.25, 3.9), (4.15, 3.9), color=fs.CAT[2], lw=1.4)
    # adversary and benign events
    box(ax, 4.0, 6.85, 3.8, 1.65, "Adversary", ["injects, replays or modifies messages;", "spoofs UDP source addresses;", "enumerates topics and resources"], fc=T_RED, ec=E_RED, tsz=8.0, lsz=6.7, lsp=0.28)
    box(ax, 9.3, 6.85, 2.7, 1.65, "Benign look-alikes", ["sensor faults, outages,", "route changes, reconnects,", "admin discovery"], fc=T_GRAY, ec=E_GRAY, tsz=7.8, lsz=6.5, lsp=0.28)
    arrow(ax, (4.75, 6.85), (4.75, 5.52), color=fs.CAT[7], lw=1.5)
    ax.text(4.95, 5.85, "Network attacks:", ha="left", va="center", fontsize=6.9, color=RED, fontweight="bold")
    ax.text(4.95, 6.15, "Flood, Amplification,", ha="left", va="center", fontsize=6.9, color=RED)
    ax.text(4.95, 6.45, "MITM, Replay, Recon", ha="left", va="center", fontsize=6.9, color=RED)
    poly(ax, [(4.0, 7.7), (3.6, 7.7), (3.6, 4.05)], color=fs.CAT[7], lw=1.5, ls=(0, (4, 2)))
    ax.text(3.5, 6.0, "FDIA: readings", ha="right", va="center", fontsize=6.9, color=RED, fontweight="bold")
    ax.text(3.5, 6.3, "altered in transit", ha="right", va="center", fontsize=6.9, color=RED)
    poly(ax, [(9.3, 7.7), (8.45, 7.7), (8.45, 5.52)], color=fs.GRAY, lw=1.3, ls=(0, (3, 2)))
    ax.text(8.35, 6.2, "enter the", ha="right", va="center", fontsize=6.6, color=fs.INK2)
    ax.text(8.35, 6.5, "same traffic", ha="right", va="center", fontsize=6.6, color=fs.INK2)
    save(fig, "fig_architecture")


# ======================================================================================= 3. feature-extraction pipeline
def fig_featpipe():
    W, H = 12.0, 7.6
    fig, ax = canvas(W, H)
    box(ax, 0.0, 0.0, 3.2, 2.2, "Messages in window t", ["MQTT and CoAP events", "(time, source, size, TTL,", "RTT, sequence, flags)", "11 sensor readings"], fc=T_BLUE, ec=E_BLUE, tsz=8.0, lsz=7.0, lsp=0.29)
    box(ax, 0.0, 2.7, 3.2, 2.2, "Commissioning, days 0–2", ["per-device RTT, TTL, rate", "balance-residual scale", "(median, MAD)", "diurnal clear-sky profile"], fc=T_GRAY, ec=E_GRAY, tsz=8.0, lsz=7.0, lsp=0.29)
    ax.text(1.6, 5.1, "calibration only,", ha="center", va="top", fontsize=6.8, color=fs.INK2, style="italic")
    ax.text(1.6, 5.4, "never used for fitting", ha="center", va="top", fontsize=6.8, color=fs.INK2, style="italic")

    def group_rows(y0, rows):
        for i, (g, n, ex) in enumerate(rows):
            yy = y0 + i * 0.66
            ax.add_patch(FancyBboxPatch((3.85, yy), 3.8, 0.58, boxstyle="round,pad=0,rounding_size=0.06", fc=T_GRAY, ec="none", zorder=2))
            ax.add_patch(plt.Rectangle((3.85, yy), 0.12, 0.58, fc=GCOL[g], ec="none", zorder=3))
            ax.text(4.12, yy + 0.18, g.replace(" (multi-scale)", ""), ha="left", va="center", fontsize=7.0, fontweight="bold", color=fs.INK, zorder=3)
            ax.text(4.12, yy + 0.43, ex, ha="left", va="center", fontsize=6.0, color=fs.INK2, zorder=3)
            ax.text(7.5, yy + 0.29, f"{n}", ha="right", va="center", fontsize=8.0, fontweight="bold", color=fs.INK, zorder=3)
    ax.add_patch(FancyBboxPatch((3.7, 0.0), 4.1, 4.2, boxstyle="round,pad=0,rounding_size=0.12", fc="white", ec=E_BLUE, lw=1.0, zorder=1))
    ax.text(3.9, 0.3, "Network branch: 45 features", ha="left", va="center", fontsize=8.0, fontweight="bold")
    group_rows(0.65, [("Traffic volume", 14, "rates, byte rates, burstiness, sizes"), ("Source dynamics", 8, "distinct sources, entropy, new sources"),
                      ("Temporal integrity", 12, "RTT, TTL deviation, skew, gaps, duplicates"), ("MQTT semantics", 6, "topics, connects, subscriptions, wildcards"),
                      ("CoAP semantics", 5, "CON share, paths, ports, 4.04 rate")])
    ax.add_patch(FancyBboxPatch((3.7, 4.5), 4.1, 2.9, boxstyle="round,pad=0,rounding_size=0.12", fc="white", ec=E_GREEN, lw=1.0, zorder=1))
    ax.text(3.9, 4.8, "Energy branch: 21 + 2 features", ha="left", va="center", fontsize=8.0, fontweight="bold")
    group_rows(5.15, [("Energy residuals", 13, "balance, physics, EWMA, CUSUM, flatline"), ("Energy bias (multi-scale)", 8, "signed bias at 5 and 10 min, peer bias"),
                      ("Context", 2, "time of day (sine, cosine)")])
    box(ax, 8.4, 1.2, 3.6, 1.7, "Feature vector", ["x(t) ∈ R^68", "log1p, then z-score with", "commissioning statistics"], fc=T_AMBER, ec=E_AMBER, tsz=8.0, lsz=7.0, lsp=0.29)
    box(ax, 8.4, 3.5, 3.6, 2.9, "Model inputs", ["single window: 68", "lag stack (t−4 … t):", "5 × 68 = 340", "sequence for CNN–LSTM:", "5 steps × 68 channels"], fc="white", ec=E_AMBER, tsz=8.0, lsz=7.0, lsp=0.30)
    arrow(ax, (3.25, 1.1), (3.7, 1.1)); arrow(ax, (3.25, 3.8), (3.7, 3.8))
    poly(ax, [(7.85, 2.1), (8.38, 2.1)])
    poly(ax, [(7.85, 5.9), (8.1, 5.9), (8.1, 2.6), (8.38, 2.6)])
    arrow(ax, (10.2, 2.9), (10.2, 3.5), ms=7)
    save(fig, "fig_featpipe")


# ======================================================================================= 4. training / validation / test protocol
def fig_split():
    W, H = 12.0, 8.6
    fig, ax = canvas(W, H)
    ax.text(0.0, 0.2, "Chronological protocol (all headline results)", fontsize=8.4, fontweight="bold", va="center")
    x0, sc, y0, h = 0.9, 11.0 / 30, 0.9, 0.9
    segs = [("calib.", 0, 3, fs.GRAY, "white"), ("training", 3, 20, fs.CAT[0], "white"), ("val.", 20, 23, fs.CAT[3], fs.INK), ("test", 23, 30, fs.CAT[1], "white")]
    for nm, a, b, c, tc in segs:
        ax.add_patch(plt.Rectangle((x0 + a * sc + 0.02, y0), (b - a) * sc - 0.04, h, fc=c, ec="none", zorder=2))
        ax.text(x0 + (a + b) / 2 * sc, y0 + h / 2, nm, ha="center", va="center", fontsize=7.6, color=tc, fontweight="bold", zorder=3)
    for d in [0, 3, 20, 23, 30]:
        ax.plot([x0 + d * sc] * 2, [y0 + h, y0 + h + 0.12], color=fs.INK2, lw=0.8)
        ax.text(x0 + d * sc, y0 + h + 0.18, f"day {d}", ha="center", va="top", fontsize=6.8, color=fs.INK2)
    for d in [20, 23]:  # purge gaps (drawn wider than scale)
        ax.add_patch(plt.Rectangle((x0 + d * sc - 0.16, y0 - 0.12), 0.16, h + 0.24, fc="white", ec=fs.INK2, hatch="////", lw=0.8, zorder=4))
    ax.text(x0 + 20 * sc - 0.1, y0 - 0.2, "purge gap: 24 windows (12 min)", ha="center", va="bottom", fontsize=6.6, color=fs.INK2, style="italic")
    ax.text(x0 + 23 * sc - 0.1, y0 - 0.2, "", ha="center")
    roles = [(0, 3, "8 640 windows", ["baselines and scales", "never fitted on"]), (3, 20, "48 936 windows", ["fit parameters", "6 736 attack windows"]),
             (20, 23, "8 616 windows", ["hyper-parameters,", "early stopping"]), (23, 30, "20 160 windows", ["scored once", "2 628 attack windows"])]
    for a, b, t, ls_ in roles:
        cx = x0 + (a + b) / 2 * sc
        ax.text(cx - (0.25 if a == 20 else 0), 2.35, t, ha="center", va="top", fontsize=7.0, fontweight="bold", color=fs.INK)
        for i, l in enumerate(ls_):
            ax.text(cx, 2.65 + 0.27 * i, l, ha="center", va="top", fontsize=6.6, color=fs.INK2)
    # contrast protocols
    rng = np.random.default_rng(3)
    ax.text(0.0, 4.15, "Contrast protocols", fontsize=8.4, fontweight="bold", va="center")
    n = 27 * 2
    cw = (27 * sc) / n
    xs = x0 + 3 * sc
    ylab = [4.75, 5.95]
    cols = {"tr": fs.CAT[0], "va": fs.CAT[3], "te": fs.CAT[1]}
    ax.text(0.0, ylab[0] + 0.3, "Random", ha="left", va="center", fontsize=7.2, fontweight="bold")
    ax.text(0.0, ylab[0] + 0.62, "70/10/20", ha="left", va="center", fontsize=6.6, color=fs.INK2)
    ax.text(0.0, ylab[1] + 0.3, "Burst-", ha="left", va="center", fontsize=7.2, fontweight="bold")
    ax.text(0.0, ylab[1] + 0.62, "grouped CV", ha="left", va="center", fontsize=6.6, color=fs.INK2)
    r = rng.random(n)
    for i in range(n):
        c = cols["tr"] if r[i] < 0.7 else cols["va"] if r[i] < 0.8 else cols["te"]
        ax.add_patch(plt.Rectangle((xs + i * cw + 0.01, ylab[0]), cw - 0.02, 0.8, fc=c, ec="none"))
    # grouped: contiguous blocks, each held out in one fold
    i, k = 0, 0
    fold_col = [fs.CAT[1], fs.CAT[0], fs.CAT[0], fs.CAT[0], fs.CAT[0]]
    seq_ = rng.integers(2, 6, size=40)
    fold_pat = rng.integers(0, 5, size=40)
    for L, f in zip(seq_, fold_pat):
        if i >= n: break
        L = min(L, n - i)
        c = fs.CAT[1] if f == 0 else fs.CAT[0]
        ax.add_patch(plt.Rectangle((xs + i * cw + 0.01, ylab[1]), L * cw - 0.02, 0.8, fc=c, ec="none"))
        i += L
    ax.text(xs, 4.5, "days 3–29 (windows, schematic)", ha="left", va="center", fontsize=6.6, color=fs.INK2, style="italic")
    ax.text(0.0, 7.1, "Median distance from a test window to the nearest training window:", fontsize=7.2, color=fs.INK2, va="center")
    for xx, (nm, v) in zip([0.9, 3.9, 7.0], [("random", "1 window (30 s)"), ("burst-grouped", "16 windows (8 min)"), ("chronological", "18 744 windows (6.5 days)")]):
        ax.text(xx, 7.55, nm, fontsize=7.2, fontweight="bold", va="center")
        ax.text(xx, 7.85, v, fontsize=7.0, color=fs.INK2, va="center")
    # legend
    lx = 6.1
    for j, (c, t) in enumerate([(fs.CAT[0], "train"), (fs.CAT[3], "validation"), (fs.CAT[1], "test / held-out fold")]):
        xx = lx + [0, 1.1, 2.7][j]
        ax.add_patch(plt.Rectangle((xx, 4.04), 0.3, 0.22, fc=c, ec="none"))
        ax.text(xx + 0.4, 4.15, t, ha="left", va="center", fontsize=6.9, color=fs.INK2)
    save(fig, "fig_split")


# ======================================================================================= 5. model design
def fig_models_design():
    W, H = 12.0, 7.4
    fig, ax = canvas(W, H)
    box(ax, 0.0, 0.5, 2.5, 1.7, "x(t)", ["68 features", "single window"], fc=T_BLUE, ec=E_BLUE, tsz=8.4, lsz=7.2)
    box(ax, 0.0, 2.7, 2.5, 1.7, "x(t−4 … t)", ["5 × 68 = 340", "lag stack"], fc=T_BLUE, ec=E_BLUE, tsz=8.4, lsz=7.2)
    box(ax, 0.0, 4.9, 2.5, 1.7, "z(t−4 … t)", ["5 steps × 68", "standardised sequence"], fc=T_BLUE, ec=E_BLUE, tsz=8.4, lsz=7.2)
    lines = [("Logistic regression", "multinomial, ℓ2, C = 30.6", 0.30), ("Random forest", "400 trees, depth 14, √p split", 0.30),
             ("XGBoost", "600 trees, depth 3, η = 0.077", 0.30), ("LightGBM", "600 trees, 63 leaves, η = 0.19", 0.30), ("MLP", "68–256–256–7, ReLU, dropout 0.3", 0.30)]
    mx, mw = 3.2, 5.0
    for i, (t, d, _) in enumerate(lines):
        yy = 0.15 + i * 0.72
        ax.add_patch(FancyBboxPatch((mx, yy), mw, 0.6, boxstyle="round,pad=0,rounding_size=0.08", fc="white", ec=fs.AXIS, lw=1.0, zorder=2))
        ax.text(mx + 0.15, yy + 0.3, t, ha="left", va="center", fontsize=7.4, fontweight="bold", zorder=3)
        ax.text(mx + mw - 0.12, yy + 0.3, d, ha="right", va="center", fontsize=6.4, color=fs.INK2, zorder=3)
        arrow(ax, (2.55, 1.35), (mx - 0.02, yy + 0.3), ms=6, lw=0.8)
    # lag model
    yy = 3.9
    ax.add_patch(FancyBboxPatch((mx, yy), mw, 0.78, boxstyle="round,pad=0,rounding_size=0.08", fc="white", ec=E_BLUE, lw=1.4, zorder=2))
    ax.text(mx + 0.15, yy + 0.25, "XGBoost, 5-window lag", ha="left", va="center", fontsize=7.4, fontweight="bold", zorder=3)
    ax.text(mx + 0.15, yy + 0.55, "500 trees, depth 3, η = 0.043", ha="left", va="center", fontsize=6.4, color=fs.INK2, zorder=3)
    arrow(ax, (2.55, 3.55), (mx - 0.02, yy + 0.39), ms=6, lw=0.8)
    # CNN-LSTM schematic
    yy = 5.0
    ax.add_patch(FancyBboxPatch((mx, yy), mw, 1.6, boxstyle="round,pad=0,rounding_size=0.08", fc="white", ec=fs.AXIS, lw=1.0, zorder=2))
    ax.text(mx + 0.15, yy + 0.28, "CNN–LSTM", ha="left", va="center", fontsize=7.4, fontweight="bold", zorder=3)
    ax.text(mx + mw - 0.12, yy + 0.28, "dropout 0.4, Adam 2e−3", ha="right", va="center", fontsize=6.4, color=fs.INK2, zorder=3)
    blocks = ["Conv1d\n64, k = 3", "Conv1d\n128, k = 3", "LSTM\n100", "FC 64\nReLU", "FC 7"]
    bw_ = 0.88
    for i, b in enumerate(blocks):
        bx = mx + 0.15 + i * (bw_ + 0.12)
        ax.add_patch(FancyBboxPatch((bx, yy + 0.55), bw_, 0.85, boxstyle="round,pad=0,rounding_size=0.06", fc=T_GRAY, ec="none", zorder=3))
        ax.text(bx + bw_ / 2, yy + 0.975, b, ha="center", va="center", fontsize=6.3, color=fs.INK, zorder=4, linespacing=1.1)
        if i < 4:
            arrow(ax, (bx + bw_ + 0.005, yy + 0.975), (bx + bw_ + 0.115, yy + 0.975), ms=5, lw=0.7, z=4)
    arrow(ax, (2.55, 5.75), (mx - 0.02, yy + 0.8), ms=6, lw=0.8)
    # output
    box(ax, 9.0, 1.4, 3.0, 2.2, "Softmax output", ["P(Normal), P(FDIA),", "P(Flood), P(Amplification),", "P(MITM), P(Replay), P(Recon)"], fc=T_AMBER, ec=E_AMBER, tsz=8.0, lsz=6.8, lsp=0.30)
    box(ax, 9.0, 4.15, 3.0, 2.45, "Training", ["class weights ∝ (N/Kn_c)^0.5", "random search on validation", "5 seeds per model", "networks: early stopping,", "patience 6 on macro-F1"], fc="white", ec=fs.AXIS, tsz=8.0, lsz=6.8, lsp=0.29)
    ys_ = [0.45, 1.17, 1.89, 2.61, 3.33, 4.29, 5.8]
    for y_ in ys_:
        ax.plot([mx + mw, 8.55], [y_, y_], color=fs.INK2, lw=0.9, zorder=1)
    ax.plot([8.55, 8.55], [ys_[0], ys_[-1]], color=fs.INK2, lw=0.9, zorder=1)
    arrow(ax, (8.55, 2.5), (8.98, 2.5), lw=1.2)
    save(fig, "fig_models_design")


# ======================================================================================= 6. training diagnostics
def fig_training():
    from matplotlib.lines import Line2D
    c = jload("curves")["curves"]
    t = jload("tuning")
    fig, ax = plt.subplots(1, 3, figsize=(6.6, 3.1), gridspec_kw=dict(width_ratios=[1, 1, 1.15]))
    for nm, lab, col in (("mlp", "MLP", fs.MODEL_COL["mlp"]), ("cnnlstm", "CNN–LSTM", fs.MODEL_COL["cnnlstm"])):
        cv = np.array(c[nm])
        ep, tl, vl, f1 = cv[:, 0], cv[:, 1], cv[:, 2], cv[:, 3]
        ax[0].plot(ep, tl, "-", color=col, lw=1.5)
        ax[0].plot(ep, vl, "--", color=col, lw=1.5)
        b = int(np.argmax(f1))
        ax[1].plot(ep, f1, "-", color=col, lw=1.5)
        ax[1].plot(ep[b], f1[b], "o", mfc="none", color=fs.INK, ms=9, mew=1.3, zorder=5)
    ax[0].set_xlabel("Epoch"); ax[0].set_ylabel("Weighted cross-entropy"); ax[0].set_title("Loss")
    ax[1].set_xlabel("Epoch"); ax[1].set_ylabel("Validation macro-F1"); ax[1].set_title("Early stopping")
    names = ["lr", "rf", "xgb", "lgbm", "mlp", "cnnlstm", "xgblag"]
    short = {"lr": "LR", "rf": "RF", "xgb": "XGB", "lgbm": "LGBM", "mlp": "MLP", "cnnlstm": "CNN–LSTM", "xgblag": "XGB-lag"}
    rng = np.random.default_rng(1)
    for i, k in enumerate(names):
        v = np.array([tr["val_macro_f1"] for tr in t[k]["trials"]])
        ax[2].plot(i + rng.uniform(-0.12, 0.12, len(v)), v, "o", color=fs.MODEL_COL[k], ms=4.2, alpha=0.75, mec="none")
        ax[2].plot(i, t[k]["best_val_f1"], "o", mfc="none", mec=fs.INK, mew=1.3, ms=9)
    ax[2].set_xticks(range(7), [short[k] for k in names], fontsize=7.0, rotation=40, ha="right", rotation_mode="anchor")
    ax[2].set_ylabel("Validation macro-F1"); ax[2].set_title("Random search"); ax[2].grid(axis="x", visible=False)
    ax[2].set_ylim(0.70, 0.88)
    fig.tight_layout()
    h = [Line2D([], [], color=fs.MODEL_COL["mlp"], lw=1.6), Line2D([], [], color=fs.MODEL_COL["cnnlstm"], lw=1.6),
         Line2D([], [], color=fs.INK2, lw=1.5), Line2D([], [], color=fs.INK2, lw=1.5, ls="--"),
         Line2D([], [], marker="o", mfc="none", mec=fs.INK, mew=1.3, ms=8, lw=0)]
    fig.legend(h, ["MLP", "CNN–LSTM", "training loss", "validation loss", "restored epoch / selected trial"], ncol=5, fontsize=7.4,
               loc="lower center", bbox_to_anchor=(0.5, -0.07), handlelength=1.8, columnspacing=1.0)
    save(fig, "fig_training")


# ======================================================================================= 7. explanation workflow
def fig_xai_flow():
    W, H = 12.0, 6.2
    fig, ax = canvas(W, H)
    box(ax, 0.0, 1.5, 2.2, 2.6, "Sample", ["1 050 test windows", "150 per class,", "stratified by class"], fc=T_BLUE, ec=E_BLUE, tsz=8.2, lsz=6.9, lsp=0.30)
    ex = ["TreeSHAP (exact, trees)", "KernelSHAP (check)", "LIME (own tabular)", "Integrated Gradients", "gain, permutation", "random (null)"]
    ax.add_patch(FancyBboxPatch((2.6, 0.0), 2.6, 5.6, boxstyle="round,pad=0,rounding_size=0.12", fc=T_AMBER, ec=E_AMBER, lw=1.0, zorder=1))
    ax.text(3.9, 0.3, "Explainers", ha="center", va="center", fontsize=8.2, fontweight="bold")
    for i, e in enumerate(ex):
        ax.add_patch(FancyBboxPatch((2.75, 0.65 + i * 0.77), 2.3, 0.62, boxstyle="round,pad=0,rounding_size=0.06", fc="white", ec="none", zorder=2))
        ax.text(3.9, 0.96 + i * 0.77, e, ha="center", va="center", fontsize=6.9, color=fs.INK, zorder=3)
    box(ax, 5.8, 0.0, 3.8, 1.2, "Attribution ranking φ(x)", ["scores of every feature for the predicted class"], fc="white", ec=fs.AXIS, tsz=7.8, lsz=6.4)
    aud = [("Faithfulness", "deletion / insertion AUC;", "comprehensiveness, sufficiency @ 5"), ("Stability", "input noise of 5 % of σ:", "top-5 Jaccard, relative ℓ2 change"),
           ("Agreement", "Spearman, Kendall τ and Jaccard", "between methods and models"), ("Group view", "SHAP summed over the eight", "feature groups, per class")]
    for i, (t, a_, b_) in enumerate(aud):
        yy = 1.4 + i * 1.1
        box(ax, 5.8, yy, 3.8, 1.0, t, [a_, b_], fc=T_GREEN, ec=E_GREEN, tsz=7.6, lsz=6.3, lsp=0.24, tdy=0.19)
    box(ax, 10.1, 1.5, 1.9, 2.6, "Report", ["method and value", "function stated;", "faithfulness and", "stability given", "with every", "explanation"], fc="white", ec=fs.AXIS, tsz=8.0, lsz=6.6, lsp=0.28)
    arrow(ax, (2.22, 2.8), (2.58, 2.8))
    poly(ax, [(5.22, 0.6), (5.8, 0.6)])
    ax.plot([5.5, 5.5], [0.6, 5.2], color=fs.INK2, lw=0.9, zorder=1)
    for i in range(4):
        arrow(ax, (5.5, 1.9 + i * 1.1), (5.78, 1.9 + i * 1.1), ms=6, lw=0.9)
        ax.plot([9.6, 9.85], [1.9 + i * 1.1] * 2, color=fs.INK2, lw=0.9, zorder=1)
    ax.plot([9.85, 9.85], [1.9, 5.2], color=fs.INK2, lw=0.9, zorder=1)
    arrow(ax, (9.85, 3.0), (10.08, 3.0), ms=6, lw=1.0)
    save(fig, "fig_xai_flow")


if __name__ == "__main__":
    for n in sys.argv[1:] or ["framework", "architecture", "featpipe", "split", "models_design", "training", "xai_flow"]:
        globals()["fig_" + n]()
