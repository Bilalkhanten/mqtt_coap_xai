"""One figure style for the whole paper (validated categorical palette, light surface, thin marks, direct labels)."""
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

SURF, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
GRAY = "#898781"
# class colours: Normal neutral, attacks take slots 1..6 in fixed order
CLASSES = ["Normal", "FDIA", "Flood", "Amplification", "MITM", "Replay", "Recon"]
CLS_COL = dict(zip(CLASSES, [GRAY] + CAT[:6]))
MODEL_COL = dict(zip(["lr", "rf", "xgb", "lgbm", "mlp", "cnnlstm", "xgblag"], CAT[:7]))
MARK = dict(zip(["lr", "rf", "xgb", "lgbm", "mlp", "cnnlstm", "xgblag"], ["o", "s", "^", "D", "v", "P", "X"]))
SEQ = LinearSegmentedColormap.from_list("seq", ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
DIV = LinearSegmentedColormap.from_list("div", ["#e34948", "#f0efec", "#2a78d6"])  # negative red, positive blue


def apply():
    mpl.rcParams.update({
        "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF, "figure.dpi": 110, "savefig.dpi": 300,
        "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans", "Liberation Sans", "Arial"], "font.size": 9,
        "axes.labelsize": 9, "axes.titlesize": 9.5, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "axes.labelcolor": INK2, "text.color": INK,
        "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
        "xtick.major.size": 2.5, "ytick.major.size": 2.5, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
        "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False, "legend.fontsize": 8.3,
        "lines.linewidth": 1.6, "lines.markersize": 5, "patch.linewidth": 0.0, "axes.prop_cycle": mpl.cycler(color=CAT),
        "pdf.fonttype": 42, "svg.fonttype": "none"})


def label(ax, x, y, s, color=INK2, **kw):
    ax.text(x, y, s, color=color, fontsize=kw.pop("fontsize", 8.3), **kw)


def panel(ax, letter):
    """Panel letters are deliberately not drawn (panels are identified by their titles)."""
    return
