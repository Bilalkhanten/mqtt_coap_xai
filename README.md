# MQTT/CoAP hybrid-microgrid attack classification and explanation — implementation

A message-level co-simulation of a PV–wind–diesel–battery microgrid that reports over MQTT (TCP) and CoAP (UDP),
with six attack families and five benign look-alike classes, a 68-feature extractor, seven classifiers, a leakage-aware
evaluation protocol, stress tests and an explanation audit (TreeSHAP, KernelSHAP, LIME, Integrated Gradients).
All data are synthetic and exactly reproducible from the seeds in `src/config.py`.

Tested with Python 3.13 (NumPy 2.5, SciPy 1.18, pandas 3.0, scikit-learn 1.9, XGBoost 3.4, LightGBM 4.7, PyTorch 2.14,
SHAP 0.54, matplotlib 3.11) on 2 virtual CPU cores, no GPU.

## Quick start
```bash
pip install -r requirements.txt
bash src/run_all.sh          # data -> tuning -> all experiments -> figures (skips steps whose outputs exist)
```
The data are not shipped (they are deterministic and small to regenerate): `python3 src/generate_data.py` writes
`data/siteA.pkl` and `data/siteB.pkl` in about 15 s per site, and `run_all.sh` does this automatically when they are missing.
The generator was verified to reproduce the original data bit for bit. All result files and figures are shipped, so after
generating the data any single stage can be re-run on its own, e.g. `cd src && python3 exp_stats.py`
(which reproduces `results/stats.json` exactly).

## Layout
| Path | Content |
|---|---|
| `src/config.py` | all simulator parameters, class taxonomy, sensors, `SITE_A` / `SITE_B` (seeds 20261022 / 77031) |
| `src/plant.py` | clear-sky irradiance (Haurwitz) + OU clearness, Weibull-copula wind, cubic power curve, load, battery/diesel dispatch with exact power balance |
| `src/schedule.py` | attack-burst scheduler (878 bursts at site A) and benign-event scheduler |
| `src/traffic.py` | MQTT/CoAP message generation: periodic publishers, RFC 7252 back-off/duplicates, TTL/RTT baselines, clock skew, attacks (flood, amplification, MITM, replay, recon) and FDIA variants V1 bias / V2 ramp / V3 balance-preserving |
| `src/simulate.py` | end-to-end simulation → one row per 30 s window (labels + 68 features) |
| `src/features.py` | the 68 features in eight groups (45 network, 21 energy, 2 context) with commissioning-interval calibration |
| `src/common.py` | `Data` (chronological / random / grouped splits with purge), model wrappers (LR, RF, XGBoost, LightGBM, MLP, CNN–LSTM, lagged XGBoost), metrics, block bootstrap, ECE |
| `src/generate_data.py` | writes `data/siteA.pkl`, `data/siteB.pkl`, `data/bursts.pkl` |
| `src/tune.py` | random-search hyper-parameter tuning on the validation days → `results/tuning.json` |
| `src/exp_main.py` | 7 models × 5 seeds on the chronological test block → `main.json`, `main_probs.npz` |
| `src/exp_stats.py` | Friedman–Nemenyi, paired block-bootstrap with Holm correction, per-class/variant/burst-level analyses → `stats.json` |
| `src/exp_protocols.py` | random vs burst-grouped vs chronological protocols (leakage) → `protocols.json` |
| `src/exp_ablation.py` | feature-group ablation and class-imbalance strategies → `ablation.json`, `imbalance.json` |
| `src/exp_robust.py` | unseen FDIA variant, attack-intensity evasion, cross-site transfer and few-shot adaptation → `robust.json` |
| `src/exp_xai.py` | explanation faithfulness, stability and agreement → `xai.json`, `xai_arrays.npz` |
| `src/exp_latency.py`, `exp_latency2.py` | per-window latency and throughput of detectors and explainers → `latency*.json` |
| `src/exp_fdia_auc.py`, `diag_siteb.py` | single-feature FDIA separability; site-B false-alarm diagnosis |
| `src/figures_a.py` … `figures_d.py`, `figstyle.py` | all figures (results plots and conceptual diagrams); `make_figures.sh` runs them |
| `data/` | created by `generate_data.py`: windows of both sites (`siteA.pkl`, `siteB.pkl`, `pandas.read_pickle`) and the burst schedules (`bursts.pkl`, shipped) |
| `results/` | all result JSON/NPZ files and experiment logs |
| `figs/` | figures at 300 dpi |

Intermediate files that are regenerated on demand and therefore not shipped: `data/siteA_scale*.pkl` (made by
`exp_robust.py`) and `data/sim_cache.pkl` (made by `figures_b.py`).

## Data schema
Each row of `siteA.pkl` / `siteB.pkl` is a 30 s window: `win`, `day`, `label` (0 Normal, 1 FDIA, 2 Flood, 3 Amplification,
4 MITM, 5 Replay, 6 Recon), `overlap` (fraction of the window covered by the labelled burst), `burst` (burst id),
`scope` (transport), `variant` (FDIA variant), `benign` (bitmask of benign events) and the 68 feature columns listed in
`common.GROUPS`.

## Protocol
Days 0–2 are clean commissioning (calibration only), 3–19 train, 20–22 validation, 23–29 test, with a 24-window purge gap
at each boundary. A window is labelled by the burst that covers at least half of it. Contrast protocols (random 70/10/20
split, burst-grouped 5-fold) are in `exp_protocols.py`.

## Headline results (site A, chronological test block, mean of 5 seeds)
| Model | Macro-F1 |
|---|---|
| XGBoost, 5-window lag | 0.913 (95 % block-bootstrap CI 0.893–0.927), MCC 0.907, accuracy 0.978, macro-AUC 0.974 |
| XGBoost / LightGBM / Random forest | 0.907 / 0.906 / 0.896 |
| CNN–LSTM / MLP / Logistic regression | 0.888 / 0.883 / 0.747 |

FDIA is the hard class (F1 0.609): recall 80.7 % (V1 bias), 64.1 % (V2 ramp), 0 % (V3 balance-preserving).
TreeSHAP: deletion AUC 0.22 (random 0.51), top-5 stability 0.89; XGBoost p95 latency 0.49 ms, TreeSHAP ≈ 1.25 ms.

## Notes and limitations
- Everything is simulation-based; results should not be read as field performance.
- The random forest with SMOTE + class weights was not run (XGBoost only).
- Latency numbers are single-process timings on 2 virtual CPU cores; run `exp_latency*.py` on an idle machine.
- Neural-network and bootstrap results depend on library versions; exact bit-for-bit reproduction of model results
  beyond `stats.json` was not verified across machines.
