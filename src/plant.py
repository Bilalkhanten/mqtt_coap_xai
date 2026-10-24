"""Physics-based hybrid PV / wind / diesel / battery microgrid (5 s resolution).

Models
------
* PV      : Haurwitz clear-sky GHI x stochastic clearness index (OU process with a
            daily regime) x temperature derating (NOCT cell model).
* Wind    : Gaussian-copula AR(1) process mapped to a Weibull marginal, cubic
            power curve with cut-in / rated / cut-out speeds.
* Load    : two feeders with double-peak residential and working-hours commercial
            profiles, AR(1) multiplicative noise, weekday/weekend factor.
* Storage : energy-limited battery with round-trip efficiency, SOC-based dispatch.
* Diesel  : minimum-stable-load genset dispatched with hysteresis.

The returned `truth` matrix holds the *true* electrical quantities; `meas` adds
sensor noise.  An unmeasured distribution loss makes the energy-balance residual
non-trivial, as in a real plant.
"""
import numpy as np
from scipy.stats import norm
from config import *


def _ou(n, tau, sigma, rng, dt=DT):
    """Stationary Ornstein-Uhlenbeck sample path with std `sigma` and time constant `tau`."""
    from scipy.signal import lfilter
    a = np.exp(-dt / tau)
    e = rng.standard_normal(n) * sigma * np.sqrt(1 - a * a)
    y = lfilter([1.0], [1.0, -a], e)
    return y + rng.standard_normal() * sigma * a ** np.arange(n)


def simulate_plant(cfg: SiteConfig):
    rng = np.random.default_rng(cfg.seed)
    n = cfg.n_days * STEPS_PER_DAY
    t = np.arange(n) * DT
    day = (t // DAY).astype(int)
    hour = (t % DAY) / 3600.0
    doy = cfg.start_doy + day

    # ------------------------------------------------------------------ solar
    phi = np.deg2rad(cfg.latitude)
    decl = np.deg2rad(23.45) * np.sin(2 * np.pi * (284 + doy) / 365.0)
    h = np.deg2rad(15.0 * (hour - 12.0))
    cosz = np.sin(phi) * np.sin(decl) + np.cos(phi) * np.cos(decl) * np.cos(h)
    cosz = np.clip(cosz, 0, None)
    ghi_cs = np.where(cosz > 0.02, 1098.0 * cosz * np.exp(-0.059 / np.maximum(cosz, 0.02)), 0.0)

    regime = rng.choice(3, size=cfg.n_days, p=[0.55, 0.30, 0.15])
    kbar_d = np.array([rng.uniform(*[(0.85, 0.98), (0.55, 0.80), (0.20, 0.45)][r]) for r in regime])
    sig_d = np.array([[0.03, 0.18, 0.07][r] for r in regime])
    kbar_t = kbar_d[day]
    sig_t = sig_d[day]
    x_common = _ou(n, 480.0, 1.0, rng)
    x_local = [_ou(n, 300.0, 1.0, rng) for _ in range(3)]
    rho = 0.85
    k_sites = []
    for i in range(3):
        xi = np.sqrt(rho) * x_common + np.sqrt(1 - rho) * x_local[i]
        k_sites.append(np.clip(kbar_t + sig_t * xi, 0.05, 1.05))
    k_pyr = np.clip(kbar_t + sig_t * x_common, 0.05, 1.05)
    t_amb = 30.0 + 8.0 * np.sin(2 * np.pi * (hour - 9.0) / 24.0) + 0.5 * (rng.random(cfg.n_days)[day] - 0.5) * 10
    p_pv_avail = []
    for i in range(3):
        g = ghi_cs * k_sites[i]
        tcell = t_amb + 0.031 * g
        p_pv_avail.append(cfg.pv_rated[i] * (g / 1000.0) * (1 - 0.004 * (tcell - 25.0)) * 0.97)
    g_pyr = ghi_cs * k_pyr

    # ------------------------------------------------------------------- wind
    z_common = _ou(n, 2400.0, 1.0, rng)
    v = []
    for j in range(2):
        zj = np.sqrt(0.9) * z_common + np.sqrt(0.1) * _ou(n, 900.0, 1.0, rng)
        u = np.clip(norm.cdf(zj), 1e-6, 1 - 1e-6)
        vj = cfg.wind_scale_c * (-np.log(1 - u)) ** (1.0 / cfg.wind_shape_k)
        vj *= 1 + 0.15 * np.sin(2 * np.pi * (hour - 15.0) / 24.0)
        vj *= 1 + 0.04 * rng.standard_normal(n)
        v.append(np.clip(vj, 0, None))

    def wind_power(vv, pr):
        cin, vr, cout = 3.0, 12.0, 25.0
        p = np.clip((vv ** 3 - cin ** 3) / (vr ** 3 - cin ** 3), 0, 1) * pr
        return np.where(vv > cout, 0.0, p)

    p_w = [wind_power(v[j], cfg.wind_rated[j]) for j in range(2)]
    v_mast = 0.5 * (v[0] + v[1])

    # ------------------------------------------------------------------- load
    wd = (day % 7)                              # day 0 = Monday
    weekend = (wd >= 5).astype(float)
    shapeA = 0.35 + 0.25 * np.exp(-((hour - 7.5) / 1.8) ** 2) + 0.60 * np.exp(-((hour - 20.0) / 2.2) ** 2)
    shapeA /= 0.35 + 0.60
    shapeB = 0.25 + 0.65 / (1 + np.exp(-(hour - 8.0) * 2.0)) / (1 + np.exp((hour - 17.0) * 2.0))
    shapeB *= (1 - 0.5 * weekend)
    tfac = 1 + 0.015 * np.clip(t_amb - 28.0, 0, None)
    loads = []
    for k, shape in enumerate([shapeA, shapeB]):
        noise = _ou(n, 300.0, 0.04, rng)
        loads.append(cfg.load_peak[k] * shape * tfac * (1 + noise))
    load = np.clip(loads[0] + loads[1], 1.0, None)

    # --------------------------------------------------------------- dispatch
    # Exact energy-balance dispatch:  c*(PV+W) + P_dg + P_bat = (1+loss)*load
    cap = cfg.bat_energy
    E = 0.6 * cap
    dg_on = False
    low_cnt = 0
    dt_h = DT / 3600.0
    eta = 0.96
    P_dg = np.zeros(n)
    P_bat = np.zeros(n)           # +discharge / -charge
    curt = np.ones(n)             # renewable curtailment factor
    soc_tr = np.zeros(n)
    gen_pv = p_pv_avail[0] + p_pv_avail[1] + p_pv_avail[2]
    gen_w = p_w[0] + p_w[1]
    gen = gen_pv + gen_w
    ld_tot = load * (1 + cfg.loss_frac)
    for i in range(n):
        net = ld_tot[i] - gen[i]
        soc = E / cap
        if dg_on:
            low_cnt = low_cnt + 1 if (soc > 0.5 and net < 0.5 * cfg.bat_power) else 0
            if low_cnt > 120:
                dg_on = False
                low_cnt = 0
        elif net > cfg.bat_power or soc < 0.22:
            dg_on = True
            low_cnt = 0
        pdg = 0.0
        if dg_on:
            target_ch = 15.0 if soc < 0.5 else 0.0
            pdg = float(np.clip(net + target_ch, cfg.dg_min, cfg.dg_rated))
        r = net - pdg
        ch_max = max(min(cfg.bat_power, (0.95 * cap - E) / (eta * dt_h)), 0.0)
        dis_max = max(min(cfg.bat_power, (E - 0.20 * cap) * eta / dt_h), 0.0)
        pb = float(np.clip(r, -ch_max, dis_max))
        left = r - pb
        c = 1.0
        if left > 0:                                   # unmet deficit -> more diesel
            add = min(left, cfg.dg_rated - pdg) if dg_on else min(max(left, cfg.dg_min), cfg.dg_rated)
            pdg += add
            left -= add
            if not dg_on and pdg > 0:
                dg_on = True
        elif left < 0:                                 # excess supply -> curtail renewables first
            ex = -left
            if gen[i] > 1e-9:
                c = max(0.0, 1 - ex / gen[i])
                ex -= (1 - c) * gen[i]
            if ex > 1e-9:
                pdg = max(pdg - ex, 0.0)
        if pb >= 0:
            E -= pb / eta * dt_h
        else:
            E += eta * (-pb) * dt_h
        P_dg[i] = pdg
        P_bat[i] = pb
        curt[i] = c
        soc_tr[i] = E / cap
    pv_out = [p * curt for p in p_pv_avail]
    p_w = [p * curt for p in p_w]

    truth = np.stack(pv_out + p_w + [P_dg, P_bat, loads[0], loads[1], g_pyr, v_mast], axis=1)
    # --- measurement noise
    rat = np.array(list(cfg.pv_rated) + list(cfg.wind_rated) + [cfg.dg_rated, cfg.bat_power] +
                   list(cfg.load_peak) + [1000.0, 12.0])
    sd = cfg.sensor_noise_rel * rat[None, :] + cfg.sensor_noise_prop * np.abs(truth)
    sd[:, 9] = 4.0 + 0.01 * np.abs(truth[:, 9])        # pyranometer [W/m^2]
    sd[:, 10] = 0.15                                   # anemometer [m/s]
    meas = truth + rng.standard_normal(truth.shape) * sd
    meas[:, 9] = np.clip(meas[:, 9], 0, None)
    meas[:, 10] = np.clip(meas[:, 10], 0, None)
    aux = dict(ghi_cs=ghi_cs, soc=soc_tr, regime=regime, rated=rat, t_amb=t_amb)
    return truth, meas, aux
