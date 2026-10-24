"""Attack campaigns and benign anomalies (ground truth of the simulation).

Attack campaigns are mutually exclusive in time (one active campaign at a time,
with a 60 s guard) so that every 30 s window has an unambiguous dominant label.
Benign anomalies (sensor faults, link outages with back-log flushes, route
changes, client reconnects, administrator discovery) are labelled *Normal* and
act as hard negatives.
"""
import numpy as np
from config import *

ATTACK_CLASSES = ["FDIA", "Flood", "Amplification", "MITM", "Replay", "Recon"]
MQ, CO = 0, 1
MQ_NODES = [i for i, s in enumerate(SENSORS) if s[1] == MQ]
CO_NODES = [i for i, s in enumerate(SENSORS) if s[1] == CO]
GEN_NODES = [SENSOR_IDX[n] for n in ("pv1", "pv2", "pv3", "w1", "w2", "dg", "bat")]
LOAD_NODES = [SENSOR_IDX["load1"], SENSOR_IDX["load2"]]
FDIA_VARIANTS = ["V1-bias", "V2-ramp", "V3-balanced"]


def _logu(rng, lo, hi):
    return float(np.exp(rng.uniform(np.log(lo), np.log(hi))))


def make_attack_schedule(cfg: SiteConfig, rng, scale: dict | None = None):
    """Return list of burst dicts sorted by start time.

    `scale` (optional) multiplies attack *intensity* for the problem-space
    evasion study: keys 'rate' (flood/recon/amp rates), 'bias' (FDIA/MITM
    magnitude), 'frac' (MITM tamper fraction / replay fraction).
    """
    scale = scale or {}
    sr, sb, sf = scale.get("rate", 1.0), scale.get("bias", 1.0), scale.get("frac", 1.0)
    t_lo = cfg.n_commission_days * DAY
    t_hi = cfg.n_days * DAY
    n_act = cfg.n_days - cfg.n_commission_days
    cand = []
    for c in ATTACK_CLASSES:
        n = rng.poisson(cfg.burst_rate * n_act)
        for _ in range(n):
            dur = rng.uniform(cfg.burst_min, cfg.burst_max)
            b = dict(cls=c, dur=dur, params={})
            p = b["params"]
            if c == "FDIA":
                v = rng.choice(3, p=[0.4, 0.35, 0.25])
                p["variant"] = FDIA_VARIANTS[v]
                if v < 2:
                    k = rng.choice([1, 2], p=[0.7, 0.3])
                    p["sensors"] = [int(x) for x in rng.choice(N_SENS, size=k, replace=False)]
                    p["gamma"] = (rng.uniform(0.10, 0.35) if v == 0 else rng.uniform(0.20, 0.50)) * sb
                    p["sign"] = [int(rng.choice([-1, 1])) for _ in p["sensors"]]
                else:
                    gi = int(rng.choice(GEN_NODES))
                    lj = int(rng.choice(LOAD_NODES))
                    p["sensors"] = [gi, lj]
                    p["gamma"] = rng.uniform(0.10, 0.35) * sb
                    p["sign"] = [int(rng.choice([-1, 1]))]
                b["scope"] = "both"
            elif c == "Flood":
                tr = int(rng.choice([MQ, CO]))
                p["tr"] = tr
                p["rate"] = _logu(rng, 4, 300) * sr
                p["n_src"] = int(rng.integers(1, 7))
                p["mode"] = int(rng.integers(0, 2))      # MQTT: 0 publish / 1 connect ; CoAP: 0 GET / 1 POST
                b["scope"] = "mqtt" if tr == MQ else "coap"
            elif c == "Amplification":
                p["tr"] = CO
                p["pool"] = int(_logu(rng, 50, 2000))
                p["rate"] = _logu(rng, 2, 150) * sr
                p["req"] = float(rng.uniform(40, 70))
                p["resp"] = float(rng.uniform(300, 1100))
                b["scope"] = "coap"
            elif c == "MITM":
                tr = int(rng.choice([MQ, CO]))
                pool = MQ_NODES if tr == MQ else CO_NODES
                k = int(rng.integers(1, min(3, len(pool)) + 1))
                p["tr"] = tr
                p["nodes"] = [int(x) for x in rng.choice(pool, size=k, replace=False)]
                p["delay"] = float(rng.uniform(0.008, 0.080) * sb)
                p["q"] = float(min(1.0, rng.uniform(0.10, 0.70) * sf))
                p["gamma"] = float(rng.uniform(0.05, 0.30) * sb)
                p["p_drop"] = float(rng.uniform(0, 0.03))
                p["p_dup"] = float(rng.uniform(0, 0.02))
                b["scope"] = "mqtt" if tr == MQ else "coap"
            elif c == "Replay":
                tr = int(rng.choice([MQ, CO]))
                pool = MQ_NODES if tr == MQ else CO_NODES
                k = int(rng.integers(1, min(3, len(pool)) + 1))
                p["tr"] = tr
                p["nodes"] = [int(x) for x in rng.choice(pool, size=k, replace=False)]
                p["frac"] = float(min(1.0, rng.uniform(0.3, 1.0) * sf))
                p["lag"] = _logu(rng, 60, 43200)
                b["scope"] = "mqtt" if tr == MQ else "coap"
            elif c == "Recon":
                tr = int(rng.choice([MQ, CO]))
                p["tr"] = tr
                p["rate"] = _logu(rng, 0.5, 15) * sr
                b["scope"] = "mqtt" if tr == MQ else "coap"
            cand.append(b)
    order = rng.permutation(len(cand))
    placed = []
    for idx in order:
        b = cand[idx]
        for _ in range(60):
            t0 = rng.uniform(t_lo, t_hi - b["dur"] - 1)
            t1 = t0 + b["dur"]
            if all(t1 + 60 < q["t0"] or t0 > q["t1"] + 60 for q in placed):
                b["t0"], b["t1"] = t0, t1
                placed.append(b)
                break
    placed.sort(key=lambda b: b["t0"])
    for i, b in enumerate(placed):
        b["id"] = i
    return placed


def make_benign_schedule(cfg: SiteConfig, rng):
    """Benign anomalies: dict of lists of events (start, end, node/params)."""
    T = cfg.n_days * DAY
    ev = dict(outage=[], route=[], reconnect=[], admin=[], fault=[])
    for i in range(N_SENS):
        for _ in range(rng.poisson(cfg.r_outage * cfg.n_days)):
            s = rng.uniform(0, T - 700)
            ev["outage"].append(dict(node=i, t0=s, t1=s + rng.uniform(60, 600)))
        for _ in range(rng.poisson(cfg.r_route * cfg.n_days)):
            s = rng.uniform(0, T - 22000)
            ev["route"].append(dict(node=i, t0=s, t1=s + rng.uniform(1800, 21600),
                                    dttl=int(rng.choice([-1, 1])), rtt_mult=float(rng.uniform(1.1, 1.4))))
        for _ in range(rng.poisson(cfg.r_fault * cfg.n_days)):
            s = rng.uniform(0, T - 1300)
            kind = str(rng.choice(["stuck", "spike", "drift"]))
            ev["fault"].append(dict(node=i, t0=s, t1=s + rng.uniform(120, 1200), kind=kind,
                                    mag=float(rng.uniform(0.05, 0.2)), sgn=int(rng.choice([-1, 1]))))
    for i in MQ_NODES:
        for _ in range(rng.poisson(cfg.r_reconnect * cfg.n_days)):
            ev["reconnect"].append(dict(node=i, t0=float(rng.uniform(0, T - 10)), t1=0.0))
    for _ in range(rng.poisson(cfg.r_admin_wild * cfg.n_days)):
        s = rng.uniform(0, T - 100)
        ev["admin"].append(dict(t0=s, t1=s + rng.uniform(10, 60), n_get=int(rng.integers(3, 11))))
    for k in ev:
        for e in ev[k]:
            if e["t1"] == 0.0:
                e["t1"] = e["t0"] + 5.0
    return ev
