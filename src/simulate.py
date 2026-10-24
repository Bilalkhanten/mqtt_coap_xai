"""Driver: simulate a site, extract window features, attach ground-truth labels."""
import dataclasses, sys, time
from collections import defaultdict
import numpy as np
import pandas as pd
from config import *
from schedule import *
from traffic import *
from features import *
from plant import simulate_plant

SPD = STEPS_PER_DAY


def simulate(cfg: SiteConfig, scale=None, only_days=None, verbose=True, return_msgs_day=None):
    t00 = time.time()
    rs = lambda *a: np.random.default_rng([cfg.seed, *a])
    truth, meas, aux = simulate_plant(cfg)
    nodes = make_nodes(cfg, rs(0))
    ben = make_benign_schedule(cfg, rs(0, 1))
    bursts = make_attack_schedule(cfg, rs(1), scale)
    O, bmask, kmap = build_observed(meas, cfg, nodes, ben, bursts, rs(3))
    nc = cfg.n_commission_days
    em = EnergyModel().fit(O[: nc * SPD], aux["ghi_cs"][: nc * SPD], cfg)
    em.ghi_cs = aux["ghi_cs"]
    EF = em.window_features(O)
    if verbose:
        print(f"[{cfg.name}] plant+energy {time.time()-t00:.0f}s, bursts={len(bursts)}", flush=True)
    # attack message buckets (per day)
    buckets = defaultdict(list)
    for b in bursts:
        M = burst_msgs(b, cfg, nodes, rs(4, b["id"]))
        if M is None or len(M["t"]) == 0:
            continue
        dd = np.floor(M["t"] / DAY).astype(int)
        for d in np.unique(dd):
            m = dd == d
            buckets[int(d)].append({k: v[m] for k, v in M.items()})
    base_ttl = base_ttl_array(nodes)
    days = list(range(cfg.n_days)) if only_days is None else sorted(set(range(nc)) | set(only_days))
    comm_msgs, base_skew = {}, None
    rows = {}
    keep_msgs = {}

    def feats_for_day(d, M, base_skew):
        t0 = d * DAY
        F = {}
        F.update(tr_features(M, 0, t0, WINS_PER_DAY, base_ttl, base_skew))
        F.update(tr_features(M, 1, t0, WINS_PER_DAY, base_ttl, base_skew))
        return F

    for d in days:
        rng = rs(2, d)
        p_loss = cfg.loss_base[1] * float(np.exp(0.4 * rng.standard_normal()))
        bday = [b for b in bursts if b["t1"] > d * DAY and b["t0"] < (d + 1) * DAY]
        parts = [node_day(i, d, cfg, nodes, ben, bday, rng, p_loss) for i in range(N_SENS)]
        parts.append(control_day(d, cfg, nodes, ben, rng, p_loss))
        parts += buckets.get(d, [])
        M = cat(parts)
        if return_msgs_day is not None and d in return_msgs_day:
            keep_msgs[d] = M
        if d < nc:
            comm_msgs[d] = M
            if d == nc - 1:
                base_skew = calibrate_network(cat(list(comm_msgs.values())))
                for dd_, MM in comm_msgs.items():
                    rows[dd_] = feats_for_day(dd_, MM, base_skew)
        else:
            rows[d] = feats_for_day(d, M, base_skew)
        if verbose and d % 5 == 0:
            print(f"  day {d} done  msgs={len(M['t']):,}  t={time.time()-t00:.0f}s", flush=True)
    # ---- assemble windows
    W = cfg.n_days * WINS_PER_DAY
    wstart = np.arange(W) * WIN
    label = np.zeros(W, dtype=np.int8)
    frac = np.zeros(W)
    bid = np.full(W, -1, dtype=np.int32)
    scope = np.array([""] * W, dtype=object)
    variant = np.array([""] * W, dtype=object)
    for b in bursts:
        w0, w1 = int(b["t0"] // WIN), int(b["t1"] // WIN)
        ww = np.arange(w0, min(w1 + 1, W))
        ov = np.clip(np.minimum(wstart[ww] + WIN, b["t1"]) - np.maximum(wstart[ww], b["t0"]), 0, None) / WIN
        upd = ov > frac[ww]
        sel = ww[upd]
        frac[sel] = ov[upd]
        label[sel] = CLS[b["cls"]] if True else 0
        bid[sel] = b["id"]
        scope[sel] = b["scope"]
        variant[sel] = b["params"].get("variant", "")
    label = np.where(frac >= 0.5, label, 0).astype(np.int8)
    bm = np.bitwise_or.reduce(bmask[: W * STEPS_PER_WIN].reshape(W, STEPS_PER_WIN), axis=1)
    df = pd.DataFrame(dict(win=np.arange(W), day=np.arange(W) // WINS_PER_DAY, label=label, overlap=frac,
                           burst=bid, scope=scope, variant=variant, benign=bm))
    keys = list(next(iter(rows.values())).keys())
    for k in keys:
        col = np.full(W, np.nan)
        for d, F in rows.items():
            col[d * WINS_PER_DAY:(d + 1) * WINS_PER_DAY] = F[k]
        df[k] = col
    for k, v in EF.items():
        df[k] = v[:W]
    hh = (wstart % DAY) / 3600.0
    df["tod_sin"] = np.sin(2 * np.pi * hh / 24)
    df["tod_cos"] = np.cos(2 * np.pi * hh / 24)
    if only_days is not None:
        df = df[df.day.isin(days)].reset_index(drop=True)
    meta = dict(bursts=bursts, ben=ben, cfg=cfg, nodes=nodes, em=em, base_skew=base_skew, base_ttl=base_ttl)
    if verbose:
        print(f"[{cfg.name}] done in {time.time()-t00:.0f}s, windows={len(df):,}", flush=True)
    if return_msgs_day is not None:
        meta["msgs"] = keep_msgs
    return df, meta, (truth, meas, O, aux)


if __name__ == "__main__":
    cfg = dataclasses.replace(SITE_A, n_days=int(sys.argv[1]) if len(sys.argv) > 1 else 5)
    df, meta, _ = simulate(cfg)
    print(df.label.value_counts().sort_index())
    print(df.shape)
