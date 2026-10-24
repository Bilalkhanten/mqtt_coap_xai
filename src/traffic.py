"""Message-level MQTT/CoAP traffic generator with benign anomalies and attacks.

Every message is a row in a struct-of-arrays:
  t    arrival time at the gateway [s]
  src  source id (0-10 field nodes, 50/51 administrator tools, 60 control centre,
       >=1e5 unknown/attacker addresses)
  tr   transport (0 = MQTT/TCP, 1 = CoAP/UDP)
  kind 0 PUBLISH/telemetry, 1 CONNECT, 2 SUBSCRIBE, 3 GET, 4 POST
  topic MQTT topic id / CoAP resource-path id
  qos  MQTT QoS level (CoAP: 0)
  sin / sout  request / response bytes seen at the gateway
  seq  application sequence number (-1 if not applicable)
  flags bit mask  GAP | DUP | NF | RST | CON | WILD
  rtt  gateway-measured round-trip time [s] (NaN if none)
  ttl  IP TTL / hop-limit as received
  skew arrival minus device time-stamp [s] (NaN if none)
  sport source port
"""
import numpy as np
from config import *
from schedule import *

GAP, DUP, NF, RST, CON, WILD = 1, 2, 4, 8, 16, 32
FIELDS = dict(t="f8", src="i4", tr="i1", kind="i1", topic="i2", qos="i1", sin="i2", sout="i2",
              seq="i4", flags="i2", rtt="f4", ttl="i2", skew="f4", sport="i4")
KNOWN_SRC = list(range(N_SENS)) + [50, 51, 60]
PATH_CORE = 99


def empty(n=0):
    return {k: np.empty(n, dtype=v) for k, v in FIELDS.items()}


def cat(parts):
    parts = [p for p in parts if p is not None and len(p["t"])]
    if not parts:
        return empty(0)
    return {k: np.concatenate([p[k] for p in parts]) for k in FIELDS}


def pack(n, **kw):
    m = empty(n)
    defaults = dict(src=-1, tr=0, kind=0, topic=-1, qos=0, sin=0, sout=0, seq=-1, flags=0,
                    rtt=np.nan, ttl=64, skew=np.nan, sport=0)
    for k, v in defaults.items():
        m[k][:] = kw.get(k, v)
    m["t"][:] = kw["t"]
    return m


def u_hash(k, node, bid):
    """Deterministic pseudo-uniform in [0,1) from (message index, node, burst id)."""
    x = np.sin(np.asarray(k, float) * 12.9898 + node * 78.233 + bid * 37.719 + 0.5) * 43758.5453
    return x - np.floor(x)


def make_nodes(cfg, rng):
    tr = SENS_TRANSPORT
    return dict(
        tr=tr, T=PERIOD.copy(),
        phase=rng.uniform(0, PERIOD),
        clock=rng.normal(0, cfg.clock_offset_sd, N_SENS),
        ttl=np.where(tr == 0, 64 - rng.integers(6, 17, N_SENS), 64 - rng.integers(8, 19, N_SENS)).astype(int),
        sport=rng.integers(20000, 60000, N_SENS),
        rtt_mult=np.exp(rng.normal(0, 0.15, N_SENS)),
        ttl_known={50: 58, 51: 57, 60: 60},
    )


def base_ttl_array(nodes):
    arr = np.full(200, -1, dtype=int)
    for i in range(N_SENS):
        arr[i] = nodes["ttl"][i]
    for k, v in nodes["ttl_known"].items():
        arr[k] = v
    return arr


# ----------------------------------------------------------------------------
# legitimate telemetry (one sensor, one day)
# ----------------------------------------------------------------------------
def node_day(i, d, cfg, nodes, ben, bursts, rng, p_loss):
    T = nodes["T"][i]
    tr = int(nodes["tr"][i])
    ph = nodes["phase"][i]
    k0 = int(np.ceil((d * DAY - ph) / T))
    k1 = int(np.ceil(((d + 1) * DAY - ph) / T))
    ks = np.arange(k0, k1)
    n = len(ks)
    ts = ks * T + ph + rng.normal(0, 0.08, n)
    rttm = np.full(n, nodes["rtt_mult"][i])
    dttl = np.zeros(n, dtype=int)
    for e in ben["route"]:
        if e["node"] == i and e["t1"] > d * DAY and e["t0"] < (d + 1) * DAY:
            m = (ts >= e["t0"]) & (ts < e["t1"])
            dttl[m] = e["dttl"]
            rttm[m] *= e["rtt_mult"]
    rtt = cfg.rtt_med[tr] * rttm * np.exp(cfg.rtt_sig[tr] * rng.standard_normal(n))
    owd = rtt / 2
    arr = ts + owd
    flags = np.zeros(n, dtype=np.int16)
    qos = np.zeros(n, dtype=np.int8)
    lost = np.zeros(n, dtype=bool)
    extra_rows = []
    if tr == 0:
        qos = (rng.random(n) < 0.8).astype(np.int8)
        sin = (2 + 28 + np.round(rng.normal(64, 6, n))).astype(np.int16)
        sout = (4 * qos).astype(np.int16)
        rtt_f = np.where(qos > 0, rtt, np.nan).astype(np.float32)
        dup_m = (rng.random(n) < 0.002) & (qos > 0)
    else:
        con = rng.random(n) < cfg.con_share
        flags |= (con * CON).astype(np.int16)
        f1 = rng.random(n) < p_loss
        f2 = f1 & (rng.random(n) < p_loss)
        f3 = f2 & (rng.random(n) < p_loss)
        back = (f1 * 2.0 * rng.uniform(1, 1.5, n) + f2 * 4.0 * rng.uniform(1, 1.5, n) + f3 * 8.0 * rng.uniform(1, 1.5, n))
        back = np.where(con, back, 0.0)
        lost = (~con) & f1
        arr = arr + back
        sin = (4 + 8 + np.round(rng.normal(26, 5, n))).astype(np.int16)
        sout = np.where(con, 6, 0).astype(np.int16)
        rtt_f = np.where(con, rtt, np.nan).astype(np.float32)
        dup_m = con & (rng.random(n) < p_loss)          # ACK lost -> duplicate request arrives later
    skew = (arr - ts - nodes["clock"][i]).astype(np.float32)
    ttl = (nodes["ttl"][i] + dttl).astype(np.int16)
    # --- benign outage: buffer, then flush at 10x rate with old device time-stamps
    for e in ben["outage"]:
        if e["node"] == i and e["t1"] > d * DAY and e["t0"] < (d + 1) * DAY:
            m = (ts >= e["t0"]) & (ts < e["t1"])
            cnt = int(m.sum())
            if cnt:
                arr[m] = e["t1"] + np.arange(cnt) * (T / 10.0) + owd[m]
                skew[m] = (arr[m] - ts[m] - nodes["clock"][i]).astype(np.float32)
    # --- attacks that act on legitimate messages
    for b in bursts:
        p = b["params"]
        if b["cls"] == "MITM" and i in p["nodes"]:
            m = (ts >= b["t0"]) & (ts < b["t1"])
            if m.any():
                dl = p["delay"] * (1 + 0.2 * rng.standard_normal(int(m.sum())))
                arr[m] += dl
                skew[m] += dl.astype(np.float32)
                rtt_f[m] += (2 * dl).astype(np.float32)
                ttl[m] -= 1
                tam = np.zeros(n, dtype=bool)
                tam[m] = u_hash(ks[m], i, b["id"]) < p["q"]
                sin[tam] += rng.integers(-4, 7, int(tam.sum())).astype(np.int16)
                drop = m & (rng.random(n) < p["p_drop"])
                lost |= drop
                dupm = m & (~drop) & (rng.random(n) < p["p_dup"])
                dup_m |= dupm
        if b["cls"] == "Flood" and p["tr"] == tr and p["rate"] > 40:
            m = (ts >= b["t0"]) & (ts < b["t1"])
            if m.any():
                ov = min(p["rate"] / 300.0, 1.0)
                ex = ov * rng.uniform(0.02, 0.4, int(m.sum()))
                arr[m] += ex
                skew[m] += ex.astype(np.float32)
                rtt_f[m] += (2 * ex).astype(np.float32)
                lost |= m & (rng.random(n) < 0.3 * ov)
    gap = np.zeros(n, dtype=bool)
    gap[1:] = lost[:-1]
    flags = flags | (gap * GAP).astype(np.int16)
    keep = ~lost
    main = pack(int(keep.sum()), t=arr[keep], src=i, tr=tr, kind=0, topic=(i if tr == 0 else i - 7), qos=qos[keep],
                sin=sin[keep], sout=sout[keep], seq=ks[keep], flags=flags[keep], rtt=rtt_f[keep], ttl=ttl[keep],
                skew=skew[keep], sport=nodes["sport"][i])
    # duplicates (retransmissions) arrive 1-3 s later (MQTT DUP / CoAP duplicate MID)
    dm = dup_m & keep
    if dm.any():
        nd = int(dm.sum())
        dd = pack(nd, t=arr[dm] + rng.uniform(1, 3, nd), src=i, tr=tr, kind=0, topic=(i if tr == 0 else i - 7),
                  qos=qos[dm], sin=sin[dm], sout=sout[dm], seq=ks[dm], flags=(flags[dm] | DUP) & ~GAP,
                  ttl=ttl[dm], skew=skew[dm] + 2.0, sport=nodes["sport"][i])
        return cat([main, dd])
    return main


def control_day(d, cfg, nodes, ben, rng, p_loss):
    parts = []
    lo, hi = d * DAY, (d + 1) * DAY
    # control-centre set-points over MQTT
    n = rng.poisson(DAY / 45.0)
    t = rng.uniform(lo, hi, n)
    rtt = cfg.rtt_med[0] * np.exp(cfg.rtt_sig[0] * rng.standard_normal(n))
    parts.append(pack(n, t=t, src=60, tr=0, kind=0, topic=rng.integers(7, 9, n), qos=1,
                      sin=(30 + np.round(rng.normal(40, 5, n))).astype(np.int16), sout=4, ttl=60, rtt=rtt, sport=40000))
    # client reconnects: CONNECT + SUBSCRIBE
    for e in ben["reconnect"]:
        if lo <= e["t0"] < hi:
            i = e["node"]
            parts.append(pack(1, t=e["t0"], src=i, tr=0, kind=1, sin=68, sout=4, ttl=nodes["ttl"][i], sport=nodes["sport"][i]))
            parts.append(pack(1, t=e["t0"] + 0.15, src=i, tr=0, kind=2, topic=7, qos=1, sin=40, sout=5, ttl=nodes["ttl"][i],
                              sport=nodes["sport"][i]))
    # administrator discovery (hard negative for Recon)
    for e in ben["admin"]:
        if lo <= e["t0"] < hi:
            parts.append(pack(1, t=e["t0"], src=50, tr=0, kind=2, topic=-1, qos=0, sin=24, sout=6, ttl=58, flags=WILD, sport=41000))
            ng = e["n_get"]
            tg = np.sort(rng.uniform(e["t0"], e["t1"], ng))
            nf = rng.random(ng) < 0.3
            parts.append(pack(ng, t=tg, src=51, tr=1, kind=3, topic=np.where(nf, rng.integers(5, 30, ng), PATH_CORE),
                              sin=25, sout=np.where(nf, 5, rng.integers(150, 400, ng)), ttl=57,
                              flags=(nf * NF).astype(np.int16) | CON, rtt=cfg.rtt_med[1] * np.exp(0.3 * rng.standard_normal(ng)),
                              sport=rng.integers(30000, 60000)))
    return cat(parts)


# ----------------------------------------------------------------------------
# attack injections (whole burst at once, split per day by caller)
# ----------------------------------------------------------------------------
def burst_msgs(b, cfg, nodes, rng):
    p = b["params"]
    dur = b["t1"] - b["t0"]
    c = b["cls"]
    bid = b["id"]
    if c == "Flood":
        n = rng.poisson(p["rate"] * dur)
        if n == 0:
            return None
        t = np.sort(rng.uniform(b["t0"], b["t1"], n))
        ns = p["n_src"]
        srcs = 100000 + bid * 10 + np.arange(ns)
        sidx = rng.integers(0, ns, n)
        sport = rng.integers(1024, 65000, ns)[sidx]
        ttl = (rng.integers(48, 62, ns))[sidx]
        if p["tr"] == 0:
            if p["mode"] == 0:
                topic = np.where(rng.random(n) < 0.5, rng.integers(0, 9, n), rng.integers(9, 500, n))
                return pack(n, t=t, src=srcs[sidx], tr=0, kind=0, topic=topic, qos=0,
                            sin=(30 + rng.uniform(60, 1200, n)).astype(np.int16), sout=0, ttl=ttl, sport=sport)
            return pack(n, t=t, src=srcs[sidx], tr=0, kind=1, sin=rng.integers(60, 91, n).astype(np.int16), sout=4,
                        ttl=ttl, sport=sport)
        kind = 3 if p["mode"] == 0 else 4
        con = rng.random(n) < 0.2
        nf = rng.random(n) < 0.4
        sin = rng.integers(20, 61, n) if kind == 3 else rng.integers(60, 401, n)
        return pack(n, t=t, src=srcs[sidx], tr=1, kind=kind, topic=np.where(nf, rng.integers(5, 60, n), rng.integers(0, 4, n)),
                    sin=sin.astype(np.int16), sout=np.where(con, rng.integers(6, 21, n), 0).astype(np.int16),
                    flags=(nf * NF).astype(np.int16) | (con * CON).astype(np.int16), ttl=ttl, sport=sport)
    if c == "Amplification":
        n = rng.poisson(p["rate"] * dur)
        if n == 0:
            return None
        t = np.sort(rng.uniform(b["t0"], b["t1"], n))
        src = 200000 + bid * 2500 + rng.integers(0, p["pool"], n)
        sin = np.clip(rng.normal(p["req"], 5, n), 20, 120).astype(np.int16)
        sout = np.clip(rng.normal(p["resp"], 40, n), 100, 1400).astype(np.int16)
        return pack(n, t=t, src=src, tr=1, kind=3, topic=PATH_CORE, sin=sin, sout=sout, ttl=rng.integers(30, 128, n),
                    sport=int(rng.integers(1024, 65000)))
    if c == "Recon":
        n = rng.poisson(p["rate"] * dur)
        if n == 0:
            return None
        t = np.sort(rng.uniform(b["t0"], b["t1"], n))
        src = 300000 + bid
        ttl = int(rng.integers(48, 62))
        if p["tr"] == 0:
            u = rng.random(n)
            kind = np.where(u < 0.4, 2, np.where(u < 0.7, 1, 0)).astype(np.int8)
            wild = (kind == 2) & (rng.random(n) < 0.5)
            return pack(n, t=t, src=src, tr=0, kind=kind, topic=np.where(wild, -1, rng.integers(0, 500, n)),
                        qos=0, sin=rng.integers(20, 81, n).astype(np.int16), sout=rng.integers(4, 11, n).astype(np.int16),
                        flags=(wild * WILD).astype(np.int16), ttl=ttl, sport=int(rng.integers(1024, 65000)))
        core = rng.random(n) < 0.2
        nf = (~core) & (rng.random(n) < 0.85)
        sport = np.where(rng.random(n) < 0.5, rng.integers(1024, 65000, n), 45555)
        return pack(n, t=t, src=src, tr=1, kind=3, topic=np.where(core, PATH_CORE, rng.integers(0, 300, n)),
                    sin=rng.integers(20, 41, n).astype(np.int16),
                    sout=np.where(core, rng.integers(100, 401, n), np.where(nf, rng.integers(4, 9, n), rng.integers(10, 40, n))).astype(np.int16),
                    flags=(nf * NF).astype(np.int16), ttl=ttl, sport=sport)
    if c == "Replay":
        parts = []
        for i in p["nodes"]:
            T, ph = nodes["T"][i], nodes["phase"][i]
            k_lo, k_hi = int(np.ceil((b["t0"] - ph) / T)), int(np.floor((b["t1"] - ph) / T))
            if k_hi <= k_lo:
                continue
            ks = np.arange(k_lo, k_hi)
            m = u_hash(ks, i, bid) < p["frac"]
            ks = ks[m]
            n = len(ks)
            if n == 0:
                continue
            t = ks * T + ph + rng.uniform(0.3, 0.9, n) * T
            dttl = rng.choice([-2, -1, 0, 1, 2], size=n, p=[0.2, 0.25, 0.1, 0.25, 0.2])
            parts.append(pack(n, t=t, src=i, tr=p["tr"], kind=0, topic=(i if p["tr"] == 0 else i - 7), qos=1 if p["tr"] == 0 else 0,
                              sin=(2 + 28 + np.round(rng.normal(64, 6, n)) if p["tr"] == 0 else 12 + np.round(rng.normal(26, 5, n))).astype(np.int16),
                              sout=4 if p["tr"] == 0 else 0, seq=(ks - int(p["lag"] / T)), flags=DUP,
                              ttl=(nodes["ttl"][i] + dttl).astype(np.int16),
                              skew=(p["lag"] + t - (ks * T + ph) - nodes["clock"][i]).astype(np.float32),
                              sport=nodes["sport"][i]))
        return cat(parts)
    return None


# ----------------------------------------------------------------------------
# observed sensor matrix at the control centre
# ----------------------------------------------------------------------------
def build_observed(meas, cfg, nodes, ben, bursts, rng):
    """Apply benign faults/outages then integrity attacks to the measured series."""
    n = meas.shape[0]
    g = np.arange(n)
    rat = np.array(list(cfg.pv_rated) + list(cfg.wind_rated) + [cfg.dg_rated, cfg.bat_power] +
                   list(cfg.load_peak) + [1000.0, 12.0])
    O = np.empty_like(meas)
    kmap = np.empty((n, N_SENS), dtype=np.int64)
    for i in range(N_SENS):
        T, ph = nodes["T"][i], nodes["phase"][i]
        k = np.floor((g * DT - ph) / T).astype(np.int64)
        kmap[:, i] = k
        gsrc = np.clip(np.floor((k * T + ph) / DT).astype(np.int64), 0, n - 1)
        O[:, i] = meas[gsrc, i]
    benign_mask = np.zeros(n, dtype=np.int16)
    # --- benign events
    for e in ben["outage"]:
        g0, g1 = int(np.ceil(e["t0"] / DT)), min(int(np.ceil(e["t1"] / DT)), n)
        if g0 >= 1 and g1 > g0:
            O[g0:g1, e["node"]] = O[g0 - 1, e["node"]]
            benign_mask[g0:g1 + int((e["t1"] - e["t0"]) / DT / 10)] |= 1
    for e in ben["route"]:
        g0, g1 = int(e["t0"] / DT), min(int(e["t1"] / DT), n)
        benign_mask[g0:g1] |= 2
    for e in ben["reconnect"]:
        g0 = int(e["t0"] / DT)
        benign_mask[g0:g0 + 6] |= 4
    for e in ben["admin"]:
        g0, g1 = int(e["t0"] / DT), min(int(e["t1"] / DT) + 1, n)
        benign_mask[g0:g1] |= 8
    for e in ben["fault"]:
        i = e["node"]
        g0, g1 = int(np.ceil(e["t0"] / DT)), min(int(np.ceil(e["t1"] / DT)), n)
        if g0 < 1 or g1 <= g0:
            continue
        L = g1 - g0
        if e["kind"] == "stuck":
            O[g0:g1, i] = O[g0 - 1, i]
        elif e["kind"] == "spike":
            sp = rng.random(L) < 0.10
            O[g0:g1, i] += sp * rng.uniform(0.05, 0.25, L) * rat[i] * e["sgn"]
        else:
            O[g0:g1, i] += e["sgn"] * e["mag"] * rat[i] * np.linspace(0, 1, L)
        benign_mask[g0:g1] |= 16
    # --- integrity attacks
    for b in bursts:
        p = b["params"]
        g0, g1 = int(np.ceil(b["t0"] / DT)), min(int(np.ceil(b["t1"] / DT)), n)
        if g1 <= g0:
            continue
        sl = slice(g0, g1)
        if b["cls"] == "FDIA":
            v = p["variant"]
            if v == "V1-bias":
                for s, sg in zip(p["sensors"], p["sign"]):
                    O[sl, s] += sg * p["gamma"] * np.maximum(np.abs(O[sl, s]), 0.2 * rat[s])
            elif v == "V2-ramp":
                ramp = np.linspace(0, 1, g1 - g0)
                for s, sg in zip(p["sensors"], p["sign"]):
                    O[sl, s] += sg * p["gamma"] * np.maximum(np.abs(O[sl, s]), 0.2 * rat[s]) * ramp
            else:
                gi, lj = p["sensors"]
                a = p["sign"][0] * p["gamma"] * np.maximum(np.abs(O[sl, gi]), 0.2 * rat[gi])
                O[sl, gi] += a
                O[sl, lj] += a / (1 + cfg.loss_frac)
        elif b["cls"] == "MITM":
            for i in p["nodes"]:
                m = u_hash(kmap[sl, i], i, b["id"]) < p["q"]
                sgn = np.where(u_hash(kmap[sl, i] + 7919, i, b["id"]) < 0.5, -1.0, 1.0)
                O[sl, i] += m * sgn * p["gamma"] * np.maximum(np.abs(O[sl, i]), 0.2 * rat[i])
        elif b["cls"] == "Replay":
            lag_g = int(round(p["lag"] / DT))
            for i in p["nodes"]:
                m = u_hash(kmap[sl, i], i, b["id"]) < p["frac"]
                src_idx = np.clip(np.arange(g0, g1) - lag_g, 0, n - 1)
                O[sl, i] = np.where(m, O[src_idx, i], O[sl, i])
        if b["cls"] in ("FDIA", "MITM", "Replay"):
            pass
    return O, benign_mask, kmap
