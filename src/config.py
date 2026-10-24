"""Global configuration for the MQTT/CoAP hybrid-microgrid simulator.

All stochastic components are driven by a single seed so that the data set is
exactly reproducible.  `SiteConfig` carries every site-specific parameter; a
second, deliberately *shifted* site ("site B") is used for the cross-site
generalisation experiment.
"""
from dataclasses import dataclass, field, replace
import numpy as np

DT = 5.0                    # plant / sensor grid step [s]
WIN = 30.0                  # feature window length [s]
STEPS_PER_WIN = int(WIN / DT)
DAY = 86400.0
STEPS_PER_DAY = int(DAY / DT)
WINS_PER_DAY = int(DAY / WIN)

# ----------------------------------------------------------------------------
# Class taxonomy
# ----------------------------------------------------------------------------
CLASSES = ["Normal", "FDIA", "Flood", "Amplification", "MITM", "Replay", "Recon"]
CLS = {c: i for i, c in enumerate(CLASSES)}

# sensors: (name, transport[0=MQTT,1=CoAP], kind)
SENSORS = [
    ("pv1", 0, "pv"), ("pv2", 0, "pv"), ("pv3", 0, "pv"),
    ("w1", 0, "wind"), ("w2", 0, "wind"), ("dg", 0, "diesel"), ("bat", 0, "battery"),
    ("load1", 1, "load"), ("load2", 1, "load"), ("pyr", 1, "irradiance"), ("anem", 1, "windspeed"),
]
SENSOR_IDX = {s[0]: i for i, s in enumerate(SENSORS)}
N_SENS = len(SENSORS)
SENS_TRANSPORT = np.array([s[1] for s in SENSORS])
PERIOD = np.array([5.0] * 7 + [10.0] * 4)        # publication period [s]


@dataclass(frozen=True)
class SiteConfig:
    name: str = "A"
    seed: int = 20261022
    n_days: int = 30
    n_commission_days: int = 3          # clean commissioning interval (no attacks)
    start_doy: int = 152                # 1 June
    latitude: float = 33.3
    # --- generation / load ratings [kW]
    pv_rated: tuple = (40.0, 40.0, 25.0)
    wind_rated: tuple = (30.0, 30.0)
    wind_scale_c: float = 6.5           # Weibull scale [m/s]
    wind_shape_k: float = 2.0
    dg_rated: float = 80.0
    dg_min: float = 25.0
    bat_energy: float = 250.0           # kWh
    bat_power: float = 40.0             # kW
    load_peak: tuple = (45.0, 28.0)
    loss_frac: float = 0.015            # unmeasured distribution loss (fraction of load)
    sensor_noise_rel: float = 0.004     # of rating
    sensor_noise_prop: float = 0.002    # proportional
    # --- network
    rtt_med: tuple = (0.045, 0.120)     # median RTT per transport [s]
    rtt_sig: tuple = (0.35, 0.50)       # lognormal sigma
    loss_base: tuple = (0.003, 0.015)   # base packet-loss probability per transport
    clock_offset_sd: float = 0.15       # device clock offset sd [s]
    con_share: float = 0.7
    # --- benign anomaly rates (events per node per day)
    r_fault: float = 0.25
    r_outage: float = 0.35
    r_route: float = 0.08
    r_reconnect: float = 0.4
    r_admin_wild: float = 1.0           # per day (system-wide)
    # --- attack schedule: expected bursts per day per (class, transport)
    burst_rate: float = 6.0
    burst_min: float = 120.0
    burst_max: float = 600.0


SITE_A = SiteConfig()
# Site B: different climate, ratings, link quality and benign-event rates
SITE_B = replace(
    SiteConfig(), name="B", seed=77031, start_doy=240, latitude=24.7,
    pv_rated=(30.0, 30.0, 30.0), wind_rated=(40.0, 20.0), wind_scale_c=5.2,
    dg_rated=70.0, dg_min=20.0, bat_energy=180.0, bat_power=30.0,
    load_peak=(40.0, 34.0), loss_frac=0.025, rtt_med=(0.070, 0.180), rtt_sig=(0.45, 0.60),
    loss_base=(0.006, 0.030), clock_offset_sd=0.30, con_share=0.55,
    r_fault=0.4, r_outage=0.6, r_route=0.15, r_reconnect=0.7,
)
