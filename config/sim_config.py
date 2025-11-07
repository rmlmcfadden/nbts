import yaml
import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class TimeConfig:
    start_h: float
    stop_h: float
    step_h: float


@dataclass
class TemperatureConfig:
    start_C: float
    stop_C: float
    step_C: float


@dataclass
class TwoStepConfig:
    t1_C: float  # first‐step peak temp (°C)
    bake1_h: float  # hold time at t1_C (h)
    t2_C: float  # second‐step peak temp (°C)
    bake2_h: float  # hold time at t2_C (h)


@dataclass
class ProfileConfig:
    start_C: float
    ramp_rate_C_per_min: float
    cooling_time_h: float
    tol_K: float
    two_step: Optional[TwoStepConfig] = None


@dataclass
class GridConfig:
    x_max_nm: float
    n_x: int
    n_t: int


@dataclass
class InitialConfig:
    u0: float
    v0: float
    base_O: float
    lambda_0_nm: float


@dataclass
class CiovatiConfig:
    D_0: float
    E_A_D: float
    k_A: float
    E_A_k: float
    u_0: float
    v_0: float
    c_0: float


@dataclass
class OutputConfig:
    directory: str


@dataclass
class ArgsConfig:
    applied_field_mT: float = 0.0
    dead_layer_nm: float = 0.0
    demag_factor: float = 0.0


@dataclass
class SimConfig:
    time: TimeConfig
    temperature: TemperatureConfig
    temp_profile: ProfileConfig
    grid: GridConfig
    initial: InitialConfig
    ciovati: CiovatiConfig
    output: OutputConfig
    args: ArgsConfig


def load_sim_config(path: str) -> SimConfig:
    """Load and validate simulation configuration from a YAML file."""
    with open(path) as f:
        raw = yaml.safe_load(f)

    # 1) Time sweep
    t = raw["time"]
    time_cfg = TimeConfig(
        start_h=t["start_h"],
        stop_h=t["stop_h"],
        step_h=t["step_h"],
    )

    # 2) Temperature sweep
    tmp = raw["temperature"]
    temp_cfg = TemperatureConfig(
        start_C=tmp["start_C"],
        stop_C=tmp["stop_C"],
        step_C=tmp["step_C"],
    )

    # 3) Profile (including optional two_step)
    pp = raw["temp_profile"]
    # parse two_step if present
    ts_raw = pp.get("two_step")
    two_step_cfg = None
    if ts_raw is not None:
        two_step_cfg = TwoStepConfig(
            t1_C=ts_raw["t1_C"],
            bake1_h=ts_raw["bake1_h"],
            t2_C=ts_raw["t2_C"],
            bake2_h=ts_raw["bake2_h"],
        )

    profile_cfg = ProfileConfig(
        start_C=pp["start_C"],
        ramp_rate_C_per_min=pp["ramp_rate_C_per_min"],
        cooling_time_h=pp["cooling_time_h"],
        tol_K=pp["tol_K"],
        two_step=two_step_cfg,
    )

    # 4) Grid
    g = raw["grid"]
    grid_cfg = GridConfig(
        x_max_nm=g["x_max_nm"],
        n_x=g["n_x"],
        n_t=g["n_t"],
    )

    # 5) Initial conditions
    init = raw["initial"]
    initial_cfg = InitialConfig(
        u0=init["u0"],
        v0=init["v0"],
        base_O=init["base_O"],
        lambda_0_nm=init["lambda_0_nm"],
    )

    # 6) Ciovati model params
    civ = raw["ciovati"]
    D = civ["D"]
    k = civ["k"]
    ciovati_cfg = CiovatiConfig(
        D_0=D["D_0"],
        E_A_D=D["E_A"],
        k_A=k["A"],
        E_A_k=k["E_A"],
        u_0=civ["u0"],
        v_0=civ["v0"],
        c_0=civ["c0"],
    )

    # 7) Output
    out = raw["output"]
    output_cfg = OutputConfig(directory=out["directory"])

    # 8) Extra args (with defaults if missing)
    ar = raw.get("args", {})
    args_cfg = ArgsConfig(
        applied_field_mT=ar.get("applied_field_mT", 0.0),
        dead_layer_nm=ar.get("dead_layer_nm", 0.0),
        demag_factor=ar.get("demag_factor", 0.0),
    )

    return SimConfig(
        time=time_cfg,
        temperature=temp_cfg,
        temp_profile=profile_cfg,
        grid=grid_cfg,
        initial=initial_cfg,
        ciovati=ciovati_cfg,
        output=output_cfg,
        args=args_cfg,
    )
