#!/usr/bin/env python3
"""
temp_profile.py

Defines temperature-profile generators for SRF bake simulations.

Classes
-------
- BaseTempProfile : Abstract interface
- ConstantProfile : Flat bake at a single temperature
- ThreePhaseProfile: Ramp -> Hold -> Exponential cool

Usage
-----
From your simulation:
    profile = ThreePhaseProfile(cfg, start_K, bake_K, total_h)
    time_h, temps_K, t_hold_min = profile.generate()

Run as a script to see example plots:
    $ python temp_profile.py
"""

import numpy as np
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from types import SimpleNamespace


class BaseTempProfile(ABC):
    """Abstract interface for temperature profiles."""

    def __init__(
        self, cfg: SimpleNamespace, start_K: float, bake_K: float, total_h: float
    ):
        """
        Parameters
        ----------
        cfg : object
            Must have cfg.grid.n_t (number of time points) and, for
            three-phase, cfg.temp_profile.ramp_rate_C_per_min,
            .exp_b, .exp_c, .tol_K.
        start_K : float
            Starting temperature in K.
        bake_K : float
            Peak (bake) temperature in K.
        total_h : float
            Hold duration at bake_K in hours.
        """
        self.cfg = cfg
        self.start_K = start_K
        self.bake_K = bake_K
        self.total_h = total_h
        self.n_t = cfg.grid.n_t  # total number of output steps

    @abstractmethod
    def generate(self) -> tuple[np.ndarray, np.ndarray, float]:
        """
        Produce the profile.

        Returns
        -------
        time_h : np.ndarray, shape (n_t,)
            Time axis, in hours.
        temps_K : np.ndarray, shape (n_t,)
            Temperature at each time point, in K.
        total_hours : float
            Total hours (bake phase).
        """
        ...


class ConstantProfile(BaseTempProfile):
    """Keeps temperature flat at bake_K for the entire run."""

    def generate(self):
        # Uniform time grid from 0 -> total_h hours
        time_h = np.linspace(0.0, self.total_h, self.n_t)
        # Flat-line at bake_K
        temps_K = np.full(self.n_t, self.bake_K, dtype=np.double)
        # Everything is "hold" here
        t_hold_h = self.total_h
        return time_h, temps_K, t_hold_h


class TimeDepProfile(BaseTempProfile):
    """
    Generate a three-phase bake temperature profile in Kelvin, driven by config.

    Phases:
      1. HEATING: linear ramp from start_K to bake_K at ramp_rate (K/min).
      2. HOLD: constant at bake_K for total_h hours.
      3. COOLING: exponential decay back toward start_K.

    Parameters
    ----------
    cfg : SimConfig-like object
        Must have attributes:
          cfg.temp_profile.ramp_rate_C_per_min  Ramp rate in K/min
          cfg.grid.n_t                          Number of time steps
          cfg.temp_profile.cooling_time_h       Cooling duration in hours
          cfg.temp_profile.tol_K                Cooling tolerance (K)
    start_K : float
        Starting temperature in Kelvin.
    bake_K : float
        Peak (bake) temperature in Kelvin.
    total_h : float
        Duration to hold at bake_K, in hours.

    Returns
    -------
    time_h : np.ndarray, shape (n_t,)
        Time axis in hours.
    temps_K : np.ndarray, shape (n_t,)
        Temperature profile in Kelvin.
    total_hrs : float
        total time in hours.
    """

    def generate(self):
        # Unpack config
        ramp_rate = self.cfg.temp_profile.ramp_rate_C_per_min  # K/min
        cooling_time_h = self.cfg.temp_profile.cooling_time_h  # h
        tol_K = self.cfg.temp_profile.tol_K  # tolerance (K)

        # Convert durations to minutes
        t_heat = (self.bake_K - self.start_K) / ramp_rate
        t_hold = self.total_h * 60.0
        t_cool = cooling_time_h * 60.0

        # Compute dynamic amplitude for cooling
        a_dyn = self.bake_K - self.start_K
        if a_dyn <= 0:
            raise ValueError(
                f"Bake_K ({self.bake_K}) must exceed start_K ({self.start_K})"
            )

        # Infer cooling rate constant so that at end of cooling
        # profile drops within tol_K of start_K:
        #   a_dyn * exp(-b_per_min * t_cool) = tol_K
        b_per_min = np.log(a_dyn / tol_K) / t_cool

        # Build uniform time grid (minutes)
        total_min = t_heat + t_hold + t_cool
        t = np.linspace(0.0, total_min, self.n_t)

        # Piecewise definition with explicit 'tau'
        temps_K = np.piecewise(
            t,
            [t <= t_heat, (t > t_heat) & (t <= t_heat + t_hold), t > t_heat + t_hold],
            [
                # HEATING
                lambda tau: self.start_K + ramp_rate * tau,
                # HOLD
                lambda tau: self.bake_K,
                # COOLING
                lambda tau: a_dyn * np.exp(-b_per_min * (tau - t_heat - t_hold))
                + self.start_K,
            ],
        )

        # Convert minutes to hours
        time_h = t / 60.0
        return time_h, temps_K, total_min / 60.0


class TwoStepProfile(BaseTempProfile):
    """
    Two‐step bake profile:
    1. Ramp   from start_K → t1_K
    2. Hold   at t1_K for bake1_h hours
    3. Ramp   from t1_K → t2_K
    4. Hold   at t2_K for bake2_h hours
    5. Cool   exponentially toward start_K over cooling_time_h hours
    """

    def generate(self):
        # Unpack config
        ramp_rate = self.cfg.temp_profile.ramp_rate_C_per_min  # K/min
        cooling_time_h = self.cfg.temp_profile.cooling_time_h  # h
        tol_K = self.cfg.temp_profile.tol_K  # tolerance K

        # Unpack two-step params
        ts = self.cfg.temp_profile.two_step
        t1_K = ts.t1_C + 273.15  # first-step peak (K)
        bake1_h = ts.bake1_h  # hold time at t1_K (h)
        t2_K = ts.t2_C + 273.15  # second-step peak (K)
        bake2_h = ts.bake2_h  # hold time at t2_K (h)

        # Compute individual phase durations (minutes)
        t_ramp1 = (t1_K - self.start_K) / ramp_rate
        t_hold1 = bake1_h * 60.0
        t_ramp2 = (t2_K - t1_K) / ramp_rate
        t_hold2 = bake2_h * 60.0
        t_cool = cooling_time_h * 60.0

        # Dynamic amplitude for cooling
        a_dyn = t2_K - self.start_K
        if a_dyn <= 0:
            raise ValueError(f"t2_K ({t2_K}) must exceed start_K ({self.start_K})")

        # Infer cooling rate constant so that at end of cooling
        # a_dyn * exp(-b_per_min * t_cool) = tol_K
        b_per_min = np.log(a_dyn / tol_K) / t_cool

        # Build uniform time grid (minutes)
        total_min = t_ramp1 + t_hold1 + t_ramp2 + t_hold2 + t_cool
        t = np.linspace(0.0, total_min, self.n_t)

        # Piecewise definition with explicit 'tau'
        temps_K = np.piecewise(
            t,
            [
                t <= t_ramp1,
                (t > t_ramp1) & (t <= t_ramp1 + t_hold1),
                (t > t_ramp1 + t_hold1) & (t <= t_ramp1 + t_hold1 + t_ramp2),
                (t > t_ramp1 + t_hold1 + t_ramp2)
                & (t <= t_ramp1 + t_hold1 + t_ramp2 + t_hold2),
                t > t_ramp1 + t_hold1 + t_ramp2 + t_hold2,
            ],
            [
                # 1) ramp to t1_K
                lambda tau: self.start_K + ramp_rate * tau,
                # 2) hold at t1_K
                lambda tau: t1_K,
                # 3) ramp to t2_K
                lambda tau: t1_K + ramp_rate * (tau - t_ramp1 - t_hold1),
                # 4) hold at t2_K
                lambda tau: t2_K,
                # 5) exponential cooldown
                lambda tau: a_dyn
                * np.exp(-b_per_min * (tau - t_ramp1 - t_hold1 - t_ramp2 - t_hold2))
                + self.start_K,
            ],
        )

        # Convert minutes -> hours for the x-axis
        time_h = t / 60.0
        # return holds combined in minutes
        t_hold_total = t_hold1 + t_hold2
        return time_h, temps_K, total_min / 60.0


class RampHoldProfile(BaseTempProfile):
    """
    Generate a two-phase bake temperature profile (no cooling) in Kelvin, driven by config.

    Phases:
      1. HEATING: linear ramp from start_K to bake_K at ramp_rate (K/min).
      2. HOLD: constant at bake_K for total_h hours.

    Parameters
    ----------
    cfg : SimConfig-like object
        Must have attributes:
          cfg.temp_profile.ramp_rate_C_per_min  Ramp rate in K/min
          cfg.grid.n_t                          Number of time steps
    start_K : float
        Starting temperature in Kelvin.
    bake_K : float
        Peak (bake) temperature in Kelvin.
    total_h : float
        Duration to hold at bake_K, in hours.

    Returns
    -------
    time_h : np.ndarray, shape (n_t,)
        Time axis in hours.
    temps_K : np.ndarray, shape (n_t,)
        Temperature profile in Kelvin.
    t_hold_hrs : float
        Hold time in hours.
    """

    def generate(self):
        # Unpack config
        ramp_rate = self.cfg.temp_profile.ramp_rate_C_per_min  # K/min
        n_t = self.n_t  # number of time steps

        # Compute durations in minutes
        t_heat = (self.bake_K - self.start_K) / ramp_rate
        t_hold = self.total_h * 60.0
        total_min = t_heat + t_hold

        # Build uniform time grid in minutes
        t = np.linspace(0.0, total_min, n_t)

        # Piecewise definition with explicit 'tau'
        temps_K = np.piecewise(
            t,
            [t <= t_heat, t > t_heat],
            [
                # 1) HEATING
                lambda tau: self.start_K + ramp_rate * tau,
                # 2) HOLD
                lambda tau: self.bake_K,
            ],
        )

        # Convert minutes -> hours
        time_h = t / 60.0
        return time_h, temps_K, total_min / 60.0  # t_hold in hours


def main():
    """Quick visual sanity-check of constant, three-phase, and two-step profiles."""
    # dummy config stub
    cfg = SimpleNamespace(
        temp_profile=SimpleNamespace(
            ramp_rate_C_per_min=2.0,  # 2 K/mi
            cooling_time_h=4.0,  # 24 h
            tol_K=1.0,  # stop when within 1 K
            two_step=SimpleNamespace(
                t1_C=80.0,  # first-step peak temp (°C)
                bake1_h=24.0,  # hold at t1 for 24 h
                t2_C=120.0,  # second-step peak temp (°C)
                bake2_h=48.0,  # hold at t2 for 48 h
            ),
        ),
        grid=SimpleNamespace(n_t=2001),  # 2001 time points
    )

    # example parameters (°C → K)
    start_C, bake_C, total_h = 20.0, 120.0, 48.0
    start_K = start_C + 273.15
    bake_K = bake_C + 273.15

    # instantiate profiles
    time_dep = TimeDepProfile(cfg, start_K, bake_K, total_h)
    const = ConstantProfile(cfg, start_K, bake_K, total_h)
    two = TwoStepProfile(cfg, start_K, bake_K, total_h)
    ramp_hold = RampHoldProfile(cfg, start_K, bake_K, total_h)

    # generate
    t3, T3, _ = time_dep.generate()
    tc, Tc, _ = const.generate()
    t2s, T2s, _ = two.generate()
    t_rh, T_rh, _ = ramp_hold.generate()

    # plot them all
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(t_rh, T_rh, label="Ramp→Hold")
    ax.plot(t3, T3, label="Three-Phase Ramp→Hold→Cool")
    ax.plot(tc, Tc, "--", label="Constant @ Bake Temp")
    ax.plot(t2s, T2s, "-.", label="Two-Step Ramp→Hold1→Ramp→Hold2→Cool")

    ax.set_xlabel("Time (h)")
    ax.set_ylabel("Temperature (K)")
    ax2 = ax.secondary_yaxis(
        "right", functions=(lambda x: x - 273.15, lambda x: x + 273.15)
    )
    ax2.set_ylabel("Temperature (°C)")

    ax.legend(loc="best")
    plt.title("Temperature Profile Examples")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
