#!/usr/bin/env python3
"""
Optimise (time, temp) recipes across all oxidisation passes.

Goal
----
Minimise
    * max_current       (J_max / J_clean)
    * surface_current   (J_surface / J_clean)
while maximising
    * x_peak            (depth where J_max occurs)

Robust strategy
---------------
Instead of picking the single grid-point with the lowest distance to the
ideal (0,0,1) we look for *basins* of good performance: for every grid
point we aggregate the distances in a ±Δt / ±ΔT neighbourhood and rank
by that **robust score**.

Steps
1.   Normalise each metric → [0,1]  (low good for currents, high good for
     depth).
2.   Distance to ideal   d = √( n_max² + n_surf² + (1 - n_x)² ).
3.   Robust score  r(i,j) =  AGG_FN{ d(k,ℓ)  |  |t_i-t_k|≤Δt ∧ |T_j-T_ℓ|≤ΔT }.
4.   Pick (t,T) with the lowest robust score.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import scipy.ndimage as ndi  # ← NEW
from scripts.simulation_analyzer import MultiOxidationCurrentDensityAnalyzer

# ─── CONFIG SECTION ──────────────────────────────────────────────────────
PARENT_DIR = "experiments/2025-05-24_const_e10b3cd"
STEP_GLOB = "results"  # matches _1, _2, _d, …
N_PASSES = None  # auto-detect everything present
TOP_K = 5  # how many best combos to list (per pass)

# robustness parameters (physical units)
DTOL_H = 0.10  # ± hours tolerance band
TTOL_C = 3.0  # ± °C   tolerance band
AGG_FN = np.mean  # use np.max for worst-case, np.mean for average
# ─────────────────────────────────────────────────────────────────────────


# ─── helpers ─────────────────────────────────────────────────────────────
def normalise(arr: np.ndarray, minimise: bool) -> np.ndarray:
    """Scale arr to [0,1]; if minimise is True, 0 is best, 1 is worst."""
    a_min, a_max = np.nanmin(arr), np.nanmax(arr)
    if a_max == a_min:
        return np.zeros_like(arr)
    norm = (arr - a_min) / (a_max - a_min)
    return norm if minimise else 1.0 - norm


# ─── MAIN ────────────────────────────────────────────────────────────────
def main() -> None:

    # ── analyse all oxidation passes ────────────────────────────────────
    analyzer = MultiOxidationCurrentDensityAnalyzer(
        parent_dir=PARENT_DIR,
        n_passes=N_PASSES,
        step_glob=STEP_GLOB,
    )
    analyzer.run()

    data_dir = analyzer.analysis_dir / "data"
    data_dir.mkdir(exist_ok=True)

    Z_max, _ = analyzer.metrics["max_current"]
    Z_surf, _ = analyzer.metrics["surface_current"]
    Z_xpeak, _ = analyzer.metrics["x_peak"]

    times, temps = analyzer.times, analyzer.temps
    n_pass = analyzer.n_passes

    # ── convert physical tolerances to grid-radius (indices) ────────────
    dt_per_cell = np.diff(times).mean()  # assumes quasi-uniform grid
    dT_per_cell = np.diff(temps).mean()

    rad_t = max(1, int(round(DTOL_H / dt_per_cell)))
    rad_T = max(1, int(round(TTOL_C / dT_per_cell)))

    footprint = np.ones((2 * rad_T + 1, 2 * rad_t + 1), dtype=bool)

    global_best = (np.inf, None, None, None)  # r_score, pass#, t, T
    summary_lines: list[str] = []

    # ─── iterate over passes ────────────────────────────────────────────
    for p_idx in range(n_pass):
        z_max = Z_max[p_idx]
        z_surf = Z_surf[p_idx]
        z_x = Z_xpeak[p_idx]

        n_max = normalise(z_max, minimise=True)
        n_surf = normalise(z_surf, minimise=True)
        n_x = normalise(z_x, minimise=False)

        dist = np.sqrt(n_max**2 + n_surf**2 + (1.0 - n_x) ** 2)

        # ── robust score: aggregate distance in neighbourhood ───────────
        robust = ndi.generic_filter(dist, AGG_FN, footprint=footprint, mode="nearest")

        flat = robust.flatten()
        valid = ~np.isnan(flat)
        order = np.argsort(flat[valid])

        header = (
            f"\nPass {p_idx+1} — top {TOP_K} robust recipes "
            f"(Δt±{DTOL_H} h, ΔT±{TTOL_C} °C):"
        )
        print(header)
        summary_lines.append(header)

        if order.size == 0:
            msg = "  (no valid data)"
            print(msg)
            summary_lines.append(msg)
            continue

        for rank, flat_idx in enumerate(np.flatnonzero(valid)[order[:TOP_K]], 1):
            j, i = np.unravel_index(flat_idx, robust.shape)
            r = robust[j, i]
            t_val, T_val = times[i], temps[j]
            line = (
                f"  {rank:>2}:  time = {t_val:6.2f} h, "
                f"temp = {T_val:6.1f} °C   →  robust = {r:7.4f}"
            )
            print(line)
            summary_lines.append(line)

            if r < global_best[0]:
                global_best = (r, p_idx + 1, t_val, T_val)

        # save full robust matrix
        df = pd.DataFrame(
            robust,
            index=[f"{T:.1f}" for T in temps],
            columns=[f"{t:.2f}" for t in times],
        )
        df.to_csv(data_dir / f"robust_distance_pass{p_idx+1}.csv")

    # ─── global optimum ────────────────────────────────────────────────
    r_best, pass_no, t_best, T_best = global_best
    global_block = (
        "\n★  GLOBAL ROBUST OPTIMUM ACROSS ALL PASSES ★\n"
        f"   Pass   : {pass_no}\n"
        f"   Time   : {t_best:.2f} h\n"
        f"   Temp   : {T_best:.1f} °C\n"
        f"   Score  : {r_best:.4f}   "
        f"(AGG_FN = {AGG_FN.__name__}, Δt = ±{DTOL_H} h, ΔT = ±{TTOL_C} °C)"
    )
    print(global_block)
    summary_lines.append(global_block)

    # ─── write summary to file ─────────────────────────────────────────
    summary_path = data_dir / "summary.txt"
    summary_path.write_text("\n".join(summary_lines))
    print(f"\nSummary written → {summary_path}")


if __name__ == "__main__":
    main()
