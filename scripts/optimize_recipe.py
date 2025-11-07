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

Strategy
--------
1. Normalise each metric to [0, 1]           (low is good for currents,
                                              high is good for depth)
2. Ideal point in this space = (0, 0, 1)
3. Choose the (t, T) that minimises Euclidean distance to the ideal.
"""

# ─── CONFIG SECTION ─────────────────────────────────────────────────────
PARENT_DIR = "experiments/2025-05-24_const_e10b3cd"
STEP_GLOB = "results"  # matches _1, _2, _d, …
N_PASSES = None  # auto‑detect everything present
TOP_K = 5  # how many best combos to list
# ────────────────────────────────────────────────────────────────────────

from pathlib import Path
import numpy as np
import pandas as pd
from scripts.simulation_analyzer import MultiOxidationCurrentDensityAnalyzer


# ────────────────────────────────────────────────────────────────────────
def normalise(arr: np.ndarray, minimise: bool) -> np.ndarray:
    """Scale `arr` to [0,1]; if minimise is True, 0 is best, 1 is worst."""
    a_min, a_max = np.nanmin(arr), np.nanmax(arr)
    if a_max == a_min:
        return np.zeros_like(arr)
    norm = (arr - a_min) / (a_max - a_min)
    return norm if minimise else 1.0 - norm


def main() -> None:
    # ─── run the multi‑pass analyser ────────────────────────────────────
    analyzer = MultiOxidationCurrentDensityAnalyzer(
        parent_dir=PARENT_DIR,
        n_passes=N_PASSES,
        step_glob=STEP_GLOB,
    )
    analyzer.run()

    # directories for output
    data_dir = analyzer.analysis_dir / "data"
    data_dir.mkdir(exist_ok=True)

    Z_max, _ = analyzer.metrics["max_current"]
    Z_surf, _ = analyzer.metrics["surface_current"]
    Z_xpeak, _ = analyzer.metrics["x_peak"]

    times, temps = analyzer.times, analyzer.temps
    n_pass = analyzer.n_passes

    global_best = (np.inf, None, None, None)  # dist, pass, t, T
    summary_lines: list[str] = []  # collect console text

    # ─── examine each pass ──────────────────────────────────────────────
    for p_idx in range(n_pass):
        z_max = Z_max[p_idx]
        z_surf = Z_surf[p_idx]
        z_x = Z_xpeak[p_idx]

        n_max = normalise(z_max, minimise=True)
        n_surf = normalise(z_surf, minimise=True)
        n_x = normalise(z_x, minimise=False)

        dist = np.sqrt(n_max**2 + n_surf**2 + (1 - n_x) ** 2)

        flat = dist.flatten()
        valid = ~np.isnan(flat)
        order = np.argsort(flat[valid])  # ascending = better

        header = f"\nPass {p_idx+1} — best {TOP_K} recipes (closest to ideal):"
        print(header)
        summary_lines.append(header)

        if order.size == 0:
            msg = "  (no valid data)"
            print(msg)
            summary_lines.append(msg)
            continue

        for rank, flat_idx in enumerate(np.flatnonzero(valid)[order[:TOP_K]], 1):
            j, i = np.unravel_index(flat_idx, dist.shape)
            d = dist[j, i]
            t_val, T_val = times[i], temps[j]
            line = (
                f"  {rank:>2}:  time = {t_val:6.2f} h, "
                f"temp = {T_val:6.1f} °C   →  distance = {d:7.4f}"
            )
            print(line)
            summary_lines.append(line)

            if d < global_best[0]:
                global_best = (d, p_idx + 1, t_val, T_val)

        # save full distance matrix
        df = pd.DataFrame(
            dist,
            index=[f"{T:.1f}" for T in temps],
            columns=[f"{t:.2f}" for t in times],
        )
        df.to_csv(data_dir / f"distance_pass{p_idx+1}.csv")

    # ─── global optimum ────────────────────────────────────────────────
    d_best, pass_no, t_best, T_best = global_best
    global_block = (
        "\n★  GLOBAL OPTIMUM ACROSS ALL PASSES  ★\n"
        f"   Pass   : {pass_no}\n"
        f"   Time   : {t_best:.2f} h\n"
        f"   Temp   : {T_best:.1f} °C\n"
        f"   Score  : (distance) {d_best:.4f}"
    )
    print(global_block)
    summary_lines.append(global_block)

    # ─── write summary to file ──────────────────────────────────────────
    summary_path = data_dir / "summary.txt"
    summary_path.write_text("\n".join(summary_lines))
    print(f"\nSummary written → {summary_path}")


if __name__ == "__main__":
    main()
