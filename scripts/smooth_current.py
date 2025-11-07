import os
from pathlib import Path
import glob
import numpy as np
import pandas as pd
from scipy import interpolate

BASE_DIR = Path("experiments/2025-05-24_const_e10b3cd")
RESULTS_DIR = Path("results")


folders = glob.glob(str(BASE_DIR / RESULTS_DIR / "sim_t*_T*"))

for f in folders:
    data_dir = Path(f) / "data"
    df = pd.read_csv(
        data_dir / "current_density_corrected.csv",
        usecols=["x", "current_density_corrected", "J_clean_corr", "J_dirty_corr"],
    )
    x = df["x"].to_numpy()  # 1‑D array of x values
    current_density = df[
        "current_density_corrected"
    ].to_numpy()  # 1‑D array of current density
    J_clean = df["J_clean_corr"].to_numpy()  # 1‑D array of clean current density
    J_dirty = df["J_dirty_corr"].to_numpy()  # 1‑D array of dirty current density

    df_critical_current = pd.read_csv(
        data_dir / "critical_current_density.csv",
        usecols=["x", "critical_current_density"],
    )
    critical_current = df_critical_current["critical_current_density"].to_numpy()

    # create a smooth interpolation of J(x) data
    j_smooth = interpolate.Akima1DInterpolator(
        x,
        current_density,
        method="akima",
        extrapolate=False,
    )

    j_smooth_clean = interpolate.Akima1DInterpolator(
        x,
        J_clean,
        method="akima",
        extrapolate=False,
    )
    j_smooth_dirty = interpolate.Akima1DInterpolator(
        x,
        J_dirty,
        method="akima",
        extrapolate=False,
    )

    # create a smooth interpolation of dJ(x)/dx
    deriv_j_smooth = j_smooth.derivative(1)

    x_smooth = np.linspace(0, df["x"].max(), 10000)  # 1‑D array of x values

    j_critical_smooth = interpolate.Akima1DInterpolator(
        x,
        critical_current,
        method="akima",
        extrapolate=False,
    )

    output_file = data_dir / "smooth_current_density.csv"
    output_df = pd.DataFrame(
        {
            "x": x_smooth,
            "current_density_smooth": j_smooth(x_smooth),
            "derivative_current_density_smooth": deriv_j_smooth(x_smooth),
            "J_clean": j_smooth_clean(x_smooth),
            "J_dirty": j_smooth_dirty(x_smooth),
        }
    )
    output_df.to_csv(output_file, index=False)

    output_file_critical = data_dir / "smooth_critical_current_density.csv"
    output_df_critical = pd.DataFrame(
        {
            "x": x_smooth,
            "critical_current_density_smooth": j_critical_smooth(x_smooth),
        }
    )
    output_df_critical.to_csv(output_file_critical, index=False)
