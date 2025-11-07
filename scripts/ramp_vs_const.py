#!/usr/bin/env python3
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from config.sim_config import load_sim_config
from simulation.temp_profile import RampHoldProfile, ConstantProfile
from simulation.cn_solver import CNSolver
from simulation.ciovati_model import CiovatiModel

# ─── Experiment parameters ─────────────────────────────────────────────────────
config_path = "config/sim_config.yml"
bake_temp_C = np.arange(200, 201, 1)  # °C
hold_time_h = np.arange(6, 7, 1)  # hours
ramp_rates = [1, 2, 5, 10]  # °C/min

base_exp_dir = "experiments/ramp_rate_comparison"
os.makedirs(base_exp_dir, exist_ok=True)
_paths = lambda fn, subdir="": os.path.join(base_exp_dir, subdir, fn)

# filenames
plot_cmp = "ramp_rate_comparison.pdf"
# plot_contour = "concentration_contour.pdf"
plot_diff = "diff_vs_rate.pdf"
plot_diff_const = "diff_vs_const.pdf"
metrics_const_csv = "metrics_const.csv"
metrics_rates_csv = "metrics_rates.csv"
stats_csv = "comparison_stats.csv"
# oxygen_csv = "oxygen_profile.csv"
oxygen_csv_const = "oxygen_profile_const.csv"
diff_csv = "diff_profile.csv"
temps_csv = "temps_const.csv"

# zoom
pen_depth_nm = 100.0
zoom_depth_nm = 150.0
zoom_depth_nm_contour = 100.0
# ───────────────────────────────────────────────────────────────────────────────

# load config & prepare
cfg = load_sim_config(config_path)
profiles = []
rates_metrics = []
comparison_stats = []

# ─── 1) Generate curves & metrics ──────────────────────────────────────────────
ref_profile = None
for bake_C in bake_temp_C:
    for hold in hold_time_h:

        start_K = cfg.temp_profile.start_C + 273.15
        bake_K = bake_C + 273.15

        subdir = f"{int(bake_C)}C_{int(hold)}h"
        run_dir = os.path.join(base_exp_dir, subdir)
        os.makedirs(run_dir, exist_ok=True)

        # ——— 1) CONSTANT profile ———————————————————————————————
        times_const_h, T_K_const, total_h_const = ConstantProfile(
            cfg, start_K, bake_K, hold
        ).generate()
        # solve diffusion
        print(f"Running constant profile @ {bake_C}°C, {hold}h")
        sol_c = CNSolver(cfg, T_K_const, total_h_const, CiovatiModel(cfg.ciovati))
        U_const = sol_c.get_oxygen_profile()[-1]
        # x‐grid
        x_grid = np.linspace(0, cfg.grid.x_max_nm, cfg.grid.n_x)
        # compute scalars
        half = U_const[0] / 2
        idx = np.where(U_const <= half)[0]
        const_metrics = {
            "temp_C": bake_C,
            "total_time_h": hold,
            "AUC": np.trapz(U_const, x_grid),
            "mean_U": U_const.mean(),
            "std_U": U_const.std(),
            "median_U": np.median(U_const),
            "d50_nm": x_grid[idx[0]] if idx.size else np.nan,
        }
        # save
        pd.DataFrame([const_metrics]).to_csv(
            _paths(metrics_const_csv, subdir), index=False
        )
        df_const = pd.DataFrame(
            {
                "depth_nm": x_grid,
                "U_const": U_const,
            }
        )
        df_const.to_csv(_paths(oxygen_csv_const, subdir), index=False)

        df_temps_const = pd.DataFrame(
            {
                "time_h": times_const_h,
                "T_K": T_K_const,
            }
        )
        df_temps_const.to_csv(_paths(temps_csv, subdir), index=False)

        # ─── 2) RAMP profile ───────────────────────────────────────────────────────
        for rr in ramp_rates:
            cfg.temp_profile.ramp_rate_C_per_min = rr

            # temperature profile
            times_h, T_K, total_h = RampHoldProfile(
                cfg, start_K, bake_K, hold
            ).generate()

            # oxygen profile
            print(f"Running ramp profile @ {bake_C}°C, {total_h:.2f}h, {rr}°C/min")
            solver = CNSolver(cfg, T_K, total_h, CiovatiModel(cfg.ciovati))
            U_final = solver.get_oxygen_profile()[-1]
            x_grid = np.linspace(0, cfg.grid.x_max_nm, cfg.grid.n_x)

            # save profile
            df_timedep = pd.DataFrame(
                {
                    "depth_nm": x_grid,
                    "U_final": U_final,
                }
            )
            filename = f"oxygen_profile_{rr:.2g}C_min.csv"
            df_timedep.to_csv(_paths(filename, subdir), index=False)

            temp_filename = f"temps_{rr:.2g}C_min.csv"
            df_temps_ramp = pd.DataFrame(
                {
                    "time_h": times_h,
                    "T_K": T_K,
                }
            )
            df_temps_ramp.to_csv(_paths(temp_filename, subdir), index=False)

            profiles.append(U_final)

            # scalar summaries
            auc = np.trapz(U_final, x_grid)
            meanU = U_final.mean()
            stdU = U_final.std()
            medU = np.median(U_final)
            half = U_final[0] / 2
            idx = np.where(U_final <= half)[0]
            d50 = x_grid[idx[0]] if idx.size else np.nan
            rates_metrics.append(
                {
                    "ramp_rate": rr,
                    "temp_C": bake_C,
                    "total_time_h": total_h,
                    "AUC": auc,
                    "mean_U": meanU,
                    "std_U": stdU,
                    "median_U": medU,
                    "d50_nm": d50,
                }
            )

            # plotting
            fig1, (ax_o, ax_t) = plt.subplots(1, 2, figsize=(12, 5))
            ax_o.plot(x_grid, U_final, label=f"{rr:.2g} °C/min")
            ax_t.plot(times_h, T_K - 273.15)
            if rr == 1:
                ref_profile = U_final

            # decorate & shared legend
            ax_o.set_xlim(0, zoom_depth_nm)
            ax_o.set_xlabel("Depth (nm)")
            ax_o.set_ylabel("[O] (at. %)")
            ax_o.set_title(f"[O] concentration @ {bake_C}°C, {total_h_const}h")
            ax_t.set_xlabel("Time (h)")
            ax_t.set_ylabel("Temperature (°C)")
            ax_t.set_title("T vs time")
        ax_o.plot(x_grid, U_const, "--", label="constant")
        ax_t.plot(times_const_h, T_K_const - 273.15)
        h_o, l_o = ax_o.get_legend_handles_labels()
        fig1.legend(h_o, l_o, loc="center right", title="Ramp rate", frameon=True)
        fig1.tight_layout(rect=[0, 0, 0.85, 1])
        fig1.savefig(_paths(plot_cmp, subdir), dpi=300)
        plt.close(fig1)

        df_rates = pd.DataFrame(rates_metrics)
        df_rates.to_csv(_paths(metrics_rates_csv, subdir), index=False)

        # ─── 2) Contour plot ───────────────────────────────────────────────────────────
        # profile_matrix = np.vstack(profiles)
        # fig2, ax2 = plt.subplots(figsize=(6, 5))
        # cf = ax2.contourf(all_x, ramp_rates, profile_matrix, levels=20, cmap="cividis")
        # fig2.colorbar(cf, ax=ax2, label="O concentration")
        # ax2.set_xlim(0, zoom_depth_nm_contour); ax2.set_xlabel("Depth (nm)")
        # ax2.set_ylabel("Ramp rate (°C/min)")
        # ax2.set_title("Concentration vs depth & ramp rate")
        # fig2.tight_layout(); fig2.savefig(_paths(plot_contour, subdir), dpi=300)

        # ─── 3) Ramp vs Constant Diff  ─────────────────────────────────────────────────────
        fig3, ax3 = plt.subplots(figsize=(6, 5))
        for rr, U in zip(ramp_rates, profiles):
            ax3.plot(x_grid, U - U_const, label=f"{rr:.2g} °C/min")
        ax3.axhline(0, color="black", linestyle="--", linewidth=0.5)
        ax3.set_xlim(0, zoom_depth_nm)
        ax3.set_xlabel("Depth (nm)")
        ax3.set_ylabel(f"Δ[O] (U_ramp - U_const)")
        ax3.set_title(f"Difference profile @ {bake_C}°C, {hold}h")
        ax3.legend(loc="best", title="Ramp rate", frameon=True)
        fig3.tight_layout()
        fig3.savefig(_paths(plot_diff_const, subdir), dpi=300)
        plt.close(fig3)

        # ─── 4) Difference plot & statistical comparison ───────────────────────────────
        fig4, ax4 = plt.subplots(figsize=(6, 5))
        for rr, U in zip(ramp_rates, profiles):
            ax4.plot(x_grid, U - ref_profile, label=f"{rr:.2g} °C/min")
            # Euclidean distance
            euclidean_dist = np.sqrt(np.sum((ref_profile - U) ** 2))
            comparison_stats.append({"ramp_rate": rr, "euclidean_dist": euclidean_dist})

        ax4.set_xlim(0, zoom_depth_nm)
        ax4.set_xlabel("Depth (nm)")
        ax4.set_ylabel("Δ[O] rel 1°C/min")
        ax4.set_title("Difference between ramp rates")
        ax4.legend()
        fig4.tight_layout()
        fig4.savefig(_paths(plot_diff, subdir), dpi=300)
        plt.close(fig4)

        # Save comparison stats
        dfs = pd.DataFrame(comparison_stats)
        dfs.to_csv(_paths(stats_csv, subdir), index=False)

        profiles = []
        metrics = []
        comparison_stats = []
