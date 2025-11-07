import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ─── USER CONFIG ────────────────────────────────────────────────────────────────
BASE_DIR = "experiments/ramp_rate_comparison"
PLOTS_DIR = "experiments/ramp_rate_plots"
os.makedirs(PLOTS_DIR, exist_ok=True)

RAMP_RATES = [1, 2, 5, 10]  # °C/min for full grids
COMP_RAMPS = [2, 5, 10]  # °C/min for 1×3 comparison plots
CONST_FILE = "oxygen_profile_const.csv"

OUT_MAX_PDF = os.path.join(PLOTS_DIR, "max_diff_heatmaps.pdf")
OUT_DEP_PDF = os.path.join(PLOTS_DIR, "depth_at_max_heatmaps.pdf")
OUT_CMP_PDF = os.path.join(PLOTS_DIR, "max_diff_vs_1C.pdf")
OUT_PCT_CONST = os.path.join(PLOTS_DIR, "pct_vs_constant.pdf")
OUT_PCT_1C = os.path.join(PLOTS_DIR, "pct_vs_1C.pdf")
OUT_COMBINED_REL = os.path.join(PLOTS_DIR, "combined_rate_comparison.pdf")
OUT_COMBINED_CONST = os.path.join(PLOTS_DIR, "combined_rate_comparison_const.pdf")
# ────────────────────────────────────────────────────────────────────────────────

# 1) discover all simulation subfolders named like "100C_6h"
all_dirs = [
    d
    for d in glob.glob(os.path.join(BASE_DIR, "*"))
    if os.path.isdir(d) and d.endswith("h") and "C_" in os.path.basename(d)
]

# 2) parse out unique bake temps and hold times
temps, hours = [], []
for d in all_dirs:
    base = os.path.basename(d)[:-1]  # strip trailing 'h'
    t_s, h_s = base.split("C_")  # split "120C_42" → ["120","42"]
    try:
        temps.append(float(t_s))
        hours.append(float(h_s))
    except ValueError:
        pass

unique_temps = sorted(set(temps))
unique_hours = sorted(set(hours))

# 3) prepare storage for metrics
nT, nH = len(unique_temps), len(unique_hours)
Z_max = {r: np.full((nT, nH), np.nan) for r in RAMP_RATES}
Z_dep = {r: np.full((nT, nH), np.nan) for r in RAMP_RATES}
Z_pct_const = {r: np.full((nT, nH), np.nan) for r in RAMP_RATES}
Z_pct = {r: np.full((nT, nH), np.nan) for r in COMP_RAMPS}
Z_cmp = {r: np.full((nT, nH), np.nan) for r in COMP_RAMPS}
Z_r_max = np.full((nT, nH), np.nan)  # peak of the ramp profile
Z_const_max = np.full((nT, nH), np.nan)  # peak of the constant profile

# 4) loop through folders, compute absolute metrics
for d in all_dirs:
    base = os.path.basename(d)[:-1]
    t_s, h_s = base.split("C_")
    try:
        t = float(t_s)
        h = float(h_s)
    except ValueError:
        continue

    i_t = unique_temps.index(t)
    i_h = unique_hours.index(h)

    # read constant profile
    const_fp = os.path.join(d, CONST_FILE)
    if not os.path.exists(const_fp):
        continue
    df0 = pd.read_csv(const_fp)
    x = df0["depth_nm"].to_numpy()  # depth grid
    c0 = df0["U_const"].to_numpy()  # concentration
    Z_const_max[i_t, i_h] = c0[0]

    # for each ramp rate, compute difference
    for r in RAMP_RATES:
        ramp_fp = os.path.join(d, f"oxygen_profile_{r}C_min.csv")
        if not os.path.exists(ramp_fp):
            continue
        df = pd.read_csv(ramp_fp)
        cr = df["U_final"].to_numpy()
        diff = np.abs(cr - c0)
        idx = np.nanargmax(diff)

        Z_max[r][i_t, i_h] = diff[idx]
        Z_dep[r][i_t, i_h] = x[idx]

        if not r in COMP_RAMPS:
            Z_r_max[i_t, i_h] = cr[0]

        if r in COMP_RAMPS:
            Z_cmp[r][i_t, i_h] = np.abs(Z_max[r][i_t, i_h] - Z_max[1][i_t, i_h])
            Z_pct[r][i_t, i_h] = 100.0 * (Z_cmp[r][i_t, i_h] / Z_r_max[i_t, i_h])
        Z_pct_const[r][i_t, i_h] = 100.0 * (Z_max[r][i_t, i_h] / Z_const_max[i_t, i_h])


# 5) build meshgrid for plotting
H, T = np.meshgrid(unique_hours, unique_temps)

# 6) Plot A: max|ΔC| heatmaps
fig1, axes1 = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
axes1 = axes1.flatten()
for ax, r in zip(axes1, RAMP_RATES):
    pcm = ax.pcolormesh(H, T, Z_max[r], cmap="cividis", shading="gouraud")
    fig1.colorbar(pcm, ax=ax, label="Max |ΔC|")
    cs = ax.contour(H, T, Z_max[r], levels=8, colors="white", linewidths=1)
    ax.clabel(cs, inline=True, fmt="%.2e", fontsize=8)
    ax.set_title(f"{r} °C/min")
    ax.set_xlabel("Hold time (h)")
    ax.set_ylabel("Bake temp (°C)")

fig1.suptitle("Maximum difference vs. constant profile", y=0.95)
fig1.tight_layout()
fig1.savefig(OUT_MAX_PDF)
print(f"Wrote {OUT_MAX_PDF}")

# 7) Plot B: depth at max|ΔC| heatmaps
fig2, axes2 = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
axes2 = axes2.flatten()
for ax, r in zip(axes2, RAMP_RATES):
    pcm = ax.pcolormesh(H, T, Z_dep[r], cmap="cividis", shading="gouraud")
    fig2.colorbar(pcm, ax=ax, label="Depth of max ΔC (nm)")
    cs = ax.contour(H, T, Z_dep[r], levels=8, colors="white", linewidths=1)
    ax.clabel(cs, inline=True, fmt="%.2f", fontsize=8)
    ax.set_title(f"{r} °C/min")
    ax.set_xlabel("Hold time (h)")
    ax.set_ylabel("Bake temp (°C)")

fig2.suptitle("Depth at which maximum difference occurs", y=0.95)
fig2.tight_layout()
fig2.savefig(OUT_DEP_PDF)
print(f"Wrote {OUT_DEP_PDF}")

# 8) Plot C: absolute difference vs 1 °C/min
fig3, axes3 = plt.subplots(1, 3, figsize=(15, 4), sharex=True, sharey=True)
axes3 = axes3.flatten()
for ax, r in zip(axes3, COMP_RAMPS):
    # Z_cmp = np.abs(Z_max[r] - Z_max[1])
    pcm = ax.pcolormesh(H, T, Z_cmp[r], cmap="cividis", shading="gouraud")
    fig3.colorbar(pcm, ax=ax, label=f"Δ(max|ΔC|) [{r}–1]")
    cs = ax.contour(H, T, Z_cmp[r], levels=8, colors="white", linewidths=1)
    ax.clabel(cs, inline=True, fmt="%.2e", fontsize=8)
    ax.set_title(f"{r} vs 1 °C/min")
    ax.set_xlabel("Hold time (h)")
    ax.set_ylabel("Bake temp (°C)")

fig3.suptitle("Change in max|ΔC| relative to 1 °C/min", y=0.98)
fig3.tight_layout()
fig3.savefig(OUT_CMP_PDF)
print(f"Wrote {OUT_CMP_PDF}")

# 9) Plot D: percent difference vs constant

fig4, axes4 = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
axes4 = axes4.flatten()
for ax, r in zip(axes4, RAMP_RATES):
    pcm = ax.pcolormesh(H, T, Z_pct_const[r], cmap="cividis", shading="gouraud")
    fig4.colorbar(pcm, ax=ax, label="% max|ΔC| / max C_const")
    cs = ax.contour(H, T, Z_pct_const[r], levels=8, colors="white", linewidths=1)
    ax.clabel(cs, inline=True, fmt="%.1f%%", fontsize=8)
    ax.set_title(f"{r} °C/min")
    ax.set_xlabel("Hold time (h)")
    ax.set_ylabel("Bake temp (°C)")

fig4.suptitle("Relative max difference vs constant [%]", y=0.95)
fig4.tight_layout()
fig4.savefig(OUT_PCT_CONST)
print(f"Wrote {OUT_PCT_CONST}")

# 10) Plot E: percent ratio vs 1 °C/min
fig5, axes5 = plt.subplots(1, 3, figsize=(15, 4), sharex=True, sharey=True)
axes5 = axes5.flatten()
for ax, r in zip(axes5, COMP_RAMPS):
    pcm = ax.pcolormesh(H, T, Z_pct[r], cmap="cividis", shading="gouraud")
    fig5.colorbar(pcm, ax=ax, label=f"% max|ΔC| [{r} / 1]")
    cs = ax.contour(H, T, Z_pct[r], levels=8, colors="white", linewidths=1)
    ax.clabel(cs, inline=True, fmt="%.1f%%", fontsize=8)
    ax.set_title(f"{r} vs 1 °C/min")
    ax.set_xlabel("Hold time (h)")
    ax.set_ylabel("Bake temp (°C)")

fig5.suptitle("Percent max|ΔC| relative to 1 °C/min", y=0.98)
fig5.tight_layout()
fig5.savefig(OUT_PCT_1C)
print(f"Wrote {OUT_PCT_1C}")


# ─── Combined Plot (2×4) ────────────────────────────────────────────────────────
fig6, axes = plt.subplots(2, 4, figsize=(22, 10), sharex=True, sharey=True)
axes = axes.flatten()

for idx, r in enumerate(RAMP_RATES):
    # Row 0: absolute Δ max|ΔC|
    ax = axes[idx]
    pcm = ax.pcolormesh(H, T, Z_max[r], cmap="cividis", shading="gouraud")
    fig6.colorbar(pcm, ax=ax, label="Max |ΔC|")
    cs = ax.contour(H, T, Z_max[r], levels=8, colors="white", linewidths=1)
    ax.clabel(cs, inline=True, fmt="%.2e", fontsize=8)
    ax.set_title(f"{r} °C/min")
    ax.set_xlabel("Hold time (h)")
    ax.set_ylabel("Bake temp (°C)")

    # Row 1: percent (ratio) max|ΔC|
    ax = axes[idx + 4]
    pcm = ax.pcolormesh(H, T, Z_pct_const[r], cmap="cividis", shading="gouraud")
    fig6.colorbar(pcm, ax=ax, label="% max|ΔC| / max C_const")
    cs = ax.contour(H, T, Z_pct_const[r], levels=8, colors="white", linewidths=1)
    ax.clabel(cs, inline=True, fmt="%.1f%%", fontsize=8)
    ax.set_title(f"{r} °C/min")
    ax.set_xlabel("Hold time (h)")
    ax.set_ylabel("Bake temp (°C)")

fig6.suptitle("Ramp‐rate comparisons against constant profile", y=0.95, fontsize=16)
fig6.tight_layout(rect=[0, 0, 1, 0.94])
fig6.savefig(OUT_COMBINED_CONST, dpi=300)
print(f"Wrote {OUT_COMBINED_CONST}")


# ─── Combined Plot (2×3) ────────────────────────────────────────────────────────
fig7, axes = plt.subplots(2, 3, figsize=(18, 10), sharex=True, sharey=True)
axes = axes.flatten()

for idx, r in enumerate(COMP_RAMPS):
    # Row 0: absolute Δ max|ΔC|
    ax = axes[idx]
    pcm = ax.pcolormesh(H, T, Z_cmp[r], cmap="cividis", shading="gouraud")
    fig7.colorbar(pcm, ax=ax, label=f"Δ max|ΔC| [{r}–1]")
    cs = ax.contour(H, T, Z_cmp[r], levels=8, colors="white", linewidths=1)
    ax.clabel(cs, inline=True, fmt="%.2e", fontsize=8)
    ax.set_title(f"{r} vs 1 °C/min\n(abs)")
    ax.set_xlabel("Hold time (h)")
    ax.set_ylabel("Bake temp (°C)")

    # Row 1: percent (ratio) max|ΔC|
    ax = axes[idx + 3]
    pcm = ax.pcolormesh(H, T, Z_pct[r], cmap="cividis", shading="gouraud")
    fig7.colorbar(pcm, ax=ax, label=f"% max|ΔC| [{r}/1]")
    cs = ax.contour(H, T, Z_pct[r], levels=8, colors="white", linewidths=1)
    ax.clabel(cs, inline=True, fmt="%.1f%%", fontsize=8)
    ax.set_title(f"{r} vs 1 °C/min\n(%)")
    ax.set_xlabel("Hold time (h)")
    ax.set_ylabel("Bake temp (°C)")

fig7.suptitle("Ramp‐rate comparisons against 1 °C/min", y=0.95, fontsize=16)
fig7.tight_layout(rect=[0, 0, 1, 0.94])
fig7.savefig(OUT_COMBINED_REL, dpi=300)
print(f"Wrote {OUT_COMBINED_REL}")
