#!/usr/bin/env python3
"""
Custom plotting for paper figures
──────────────────────────────────────────────────
* Re‑creates the three heat‑maps of CurrentDensityAnalyzer but lets you
  change any styling without affecting the main nbts pipeline.
* Adds plot_single_sim() → for single recipe plots.

"""

# ─── CONFIG SECTION ─────────────────────────────────────────────────────
BASE_DIR = "experiments/2025-11-05_const_605b3b7"  # path to the main experiment folder
RESULTS_DIR = "results"  # subfolder inside BASE_DIR
OUTPUT_DIR = "figures"  # created inside BASE_DIR/..
COLORMAP = "cividis"  # change freely
LEVELS_TILDE_J = 6  # contour levels for each heat‑map
LEVELS_TILDE_X = 6
LEVELS_TILDE_J_0 = 6
LEVELS_INTEGRAL_j = 6
LEVELS_J_DIFF = 6
DPI = None  # PDF/PNG output resolution
# ────────────────────────────────────────────────────────────────────────

from pathlib import Path
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import griddata, CubicSpline, UnivariateSpline
from scipy.ndimage import gaussian_filter
from scipy.signal import savgol_filter  # simple 1‑D smoother


# ─── helpers ------------------------------------------------------------
def parse_time_temp(dirname: str):
    """Extract numerical t and T from a folder name like 'sim_t6.0_T115'."""
    try:
        _, t_tag, T_tag = dirname.split("_")
        return float(t_tag[1:]), float(T_tag[1:])
    except ValueError:
        return None, None


def surface(df: pd.DataFrame, col: str):
    """Return value at x=0 (or the nearest available point)."""
    s = df.loc[df["x"] == 0, col]
    if s.empty:
        s = df.loc[[df["x"].idxmin()], col]
    return s.iloc[0]


# ─── main heat‑map routine ------------------------------------------------
def plot_overview_heatmaps(base_dir: Path, out_dir: Path, time, temp):
    folders = glob.glob(str(base_dir / "results" / "sim_t*_T*"))

    t_vals, T_vals, Z_A, Z_B, Z_C, Z_D, Z_E = [], [], [], [], [], [], []

    for f in folders:
        t, T = parse_time_temp(Path(f).name)
        if t is None:
            continue

        df_path = Path(f) / "data" / "current_density_corrected.csv"
        if not df_path.is_file():
            continue
        df_path2 = Path(f) / "data" / "critical_current_density.csv"
        if not df_path2.is_file():
            continue
        df = pd.read_csv(df_path, skipinitialspace=True)
        if not {"x", "current_density_corrected", "J_clean_corr"}.issubset(df.columns):
            continue
        df2 = pd.read_csv(df_path2, skipinitialspace=True)
        if not {"x", "critical_current_density"}.issubset(df2.columns):
            continue

        # --- sub‑grid peak on a cubic spline -----------------------------
        x = df["x"].to_numpy()
        J = df["current_density_corrected"].to_numpy()
        J_c = df2["critical_current_density"].to_numpy()
        spl = CubicSpline(x, J, bc_type="natural")
        fine_x = np.linspace(x.min(), x.max(), 40001)
        J_fine = spl(fine_x)
        idx_max = J_fine.argmax()
        x_peak = float(fine_x[idx_max])  # smooth peak position
        J_max = float(J_fine[idx_max])  # smooth peak value

        J_surf_clean = surface(df, "J_clean_corr")  # reference @ x = 0
        J_surf_smooth = float(spl(0.0))  # smooth J at surface

        ratioA = J_max / J_surf_clean
        ratioC = J_surf_smooth / J_surf_clean
        ratioD = J / J_c
        ratioD_integral = np.trapezoid(
            ratioD, x=x, dx=np.diff(x)[0]
        )  # integral of J / J_c
        ratioE = (J_surf_clean / J_c.max()) - (J_surf_smooth / J_c.min())

        t_vals.append(t)
        T_vals.append(T)
        Z_A.append(ratioA)
        Z_B.append(x_peak)
        Z_C.append(ratioC), Z_D.append(ratioD_integral), Z_E.append(ratioE)

    # --------------------------- remainder unchanged ----------------------
    times = np.unique(t_vals)
    temps = np.unique(T_vals)
    shape = (len(temps), len(times))
    A = np.full(shape, np.nan)
    B = np.full(shape, np.nan)
    C = np.full(shape, np.nan)
    D = np.full(shape, np.nan)
    E = np.full(shape, np.nan)

    for t, T, a, b, c, d, e in zip(t_vals, T_vals, Z_A, Z_B, Z_C, Z_D, Z_E):
        i = np.where(times == t)[0][0]
        j = np.where(temps == T)[0][0]
        A[j, i], B[j, i], C[j, i], D[j, i], E[j, i] = a, b, c, d, e

    heatmap_specs = [
        (
            "ratio_max_over_surface.pdf",
            A,
            # r"$\tilde{J} \equiv \max \{ J_\mathrm{baked}(x) \} \, / \, \max \{ J_\mathrm{clean}(x) \}$",
            r"$\tilde{J} \equiv \max \{ J(x) \} \, / \, \max \{ J_\mathrm{clean}(x) \}$",
            LEVELS_TILDE_J,
        ),
        (
            "x_peak_position.pdf",
            B,
            # r"$\tilde{x} \equiv \operatorname{arg\,max}_{x} \{ J_\mathrm{baked}(x) \}$",
            r"$\tilde{x} \equiv \operatorname{arg\,max}_{x} \{ J(x) \}$",
            LEVELS_TILDE_X,
        ),
        (
            "surface_current_ratio.pdf",
            C,
            # r"$\tilde{J}_0 \equiv J_\mathrm{baked}(x=0) \, / , \max J_\mathrm{clean}(x)$",
            r"$\tilde{J}_0 \equiv J(x=0) \, / \, \max \{ J_\mathrm{clean}(x) \}$",
            LEVELS_TILDE_J_0,
        ),
        (
            "integral_current_ratio.pdf",
            D,
            # r"$\int_{0}^{\infty} j_\mathrm{baked}(x) \, \mathrm{d}x$",
            r"$\int_{0}^{\infty} j(x) \, \mathrm{d}x$",
            LEVELS_INTEGRAL_j,
        ),
        (
            "surface_current_difference.pdf",
            E,
            # r"$j_\mathrm{diff} \equiv j_\mathrm{baked}(x=0) - j_\mathrm{clean}(x=0)$",
            r"$j_\mathrm{diff} \equiv j(x=0) - j_\mathrm{clean}(x=0)$",
            LEVELS_J_DIFF,
        ),
    ]

    ti = np.linspace(times.min(), times.max(), len(times) * 10)
    Ti = np.linspace(temps.min(), temps.max(), len(temps) * 10)
    Xd, Yd = np.meshgrid(ti, Ti)

    def valid_points(Z2d):
        jj, ii = np.nonzero(~np.isnan(Z2d))
        return np.column_stack([times[ii], temps[jj]]), Z2d[jj, ii]

    # --------------------------------------------------------------------------
    # helper: return only strictly positive data points ------------------------
    def positive_points(Z2d):
        jj, ii = np.nonzero(Z2d > 0)  # <-  ignore all 0‑cells
        return np.column_stack([times[ii], temps[jj]]), Z2d[jj, ii]

    # helper: pick the last positive cell in each time column ------------------
    def frontier_temps(Zcoarse):
        """return the (len(times),) array of frontier T for each t"""
        front = np.full(len(times), np.nan)
        for i, t in enumerate(times):
            col = Zcoarse[:, i]
            jj = np.where(col > 0)[0]
            if jj.size:
                front[i] = temps[jj[-1]]  # last positive = boundary
        return front

    # helper: last positive row (boundary) in every column ---------------------
    def boundary_T(Zcoarse):
        """frontier temperature for each time; NaN if column empty"""
        out = np.full(len(times), np.nan)
        for i, _ in enumerate(times):
            col = Zcoarse[:, i]
            pos = np.where(col > 0)[0]
            if pos.size:
                out[i] = temps[pos[-1]]
        return out

    EPS = 1e-3  # everything below this is considered “zero”
    SIG = 1.5  # Gaussian sigma in *fine* grid cells
    METHOD = "linear"  # safer than "cubic" for flat regions

    # # ── heat‑map loop -----------------------------------------------------------
    # for fname, Z2d, cbar_label, n_levels in heatmap_specs:

    #     if fname == "x_peak_position.pdf":
    #         # 1)  smooth the boundary itself (T vs t) ------------------------------
    #         Tbounds = frontier_temps(Z2d)                # NaN where no positive vals
    #         good    = ~np.isnan(Tbounds)
    #         t_good  = times[good]
    #         T_good  = Tbounds[good]

    #         # low‑order Savitzky–Golay to keep the trend but kill the stair‑steps
    #         T_smooth = savgol_filter(T_good, 9, 3)       # window=9 pts, poly=3

    #         # spline for interpolation onto the *fine* t‑grid
    #         spl_front = UnivariateSpline(t_good, T_smooth, s=1e-2)  # light smoothing
    #         T_front_fine = spl_front(ti)                            # shape (len(ti),)

    #         # 2)  build a fine‑grid mask from the smooth frontier ------------------
    #         mask_fine = (Yd <= T_front_fine)        # inside positive domain

    #         # 3)  interpolate *only* the positive coarse points --------------------
    #         jj, ii   = np.nonzero(Z2d > 0)
    #         pts      = np.column_stack([times[ii], temps[jj]])
    #         vals     = Z2d[jj, ii]
    #         Z_fine   = griddata(pts, vals, (Xd, Yd), method=METHOD)

    #         # outside the frontier → exactly 0
    #         Z_fine = np.where(mask_fine, Z_fine, 0.0)

    #         # 4)  optional interior blur, preserving zeros -------------------------
    #         pos_mask = (Z_fine > EPS).astype(float)
    #         num = gaussian_filter(Z_fine * pos_mask, SIG, mode='nearest')
    #         den = gaussian_filter(pos_mask,        SIG, mode='nearest')
    #         with np.errstate(invalid='ignore', divide='ignore'):
    #             Z_smooth = np.where(den > 0, num/den, 0.0)

    #         Z_smooth[Z_smooth < EPS] = 0.0          # ensure plateau is flat zero

    #         # 5)  contour levels ---------------------------------------------------
    #         dist_to_front = T_front_fine - Yd     # ≥0 inside, <0 outside
    #         boundary_level = 24
    #         interior_levels = [14, 19, 24]
    #         if isinstance(n_levels, int):
    #             zmin = np.nanmin(Z_smooth[Z_smooth > EPS])
    #             zmax = np.nanmax(Z_smooth)
    #             levels = np.linspace(zmin, zmax, n_levels)
    #         else:
    #             levels = np.asarray(n_levels)
    #         levels = levels[levels > EPS]

    #         # 6)  plot – your styling unchanged -----------------------------------
    #         fig, ax = plt.subplots(figsize=(4.8, 4.8), constrained_layout=True)

    #         mesh = ax.pcolormesh(ti, Ti, Z_smooth,
    #                             cmap=COLORMAP, shading='gouraud')
    #         # ‑‑ draw the boundary ONLY ONCE, thin solid ----------------------
    #         cs_bound = ax.contour(ti, Ti, Z_smooth,
    #                             levels=[boundary_level],
    #                             colors='w', linewidths=1.0, linestyles='solid')

    #         # ‑‑ safety strip so interior lines start away from the frontier --
    #         DELTA = 2.0                          # °C distance from the boundary
    #         interior_mask = np.where(dist_to_front > DELTA, Z_smooth, np.nan)

    #         # ‑‑ dotted interior contours ------------------------------------
    #         cs_int = ax.contour(ti, Ti, interior_mask,
    #                             levels=interior_levels,
    #                             colors='w', linewidths=0.8, linestyles='solid')  # dotted

    #         # label only those interior levels
    #         ax.clabel(cs_bound, fmt='%.0f', inline=False, fontsize=7, colors='white')
    #         ax.clabel(cs_int, fmt='%.0f', inline=True, fontsize=7, colors='white')

    #         cbar = fig.colorbar(mesh, ax=ax)
    #         cbar.set_label(cbar_label)

    #         # annotation – unchanged ----------------------------------------------
    #         x_lab = 0.25
    #         y_lab = 0.5 * (ax.get_position().y1 + 3)

    #         ax.scatter(time, temp, s=20, facecolor='red', lw=1.3, zorder=4)
    #         ax.annotate(
    #             rf'{time:.1f} h, {temp:.0f} °C',
    #             xy=(time + 0.5, temp + 1.5), xycoords='data',
    #             xytext=(x_lab, y_lab), textcoords='figure fraction',
    #             ha='center', va='center', fontsize=8,
    #             arrowprops=dict(arrowstyle='-', lw=0.8, color='red',
    #                             shrinkA=0, shrinkB=0),
    #             zorder=5
    #         )

    #         ax.set_xlabel(r'$t$ (h)')
    #         ax.set_ylabel(r'$T$ ($^{\circ}$C)')
    #         #fig.tight_layout()
    #         fig.savefig(out_dir / fname, dpi=DPI)
    #         plt.close(fig)

    for fname, Z2d, cbar_label, n_levels in heatmap_specs:
        if fname == "x_peak_position.pdf":
            # fname = fname.replace("x_peak_position", "x_peak_position_no_interp")
            # ── choose grid (no interpolation shown here) ──────────────────────
            Z_plot = Z2d
            X_plot, Y_plot = np.meshgrid(times, temps)

            # ── build figure WITHOUT constrained_layout ───────────────────────
            # fig, ax = plt.subplots(figsize=(4.8, 3.8), constrained_layout=True)   # ← no constrained_layout

            # ─────────────────────────────────────────────────────────────────────────
            # heat‑map
            # mesh = ax.pcolormesh(X_plot, Y_plot, Z_plot,
            #                     cmap=COLORMAP, shading='gouraud')

            # ─────────────────────────────────────────────────────────────────────────
            # 1) build a distance‑to‑boundary array on the COARSE grid
            #    (one value for every time column)
            # ------------------------------------------------------------------------
            # last positive row index (= boundary) in every column
            frontier_idx = np.full(len(times), -1, dtype=int)
            for i, _ in enumerate(times):
                col = Z_plot[:, i]
                pos = np.where(col > 0)[0]
                if pos.size:
                    frontier_idx[i] = pos[-1]  # last positive → boundary

            # temperature of the boundary for each column
            T_frontier = temps[frontier_idx]  # shape (len(times),)

            # broadcast to full 2‑D coarse mesh
            dist_to_front = T_frontier[None, :] - Y_plot  # ≥0 inside, <0 outside

            # ─────────────────────────────────────────────────────────────────────────
            # 2) contour levels
            # ------------------------------------------------------------------------
            boundary_levels = [EPS, 24]  # thin solid frontier
            interior_levels = [14, 19, 24]  # dotted interior contours

            # safety band width (°C): skip cells where 0 < d < Δ
            DELTA = 2.0

            # ── layout parameters ---------------------------------------
            FIG_W = FIG_H = 4.8  # square canvas (hard width cap)
            LEFT = 0.08  # left margin
            RIGHT = 0.01  # right margin
            TOP = 0.10  # white band reserved for annotation
            BOT = 0.05  # bottom margin

            CB_W = 0.045  # colour-bar slice
            CB_PAD = 0.015  # gap between map & bar

            sq_w = 1 - LEFT - CB_PAD - CB_W - RIGHT  # square width fraction
            sq_h = 1 - TOP - BOT  # square height fraction
            sq_h = min(sq_h, sq_w)  # ensure square fits
            sq_w = sq_h

            """
            fig = plt.figure(figsize=(FIG_W, FIG_H), dpi=DPI)

            # square heat-map axes anchored just under the top band
            ax = fig.add_axes(
                [
                    LEFT,
                    BOT,
                    sq_w,
                    sq_h,
                ]
            )

            # colour-bar axes matching the map height
            cax = fig.add_axes(
                [
                    LEFT + sq_w + CB_PAD,
                    BOT,
                    CB_W,
                    sq_h,
                ]
            )
            """

            fig, ax = plt.subplots(
                1,
                1,
                figsize=(FIG_W, FIG_H),
                constrained_layout=True,
            )

            # ── heat-map and contours -----------------------------------
            mesh = ax.pcolormesh(
                X_plot, Y_plot, Z_plot, cmap=COLORMAP, shading="gouraud"
            )
            # ax.set_aspect("equal", adjustable="box")

            # cs = ax.contour(X_plot, Y_plot, Z_plot,
            #                 levels=n_levels, colors='w', linewidths=0.8)
            # ax.clabel(cs, inline=True, fontsize=7)

            # ─────────────────────────────────────────────────────────────────────────
            # 3) draw the contours
            # ------------------------------------------------------------------------
            # boundary – draw once, thin solid
            cs_bound = ax.contour(
                X_plot,
                Y_plot,
                Z_plot,
                levels=boundary_levels,
                colors="w",
                linewidths=0.8,
                linestyles="solid",
            )

            # interior – mask out the Δ‑band so they start away from the frontier
            if interior_levels:
                Z_masked = np.where(dist_to_front > DELTA, Z_plot, np.nan)
                cs_int = ax.contour(
                    X_plot,
                    Y_plot,
                    Z_masked,
                    levels=interior_levels,
                    colors="w",
                    linewidths=0.8,
                    linestyles="solid",
                )
                ax.clabel(cs_int, inline=True, fontsize=7)

            # ── colour-bar ----------------------------------------------
            # cbar = fig.colorbar(mesh, cax=cax)
            cbar = fig.colorbar(mesh, ax=ax)

            # place label on the *outside* of the bar
            cbar.set_label(
                cbar_label,
                rotation=90,  # downward reading
                labelpad=25,  # push rightwards
                va="bottom",
                ha="center",
                fontsize=12,
            )

            # ── recipe marker -------------------------------------------
            ax.scatter(time, temp, s=20, facecolor="red", lw=1.3, zorder=4)

            # ── annotation in the dedicated top band --------------------
            # height of white band above the square map
            pad_top = 1 - (BOT + sq_h)  #  = unused top fraction

            # HEAD_Y = BOT + sq_h + pad_top / 2 + 0.02  # vertical centre of that band
            HEAD_Y = 1.05

            ax.annotate(
                # rf"{time:.1f} h, {temp:.0f} °C",
                rf"$t = {time:.1f}$ h, $T = {temp:.0f}$ °C",
                xy=(time + 0.5, temp + 1.5),
                xycoords="data",  # arrow head
                xytext=(0.5, HEAD_Y),
                # textcoords="figure fraction",
                textcoords="axes fraction",
                ha="center",
                va="center",
                # fontsize=9,  # now perfectly centred
                fontsize="small",
                color="red",
                arrowprops=dict(
                    arrowstyle="-", lw=0.8, color="red", shrinkA=0, shrinkB=0
                ),
                zorder=5,
            )

            # ── axis labels & save --------------------------------------
            ax.set_xlabel(r"$t$ (h)")
            ax.set_ylabel(r"$T$ ($^{\circ}$C)")

            fig.savefig(
                out_dir / fname,
                dpi=DPI,
                # bbox_inches="tight",
                # pad_inches=0.02,
            )
            plt.close(fig)

    for fname, Z2d, cbar_label, n_levels in heatmap_specs:
        if fname != "x_peak_position.pdf":
            # ── data grid ------------------------------------------------
            Z_plot = Z2d
            X_plot, Y_plot = np.meshgrid(times, temps)

            # ── layout parameters ---------------------------------------
            FIG_W = FIG_H = 4.8  # square canvas (hard width cap)
            LEFT = 0.08  # left margin
            RIGHT = 0.01  # right margin
            TOP = 0.10  # white band reserved for annotation
            BOT = 0.05  # bottom margin

            CB_W = 0.045  # colour-bar slice
            CB_PAD = 0.015  # gap between map & bar

            sq_w = 1 - LEFT - CB_PAD - CB_W - RIGHT  # square width fraction
            sq_h = 1 - TOP - BOT  # square height fraction
            sq_h = min(sq_h, sq_w)  # ensure square fits
            sq_w = sq_h

            """
            fig = plt.figure(figsize=(FIG_W, FIG_H), dpi=DPI)

            # square heat-map axes anchored just under the top band
            ax = fig.add_axes(
                [
                    LEFT,
                    BOT,
                    sq_w,
                    sq_h,
                ]
            )

            # colour-bar axes matching the map height
            cax = fig.add_axes(
                [
                    LEFT + sq_w + CB_PAD,
                    BOT,
                    CB_W,
                    sq_h,
                ]
            )
            """

            fig, ax = plt.subplots(
                1,
                1,
                figsize=(FIG_W, FIG_H),
                constrained_layout=True,
            )

            # ── heat-map and contours -----------------------------------
            mesh = ax.pcolormesh(
                X_plot, Y_plot, Z_plot, cmap=COLORMAP, shading="gouraud"
            )
            # ax.set_aspect("equal", adjustable="box")

            cs = ax.contour(
                X_plot, Y_plot, Z_plot, levels=n_levels, colors="w", linewidths=0.8
            )
            ax.clabel(cs, inline=True, fontsize=7)

            # ── colour-bar ----------------------------------------------
            # cbar = fig.colorbar(mesh, cax=cax)
            cbar = fig.colorbar(mesh, ax=ax)

            # place label on the *outside* of the bar
            cbar.set_label(
                cbar_label,
                rotation=90,  # downward reading
                labelpad=33,  # push rightwards
                va="bottom",
                ha="center",
                fontsize=12,
            )

            # ── recipe marker -------------------------------------------
            ax.scatter(time, temp, s=20, facecolor="red", lw=1.3, zorder=4)

            # ── annotation in the dedicated top band --------------------
            # height of white band above the square map
            pad_top = 1 - (BOT + sq_h)  #  = unused top fraction

            # HEAD_Y = BOT + sq_h + pad_top / 2 + 0.02  # vertical centre of that band
            HEAD_Y = 1.05

            ax.annotate(
                # rf"{time:.1f} h, {temp:.0f} °C",
                rf"$t = {time:.1f}$ h, $T = {temp:.0f}$ °C",
                xy=(time + 0.5, temp + 1.5),
                xycoords="data",  # arrow head
                xytext=(0.5, HEAD_Y),
                # textcoords="figure fraction",
                textcoords="axes fraction",
                ha="center",
                va="center",
                # fontsize=9,  # now perfectly centred
                fontsize="small",
                color="red",
                arrowprops=dict(
                    arrowstyle="-", lw=0.8, color="red", shrinkA=0, shrinkB=0
                ),
                zorder=5,
            )

            # ── axis labels & save --------------------------------------
            ax.set_xlabel(r"$t$ (h)")
            ax.set_ylabel(r"$T$ ($^{\circ}$C)")

            fig.savefig(
                out_dir / fname,
                dpi=DPI,
                # bbox_inches="tight",
                # pad_inches=0.02,
            )
            plt.close(fig)


# ─── running block ------------------------------------------------------
def main():
    base = Path(BASE_DIR)
    out = base / OUTPUT_DIR
    out.mkdir(exist_ok=True)

    # overview heat‑maps
    plot_overview_heatmaps(base, out, time=8.0, temp=125)
    print(f"Heat‑maps written → {out}")


if __name__ == "__main__":
    main()
