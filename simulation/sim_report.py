from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from simulation.gle_solver import GLESolver
from simulation.quantities import ell, lambda_eff, lambda_eff_corr, J, B, J_c


class GenSimReport:
    """
    Modular report class for SRF simulation outputs.

    Usage:
        report = GenSimReport(x, o_total, t, T)
        report.compute()  # compute profiles
        fig = report.plot_overview()
    """

    # -----------------------------------------------------------------
    #  Class‑level schema:  key → {new_col:  lambda self → 1‑D array}
    # -----------------------------------------------------------------
    EXTRA_COLS = {
        "current_density": {
            "J_clean": lambda self: self.J_clean,
            "J_dirty": lambda self: self.J_dirty,
        },
        "current_density_corrected": {
            "J_clean_corr": lambda self: self.J_clean_corr,
            "J_dirty_corr": lambda self: self.J_dirty_corr,
        },
        "screening_profile": {
            "B_clean": lambda self: self.B_clean,
            "B_dirty": lambda self: self.B_dirty,
        },
        "screening_profile_corrected": {
            "B_clean_corr": lambda self: self.B_clean_corr,
            "B_dirty_corr": lambda self: self.B_dirty_corr,
        },
    }

    def __init__(
        self, cfg, x, o_total, t, T, time_h, temps_K, profile, output_dir="sim_output"
    ):
        # raw inputs
        self.cfg = cfg
        self.x = np.asarray(x)
        self.o_total = np.asarray(o_total)
        self.time_h = time_h
        self.temps_K = temps_K
        self.profile = profile
        self.t = t
        self.T = T
        # TODO: fix lambda_0 to be consistent naming with lambda_L
        self.lambda_0 = 27  # nm: clean-limit penetration depth
        # placeholders for computed arrays
        self.ell_val = None
        self.lambda_eff_val = None
        self.lambda_eff_val_corr = None
        self.screening_profile = None
        self.screening_profile_corr = None
        self.current_density = None
        self.current_density_corr = None
        self.B_clean = None
        self.B_dirty = None
        self.J_clean = None
        self.J_dirty = None
        self.B_clean_corr = None
        self.B_dirty_corr = None
        self.J_clean_corr = None
        self.J_dirty_corr = None
        self.J_c = None
        self.H0 = cfg.args.applied_field_mT
        self.dead_layer = cfg.args.dead_layer_nm
        self.demag_factor = cfg.args.demag_factor
        # factors
        self.suppression_factor = None
        self.enhancement_factor = None
        self.current_suppression_factor = None
        # output directory and COMPUTE flag
        self.output_dir = Path(output_dir)
        self.COMPUTE = False

    def compute(self):
        """Compute all physics profiles from the oxygen concentration profile."""
        # mean-free-path and penetration depths
        self.ell_val = ell(self.o_total)
        self.lambda_eff_val = lambda_eff(self.ell_val)
        self.lambda_eff_val_corr = lambda_eff_corr(self.cfg, self.lambda_eff_val)

        # GLE solver for screening & current density
        gle = GLESolver(self.x, self.lambda_eff_val)
        gle_corr = GLESolver(self.x, self.lambda_eff_val_corr)

        args = (self.H0, self.dead_layer, self.demag_factor)
        self.screening_profile = gle.screening_profile(self.x, *args)
        self.screening_profile_corr = gle_corr.screening_profile(self.x, *args)
        self.current_density = gle.current_density(self.x, *args)
        self.current_density_corr = gle_corr.current_density(self.x, *args)

        # reference B and J extrema via analytical formulas
        self.B_dirty = B(self.x, self.H0, self.lambda_eff_val.max())  # max B
        self.B_clean = B(self.x, self.H0, self.lambda_eff_val.min())  # min B
        self.J_dirty = J(self.x, self.H0, self.lambda_eff_val.max())  # max J
        self.J_clean = J(self.x, self.H0, self.lambda_eff_val.min())  # min J

        self.B_dirty_corr = B(
            self.x, self.H0, self.lambda_eff_val_corr.max()
        )  # max B (corr)
        self.B_clean_corr = B(
            self.x, self.H0, self.lambda_eff_val_corr.min()
        )  # min B (corr)
        self.J_dirty_corr = J(
            self.x, self.H0, self.lambda_eff_val_corr.max()
        )  # max J (corr)
        self.J_clean_corr = J(
            self.x, self.H0, self.lambda_eff_val_corr.min()
        )  # min J (corr)

        self.J_c = J_c(self.lambda_eff_val_corr)

        # derived factors
        J0 = self.J_clean_corr[0]
        num = self.lambda_0 * J0
        den = np.clip(self.lambda_eff_val_corr * self.current_density_corr, 1e-10, None)
        self.suppression_factor = num / den
        self.enhancement_factor = self.lambda_eff_val_corr / self.lambda_0
        self.current_suppression_factor = self.current_density_corr / self.J_clean_corr

        self.COMPUTE = True

    def _make_folder(self):
        folder = self.output_dir / f"sim_t{self.t:.1f}_T{self.T-273.15:.1f}"
        folder.mkdir(parents=True, exist_ok=True)
        return folder

    def generate(self):
        """Generate set of desired plots from computed quantities."""
        self.plot_overview()
        self.plot_current_density()
        self.plot_suppression_factor()
        self.plot_temp_profile()
        self.plot_suppression_factor_comparison()

    # -----------------------------------------------------------------
    # 2.  Generic writer
    # -----------------------------------------------------------------
    def _to_dataframe(self, key, arr):
        """Return a DataFrame with the right index/extra columns."""
        arr = np.asarray(arr)

        # baseline columns
        if key == "temperature_profile":  # time series
            df = pd.DataFrame({"time": self.time_h, key: arr})
        elif arr.ndim == 1:  # spatial profile
            df = pd.DataFrame({"x": self.x, key: arr})
        else:  # 2‑D array
            cols = {f"col{i}": arr[:, i] for i in range(arr.shape[1])}
            df = pd.DataFrame(cols)

        # inject any extras declared in EXTRA_COLS
        for col, fn in self.EXTRA_COLS.get(key, {}).items():
            df[col] = fn(self)

        return df

    def save_report(self, fig, data_dict, tag):
        """
        Save figure + data arrays.

        * plots/{tag}.pdf
        * data/{key}.csv
        """
        folder = self._make_folder()
        plots_folder = folder / "plots"
        data_folder = folder / "data"
        plots_folder.mkdir(parents=True, exist_ok=True)
        data_folder.mkdir(parents=True, exist_ok=True)

        # ---- figure ---------------------------------------------------
        fig.savefig(plots_folder / f"{tag}.pdf")
        plt.close(fig)

        # ---- data -----------------------------------------------------
        for key, arr in data_dict.items():
            df = self._to_dataframe(key, arr)
            df.to_csv(data_folder / f"{key}.csv", index=False)

    # -- Single-quantity plotters, overview quantities --
    def plot_oxygen(self, ax):
        ax.plot(self.x, self.o_total, "-", zorder=2, label="Oxygen Concentration")
        ax.set_ylabel(r"$[\mathrm{O}]$ (at. %)")
        ax.legend()

    def plot_mean_free_path(self, ax):
        (line,) = ax.plot(
            self.x, self.ell_val, "-", zorder=1, label="Electron Mean-free-path"
        )
        hmin = ax.axhline(self.ell_val.min(), linestyle=":", label=r"Min $\ell$")
        hmax = ax.axhline(self.ell_val.max(), linestyle=":", label=r"Max $\ell$")
        ax.set_ylabel(r"$\ell$ (nm)")
        ax.set_ylim(0, None)
        ax.legend(handles=[line, hmin, hmax])

    def plot_penetration_depths(self, ax):
        (l1,) = ax.plot(self.x, self.lambda_eff_val, "-", label="Penetration depth")
        (l2,) = ax.plot(
            self.x, self.lambda_eff_val_corr, "-", label="Corrected penetration depth"
        )
        ax.axhline(
            self.lambda_eff_val.min(), linestyle=":", label=r"Min $\lambda_{eff}$"
        )
        ax.axhline(
            self.lambda_eff_val.max(), linestyle=":", label=r"Max $\lambda_{eff}$"
        )
        ax.axhline(
            self.lambda_eff_val_corr.max(),
            linestyle=":",
            label=r"Max $\lambda_{eff}$ (corr)",
        )
        ax.set_ylabel(r"$\lambda_{eff}$ (nm)")
        ax.legend(handles=[l1, l2])

    def plot_screening(self, ax):
        ax.plot(
            self.x, self.screening_profile, "-", zorder=1, label="Screening profile"
        )
        ax.plot(
            self.x,
            self.screening_profile_corr,
            "-",
            zorder=1,
            label="Corrected screening profile",
        )
        ax.plot(self.x, self.B_dirty, ":", zorder=0, label="B(x) dirty")
        ax.plot(self.x, self.B_clean, ":", zorder=0, label="B(x) clean")
        ax.set_ylabel(r"$B(x)$ (G)")
        ax.set_ylim(0, None)
        ax.legend()

    def plot_current(self, ax):
        ax.plot(
            self.x, self.current_density / 1e11, "-", zorder=1, label="Current density"
        )
        ax.plot(
            self.x,
            self.current_density_corr / 1e11,
            "-",
            zorder=1,
            label="Corrected current density",
        )
        ax.plot(self.x, self.J_dirty / 1e11, ":", zorder=0, label="J(x) dirty")
        ax.plot(self.x, self.J_clean / 1e11, ":", zorder=0, label="J(x) clean")
        ax.set_ylabel(r"$J(x)$ ($10^{11}$ A m$^{-2}$)")
        ax.set_ylim(0, None)
        ax.legend()

    def plot_critical_current(self, ax):
        ax.plot(self.x, self.J_c / 1e11, "-", label="Critical current density")
        ax.set_ylabel(r"$J_c(x)$ ($10^{11}$ A m$^{-2}$)")
        ax.set_ylim(0, None)
        ax.legend()

    def plot_temp(self, ax):
        ax.plot(self.time_h, self.temps_K - 273.15, label=self.profile)
        ax.set_xlabel("Time (h)")
        ax.set_ylabel("Temperature (°C)")
        ax2 = ax.secondary_yaxis(
            "right", functions=(lambda x: x + 273.15, lambda x: x - 273.15)
        )
        ax2.set_ylabel("Temperature (K)")
        ax.legend(loc="best")

    # -- Single-quantity suppression/enhancement plotters --
    def plot_suppression(self, ax, invert=False):
        if invert:
            ax.plot(
                self.x,
                1 / self.suppression_factor,
                "-",
                label="Suppression factor (inverse)",
            )
        else:
            ax.plot(self.x, self.suppression_factor, "-", label="Suppression factor")
        ax.set_ylabel("Suppression factor")
        ax.legend()

    def plot_enhancement(self, ax):
        ax.plot(
            self.x, self.enhancement_factor, "-", label="Penetration depth enhancement"
        )
        ax.set_ylabel(r"$\lambda(x)/\lambda_{clean}$")
        ax.legend()

    def plot_current_suppression(self, ax):
        ax.plot(
            self.x,
            self.current_suppression_factor,
            "-",
            label="Current density suppression",
        )
        ax.set_ylabel(r"$J(x)/J_{clean}(x)$")
        ax.set_xlabel(r"$x$ (nm)")
        ax.legend()

    # TODO: rename to plot_ratio
    def plot_ratio(self, ax):
        ratio = self.enhancement_factor * self.current_suppression_factor
        ax.plot(self.x, ratio, "-", label="Ratio of enhancement and suppression")
        ax.set_ylabel(r"$\lambda(x)/\lambda_{clean} \cdot J(x)/J_{clean}(x)$")
        ax.set_xlabel(r"$x$ (nm)")
        ax.legend()

    # -- Assemble plots --
    def plot_overview(self):
        if not self.COMPUTE:
            self.compute()
        fig, axes = plt.subplots(
            5,
            1,
            sharex=True,
            sharey=False,
            figsize=(4.8, 6.4 + 0.5 * 3.2),
            constrained_layout=True,
        )
        # panels
        self.plot_oxygen(axes[0])
        self.plot_mean_free_path(axes[1])
        self.plot_penetration_depths(axes[2])
        self.plot_screening(axes[3])
        self.plot_current(axes[4])
        # final formatting
        axes[-1].set_xlabel(r"$x$ (nm)")
        axes[-1].set_xlim(0, 150)
        plt.suptitle(
            f"Simulation overview for T = {self.T-273.15:.1f} C and t = {self.t:.1f} h"
        )
        # save data
        data = {
            "oxygen_diffusion_profile": self.o_total,
            "mean_free_path": self.ell_val,
            "penetration_depth": self.lambda_eff_val,
            "penetration_depth_corrected": self.lambda_eff_val_corr,
            "screening_profile": self.screening_profile,
            "screening_profile_corrected": self.screening_profile_corr,
            "current_density": self.current_density,
            "current_density_corrected": self.current_density_corr,
        }
        self.save_report(fig, data, tag="overview")
        return fig

    def plot_current_density(self):
        if not self.COMPUTE:
            self.compute()
        fig, axes = plt.subplots(
            2, 1, sharex=True, figsize=(4.8, 4.8), constrained_layout=True
        )
        self.plot_current(axes[0])
        self.plot_critical_current(axes[1])
        axes[-1].set_xlim(0, 150)
        plt.suptitle(
            f"Current density profiles for T = {self.T-273.15:.1f} C and t = {self.t:.1f} h"
        )
        data = {"critical_current_density": self.J_c}
        self.save_report(fig, data, tag="critical_current_density")
        return fig

    def plot_suppression_factor_comparison(self):
        if not self.COMPUTE:
            self.compute()
        fig, axes = plt.subplots(
            4, 1, sharex=True, figsize=(5, 8), constrained_layout=True
        )
        self.plot_suppression(axes[0], True)
        self.plot_enhancement(axes[1])
        self.plot_current_suppression(axes[2])
        self.plot_ratio(axes[3])
        axes[-1].set_xlim(0, 150)
        plt.suptitle(
            f"Simulation suppression factor for T = {self.T-273.15:.1f} C and t = {self.t:.1f} h"
        )
        data = {
            "suppression_factor": self.suppression_factor,
            "enhancement_factor": self.enhancement_factor,
            "current_suppression_factor": self.current_suppression_factor,
        }
        self.save_report(fig, data, tag="suppression_factor_comparison")
        return fig

    def plot_suppression_factor(self):
        """Single-panel suppression-factor vs x."""
        if not self.COMPUTE:
            self.compute()
        fig, ax = plt.subplots()
        self.plot_suppression(ax)
        ax.set_ylim(0, 5)
        ax.set_xlim(0, 40)
        ax.set_xlabel(r"$x$ (nm)")
        ax.set_title(
            f"Suppression factor at T={self.T-273.15:.1f}\u00b0C, t={self.t:.1f}h"
        )
        data = {"suppression_factor": self.suppression_factor}
        self.save_report(fig, data, tag="suppression_factor")
        return fig

    def plot_temp_profile(self):
        """Single-panel temperature profile."""
        fig, ax = plt.subplots()
        self.plot_temp(ax)
        ax.set_ylim(self.cfg.temp_profile.start_C, self.temps_K.max() - 273.15 + 10)
        ax.set_xlim(0, self.time_h.max() + 1)
        ax.set_title(
            f"{self.profile} temperature profile at T={self.T-273.15:.1f}\u00b0C, t={self.t:.1f}h,"
            f" ramp={self.cfg.temp_profile.ramp_rate_C_per_min:.1f}\u00b0C/min"
        )
        plt.tight_layout()
        data = {"temperature_profile": self.temps_K}
        self.save_report(fig, data, tag="temperature_profile")
        return fig
