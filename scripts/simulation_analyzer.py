# nbts/analysis/simulation_analyzer.py
from __future__ import annotations
from abc import ABC, abstractmethod
from pathlib import Path
from dataclasses import dataclass
import glob

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ────────────────────────────────────────────────────────────────────────
# Shared scaffolding
# ────────────────────────────────────────────────────────────────────────
@dataclass
class Metric:
    """Everything needed to render one heat‑map plot."""

    z_grid: np.ndarray
    cmap: str
    levels: int
    cbar_label: str
    fname: str


class SimulationAnalyzer(ABC):
    """
    Base class: concrete subclasses implement `_collect()`, then call `run()`.
    """

    def __init__(self, base_dir: str, ox_dir: str | None = None) -> None:
        self.base_dir = Path(base_dir)
        self.ox_dir = Path(ox_dir) if ox_dir else None
        self.analysis_dir = (
            self.base_dir.parent / f"analysis_{self.ox_dir.name}"
            if self.ox_dir
            else self.base_dir.parent / "analysis"
        )
        self.analysis_dir.mkdir(exist_ok=True)

        self.times: np.ndarray | None = None
        self.temps: np.ndarray | None = None
        self.metrics: dict[str, Metric] = {}

    # public entry point ---------------------------------------------------
    def run(self) -> None:
        self._collect()
        self._make_plots()
        print(f"{self.__class__.__name__}: analysis complete → {self.analysis_dir}")

    # to be supplied by subclass ------------------------------------------
    @abstractmethod
    def _collect(self) -> None: ...

    # common plot loop -----------------------------------------------------
    def _make_plots(self) -> None:
        X, Y = np.meshgrid(self.times, self.temps)

        for m in self.metrics.values():
            fig, ax = plt.subplots(figsize=(8, 6))
            mesh = ax.pcolormesh(X, Y, m.z_grid, shading="gouraud", cmap=m.cmap)

            cbar = fig.colorbar(mesh, ax=ax)
            cbar.set_label(m.cbar_label, fontsize=12)

            cs = ax.contour(
                X, Y, m.z_grid, levels=m.levels, colors="white", linewidths=1
            )
            ax.clabel(cs, inline=True, fontsize=8)

            ax.set_xlabel("Time (h)", fontsize=14)
            ax.set_ylabel("Temperature (°C)", fontsize=14)
            fig.tight_layout()
            fig.savefig(self.analysis_dir / m.fname)
            plt.close(fig)


# ────────────────────────────────────────────────────────────────────────
# Current‑density overview (concrete subclass)
# ────────────────────────────────────────────────────────────────────────
class CurrentDensityAnalyzer(SimulationAnalyzer):

    # helpers --------------------------------------------------------------
    @staticmethod
    def _parse_time_temp(name: str) -> tuple[float | None, float | None]:
        try:
            _, t_tag, T_tag = name.split("_")
            return float(t_tag[1:]), float(T_tag[1:])
        except (ValueError, IndexError):
            return None, None

    @staticmethod
    def _surface(df: pd.DataFrame, col: str) -> float:
        s = df.loc[df["x"] == 0, col]
        if s.empty:
            s = df.loc[[df["x"].idxmin()], col]
        return s.iloc[0]

    # core ---------------------------------------------------------------
    def _collect(self) -> None:
        folders = glob.glob(str(self.base_dir / "sim_t*_T*"))

        t_all, T_all, A_all, B_all, C_all = ([] for _ in range(5))

        for f in folders:
            t_val, T_val = self._parse_time_temp(Path(f).name)
            if t_val is None:
                continue

            csv = Path(f) / "data" / "current_density_corrected.csv"
            if not csv.is_file():
                continue

            df = pd.read_csv(csv, skipinitialspace=True)
            if not {"x", "current_density_corrected", "J_clean_corr"}.issubset(
                df.columns
            ):
                continue

            # Metric A
            J_max = df["current_density_corrected"].max()
            J_surface = self._surface(df, "J_clean_corr")
            ratio_A = J_max / J_surface

            # Metric B
            x_peak = df.loc[df["current_density_corrected"].idxmax(), "x"]

            # Metric C
            J_max_surf = self._surface(df, "current_density_corrected")
            ratio_C = J_max_surf / J_surface

            t_all.append(t_val)
            T_all.append(T_val)
            A_all.append(ratio_A)
            B_all.append(x_peak)
            C_all.append(ratio_C)

        # turn lists into 2‑D grids --------------------------------------
        self.times = np.unique(t_all)
        self.temps = np.unique(T_all)
        shape = (len(self.temps), len(self.times))
        Z_A, Z_B, Z_C = (np.full(shape, np.nan) for _ in range(3))

        for t, T, a, b, c in zip(t_all, T_all, A_all, B_all, C_all):
            i = np.where(self.times == t)[0][0]
            j = np.where(self.temps == T)[0][0]
            Z_A[j, i], Z_B[j, i], Z_C[j, i] = a, b, c

        # register the three plots
        self.metrics = {
            "ratio_A": Metric(
                Z_A,
                "cividis",
                10,
                "maximum J(x) / maximum supercurrent in clean Nb",
                "ratio_max_over_surface.pdf",
            ),
            "x_peak": Metric(
                Z_B,
                "cividis",
                3,
                "x position of supercurrent density peak",
                "x_peak_position.pdf",
            ),
            "ratio_C": Metric(
                Z_C,
                "cividis",
                10,
                "J(x) surface value / maximum supercurrent in clean Nb",
                "surface_current_ratio.pdf",
            ),
        }


# ────────────────────────────────────────────────────────────────────────
class MultiOxidationCurrentDensityAnalyzer(SimulationAnalyzer):
    """
    Collect the same three metrics (A, B, C) from several oxidization
    passes that already live in sibling directories like

        results_oxidized_1/
        results_oxidized_2/
        results_oxidized/        # final pass

    and render one figure per metric with N sub‑plots (one per pass,
    common colour‑scale).
    """

    # ─── helpers (pulled verbatim from CurrentDensityAnalyzer) ───────────
    _parse_time_temp = staticmethod(CurrentDensityAnalyzer._parse_time_temp)
    _surface = staticmethod(CurrentDensityAnalyzer._surface)

    # ─── construction ───────────────────────────────────────────────────
    def __init__(
        self,
        parent_dir: str | Path,
        n_passes: int | None = None,  # autodetect if None
        step_glob: str = "results_reoxidize*",
    ) -> None:

        self.parent_dir = Path(parent_dir)
        # collect the folders in *chronological* order 1, 2, … ,final
        step_dirs = sorted(
            self.parent_dir.glob(step_glob), key=lambda p: (p.suffix == "", p.name)
        )

        if not step_dirs:
            raise FileNotFoundError(
                f"No oxidization folders matching "
                f"'{step_glob}' found in {parent_dir}"
            )

        if n_passes is not None:
            step_dirs = step_dirs[:n_passes]

        self.step_dirs = step_dirs
        self.n_passes = len(self.step_dirs)

        # the superclass needs *some* base_dir, but we override _collect & _make_plots
        super().__init__(base_dir=str(self.step_dirs[0]))

        # one analysis folder for the bundle
        self.analysis_dir = self.parent_dir / f"analysis_multi_step"
        self.analysis_dir.mkdir(exist_ok=True)

    # ─── data collection ────────────────────────────────────────────────
    def _collect(self) -> None:
        """
        Build, for every metric, a 3‑D tensor   (pass, T, t)
        plus a shared `self.times` / `self.temps`.
        """

        time_set, temp_set = set(), set()
        for d in self.step_dirs:
            for f in d.glob("sim_t*_T*"):
                t, T = self._parse_time_temp(f.name)
                if t is not None:
                    time_set.add(t)
                    temp_set.add(T)

        self.times = np.array(sorted(time_set))
        self.temps = np.array(sorted(temp_set))
        shape2d = (len(self.temps), len(self.times))
        shape3d = (self.n_passes, *shape2d)

        Z_A = np.full(shape3d, np.nan)
        Z_B = np.full(shape3d, np.nan)
        Z_C = np.full(shape3d, np.nan)

        # second pass: fill the tensors ----------------------------------
        for p_idx, step in enumerate(self.step_dirs):
            folders = glob.glob(str(step / "sim_t*_T*"))
            for f in folders:
                t, T = self._parse_time_temp(Path(f).name)
                if t is None:
                    continue

                csv = Path(f) / "data" / "current_density_corrected.csv"
                if not csv.is_file():
                    continue

                df = pd.read_csv(csv, skipinitialspace=True)
                if not {"x", "current_density_corrected", "J_clean_corr"}.issubset(
                    df.columns
                ):
                    continue

                J_max = df["current_density_corrected"].max()
                J_surface = self._surface(df, "J_clean_corr")
                ratio_A = J_max / J_surface

                x_peak = df.loc[df["current_density_corrected"].idxmax(), "x"]

                J_max_surf = self._surface(df, "current_density_corrected")
                ratio_C = J_max_surf / J_surface

                i = np.where(self.times == t)[0][0]
                j = np.where(self.temps == T)[0][0]
                Z_A[p_idx, j, i] = ratio_A
                Z_B[p_idx, j, i] = x_peak
                Z_C[p_idx, j, i] = ratio_C

        # pack everything into self.metrics as lists ---------------------
        self.metrics = {
            "max_current": (Z_A, "maximum J(x) / J_clean"),
            "x_peak": (Z_B, "x‑position of J(x) peak"),
            "surface_current": (Z_C, "J(surface) / J_clean"),
        }

    # ─── plotting ───────────────────────────────────────────────────────
    def _make_plots(self) -> None:
        X, Y = np.meshgrid(self.times, self.temps)

        for key, (Z_stack, cbar_label) in self.metrics.items():
            fig, axes = plt.subplots(
                1,
                self.n_passes,
                figsize=(4 * self.n_passes, 5),
                sharex=True,
                sharey=True,
                constrained_layout=True,
            )

            # Wrap single‑axes case so later code can treat it uniformly
            axes_arr = np.atleast_1d(axes)  # → ndarray of Axes
            axes_list = axes_arr.ravel().tolist()

            vmin, vmax = np.nanmin(Z_stack), np.nanmax(Z_stack)

            for idx, ax in enumerate(axes_arr):
                mesh = ax.pcolormesh(
                    X,
                    Y,
                    Z_stack[idx],
                    shading="gouraud",
                    cmap="cividis",
                    vmin=vmin,
                    vmax=vmax,
                )
                cs = ax.contour(
                    X,
                    Y,
                    Z_stack[idx],
                    levels=10,
                    colors="white",
                    linewidths=1,
                    vmin=vmin,
                    vmax=vmax,
                )
                ax.clabel(cs, inline=True, fontsize=8)
                ax.set_title(f"Pass {idx+1}")
                ax.set_xlabel("Time (h)")
                if idx == 0:
                    ax.set_ylabel("Temperature (°C)")

            # one colour bar spanning every subplot
            cbar = fig.colorbar(mesh, ax=axes_list)
            cbar.set_label(cbar_label)

            fig.savefig(self.analysis_dir / f"{key}_multi_pass.pdf")
            plt.close(fig)
