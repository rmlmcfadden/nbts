#!/usr/bin/env python3
"""
Create a “ramp rate overview” figure in every experiment folder
and save it as  <subdir>/ramp_rate_overview.pdf
"""

from pathlib import Path
import re
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

BASE_DIR = Path("experiments/ramp_rate_comparison")
DEPTH_XLIM = (0, 150)  # set to None for auto‑scaling


# ─── helpers ────────────────────────────────────────────────────────────────
def find_ramp_rate(fname: str) -> str:
    if "const" in fname:
        return "const"
    m = re.search(r"_(\d+)C_min", fname)
    return f"{m.group(1)} °C/min" if m else fname


def read_profiles(subdir: Path, pattern: str):
    dfs = {}
    for csv in subdir.glob(pattern):
        lbl = find_ramp_rate(csv.name)
        df = pd.read_csv(csv)
        df.columns = ["axis", lbl]
        dfs[lbl] = df
    return dfs


def parse_T_and_t(dirname: str):
    m = re.match(r"(\d+)C_(\d+)h", dirname)
    return m.groups() if m else ("?", "?")


# ─── main plotting routine ──────────────────────────────────────────────────
def make_panel_plot(subdir: Path, depth_xlim=None):
    oxy = read_profiles(subdir, "oxygen_profile_*csv")
    temps = read_profiles(subdir, "temps_*csv")

    order = sorted(
        [k for k in oxy if k != "const"],
        key=lambda s: (s != "1 °C/min", int(re.findall(r"\d+", s)[0])),
    )
    order.append("const")

    T_C, t_h = parse_T_and_t(subdir.name)

    # Figure with constrained layout
    fig = plt.figure(figsize=(6, 8), constrained_layout=True)

    # GridSpec
    gs = gridspec.GridSpec(
        5, 1, height_ratios=[1, 1, 1, 0.05, 1], hspace=0.05, figure=fig
    )

    ax_oxy = fig.add_subplot(gs[0])
    ax_dconst = fig.add_subplot(gs[1], sharex=ax_oxy)
    ax_d1c = fig.add_subplot(gs[2], sharex=ax_oxy)
    fig.add_subplot(gs[3]).axis("off")  # thin spacer
    ax_temp = fig.add_subplot(gs[4])  # separate time axis

    # 0) Oxygen profiles
    for lbl in order:
        ax_oxy.plot(oxy[lbl]["axis"], oxy[lbl][lbl], label=lbl)
    ax_oxy.set_ylabel("[O] (at %)")
    ax_oxy.legend()

    # 1) Δ[O] vs const
    ref_const = oxy["const"].set_index("axis").squeeze()
    for lbl in order:
        if lbl == "const":
            continue
        diff = oxy[lbl].set_index("axis").squeeze() - ref_const
        ax_dconst.plot(diff.index, diff.values, label=f"{lbl} – const")
    ax_dconst.axhline(0, color="k", lw=0.5, ls="--")
    ax_dconst.set_ylabel("Δ[O] (U_ramp – U_const)")
    ax_dconst.legend(fontsize=8, ncol=2)

    # 2) Δ[O] vs 1 °C/min
    if "1 °C/min" in oxy:
        ref_1c = oxy["1 °C/min"].set_index("axis").squeeze()
        for lbl in order:
            if lbl in ("const", "1 °C/min"):
                continue
            diff = oxy[lbl].set_index("axis").squeeze() - ref_1c
            ax_d1c.plot(diff.index, diff.values, label=f"{lbl} – 1 °C/min")
    ax_d1c.axhline(0, color="k", lw=0.5, ls="--")
    ax_d1c.set_ylabel("Δ[O] rel 1 °C/min")
    ax_d1c.set_xlabel("Depth (nm)")
    ax_d1c.legend(fontsize=8, ncol=2)

    # Hide x‑tick labels on upper oxygen panels
    ax_oxy.tick_params(labelbottom=False)
    ax_dconst.tick_params(labelbottom=False)

    if depth_xlim:
        for ax in (ax_oxy, ax_dconst, ax_d1c):
            ax.set_xlim(depth_xlim)

    # 4) Temperature vs time
    for lbl in order:
        ax_temp.plot(temps[lbl]["axis"], temps[lbl][lbl], label=lbl)
    ax_temp.set_ylabel("Temperature (°C)")
    ax_temp.set_xlabel("Time (h)")
    ax_temp.legend(fontsize=8, ncol=2)

    # Add suptitle LAST so constrained_layout can account for it
    plt.suptitle(
        f"Ramp rate overview for T = {T_C} °C, t = {t_h} h", fontsize=14, y=1.05
    )  # 0.99 = slight padding below top edge

    # Save
    out_file = subdir / "ramp_rate_overview.pdf"
    fig.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out_file.relative_to(BASE_DIR)}")


# ─── batch driver ───────────────────────────────────────────────────────────
for folder in BASE_DIR.iterdir():
    if folder.is_dir() and folder.name.endswith("h") and "C_" in folder.name:
        try:
            make_panel_plot(folder, depth_xlim=DEPTH_XLIM)
        except Exception as e:
            print(f"[WARN] {folder.name}: {e}")
