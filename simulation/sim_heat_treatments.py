#!/usr/bin/env python3
"""
sim_heat_treatments.py – Niobium heat-treatment simulation driver
(revision with **batch-seed support**)

The script sweeps bake temperature and hold time for a chosen temperature-
profile class (``const``, ``time_dep``, or ``two_step``), generates a full
GenSimReport bundle for every (T, t) point, and stores all provenance under an
experiment folder.

──────────────────────────────────────────────────────────────────────────────
USAGE
──────────────────────────────────────────────────────────────────────────────
Single-seed sweep (legacy)              Batch-seed sweep
----------------------------------      --------------------------------------
$ sim [-c CFG] [-p PROFILE]             $ sim batch SEEDS.yml
      [-r [N]] [CFG]                          [-c CFG] [-p PROFILE] [-r [N]]
                                           [CFG]

``-r N``  → add *N* extra re-oxidisation passes per (T, t) point
plain ``-r`` (no number) ⇒ ``N = 1``

──────────────────────────────────────────────────────────────────────────────
SEED-RECIPE YAML (batch mode)
──────────────────────────────────────────────────────────────────────────────
• List entries (or a mapping under key ``seeds``) with fields:

  * ``name``     – optional label (auto-generated if omitted)
  * ``profile``  – ``const`` | ``time_dep`` | ``two_step``  (default ``time_dep``)
  * ``bake_C``   – peak bake temperature [°C]
  * ``time_h``   – hold time [h]

Each recipe is first simulated once and saved under ``seeds/<name>/``; the final
oxygen profile then seeds the full (T, t) sweep.

──────────────────────────────────────────────────────────────────────────────
CLI FLAGS
──────────────────────────────────────────────────────────────────────────────
| flag                        | meaning                                                      | default          |
|-----------------------------|--------------------------------------------------------------|------------------|
| ``-c / --config``           | main simulation YAML                                         | ``sim_config.yml`` |
| ``-p / --profile``          | profile class for the sweep                                  | ``time_dep``       |
| ``-r / --reoxidize [N]``    | add *N* extra re-oxidisation passes per (T, t) point         | 0                  |

──────────────────────────────────────────────────────────────────────────────
OUTPUT LAYOUT
──────────────────────────────────────────────────────────────────────────────
experiments/
└── <YYYY-MM-DD>_<profile>_<git>/
    ├── run_meta.yml                # timestamp, git commit, CLI flags
    ├── sim_config.yml              # verbatim user YAML
    ├── effective_config.yml        # only if CLI flags override YAML
    ├── seeds/                      # batch mode only
    │   ├── <seed_A>/               # one-off seed simulation bundle
    │   │   ├── recipe.yml
    │   │   ├── oxygen_profile.csv
    │   │   └── … full GenSimReport artefacts …
    │   └── <seed_B>/ …
    ├── results/                    # sweep for “fresh” seed (single-seed mode)
    │   └── … GenSimReport artefacts …
    ├── results_seed-<seed_A>/      # sweep initiated from seed A
    │   └── … GenSimReport artefacts …
    └── results_seed-<seed_B>/      # sweep initiated from seed B
        └── … GenSimReport artefacts …
"""

from __future__ import annotations

import argparse
import os
import time
from datetime import datetime
from pathlib import Path
import shutil
import subprocess
from dataclasses import asdict
from typing import List, Tuple, Optional, Any

import yaml  # PyYAML
import numpy as np

from simulation.cn_solver import CNSolver
from simulation.sim_report import GenSimReport
from simulation.temp_profile import ConstantProfile, TimeDepProfile, TwoStepProfile
from config.sim_config import load_sim_config
from simulation.ciovati_model import CiovatiModel
from scripts.simulation_analyzer import CurrentDensityAnalyzer

###############################################################################
# ─── Internal helpers ───────────────────────────────────────────────────────
###############################################################################


def _git_hash(short: bool = True) -> str:
    """Return the current commit hash (short or full)."""
    rev = "HEAD"
    cmd = ["git", "rev-parse"]
    cmd += ["--short", rev] if short else [rev]

    # 1) try current working directory
    try:
        return subprocess.check_output(
            cmd, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass

    # 2) try the directory containing this file
    try:
        return subprocess.check_output(
            cmd,
            cwd=Path(__file__).resolve().parent,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return ""  # couldn’t find a repo


def _save_run_metadata(
    cfg_path: Path, cli_ns: argparse.Namespace, run_dir: Path, cfg_obj
) -> None:
    """Archive YAML config, CLI flags, git commit, and effective YAML (if changed)."""
    run_dir.mkdir(parents=True, exist_ok=True)

    shutil.copy2(cfg_path, run_dir / "sim_config.yml")

    meta = {
        "timestamp": datetime.now().astimezone().isoformat(),
        "git_commit": _git_hash(),
        "cli_flags": {
            k: (str(v) if isinstance(v, Path) else v)
            for k, v in vars(cli_ns).items()
            if k not in {"pos_config", "config"}
        },
    }
    (run_dir / "run_meta.yml").write_text(yaml.safe_dump(meta, sort_keys=False))

    try:
        merged_yaml = yaml.safe_dump(asdict(cfg_obj), sort_keys=False)
    except TypeError:
        merged_yaml = yaml.safe_dump(cfg_obj.__dict__, sort_keys=False)

    if merged_yaml != Path(cfg_path).read_text():
        (run_dir / "effective_config.yml").write_text(merged_yaml)


###############################################################################
# ─── Seed helpers for batch mode ────────────────────────────────────────────
###############################################################################

_ProfileMap: dict[str, Any] = {
    "const": ConstantProfile,
    "time_dep": TimeDepProfile,
    "two_step": TwoStepProfile,
}


def _load_seed_yaml(seed_yaml: Path) -> list[dict[str, Any]]:
    raw = yaml.safe_load(seed_yaml.read_text())
    if isinstance(raw, dict):
        seeds = raw.get("seeds", [])
    else:
        seeds = raw
    if not isinstance(seeds, list):
        raise ValueError("Seed YAML must be a list or have key 'seeds'.")
    return seeds


def _simulate_one_seed(cfg, seed_rec: dict[str, Any], save_dir: Path) -> np.ndarray:
    name = seed_rec.get("name", "seed")
    bake_C = seed_rec["bake_C"]
    time_h = seed_rec["time_h"]
    profile_key = seed_rec.get("profile", "time_dep")
    ProfileCls = _ProfileMap.get(profile_key)
    if ProfileCls is None:
        raise ValueError(f"Unknown profile {profile_key}")

    start_K = cfg.temp_profile.start_C + 273.15
    bake_K = bake_C + 273.15
    t_h, temps_K, total_h = ProfileCls(cfg, start_K, bake_K, time_h).generate()

    civ_model = CiovatiModel(cfg.ciovati)
    solver = CNSolver(cfg, temps_K, total_h, civ_model)
    U_record = solver.get_oxygen_profile()
    o_total = U_record[-1]

    x_grid = np.linspace(0, cfg.grid.x_max_nm, cfg.grid.n_x)
    GenSimReport(
        cfg, x_grid, o_total, time_h, bake_K, t_h, temps_K, profile_key, str(save_dir)
    ).generate()
    (save_dir / "recipe.yml").write_text(yaml.safe_dump(seed_rec, sort_keys=False))

    return o_total


###############################################################################
# ─── Core simulation logic ──────────────────────────────────────────────────
###############################################################################


def run_simulation(
    cfg,
    profile: str = "time_dep",
    reoxidize: bool = False,
    n_reoxidize: int = 0,
    *,
    seed_name: str = "fresh",
    U_initial: Optional[np.ndarray] = None,
) -> None:
    """Run the simulation sweep for a given configuration and profile.

    *U_initial* is only used for the very first solver call; afterwards the run
    proceeds with the computed profile.  This keeps backwards‑compatibility for
    single‑seed runs while enabling batch‑seed mode.
    """

    times_h = np.arange(
        cfg.time.start_h, cfg.time.stop_h + cfg.time.step_h, cfg.time.step_h
    )
    bake_C_list = np.arange(
        cfg.temperature.start_C,
        cfg.temperature.stop_C + cfg.temperature.step_C,
        cfg.temperature.step_C,
    )

    start_K = cfg.temp_profile.start_C + 273.15
    x_grid = np.linspace(0, cfg.grid.x_max_nm, cfg.grid.n_x)
    civ_model = CiovatiModel(cfg.ciovati)

    match profile:
        case "const":
            ProfileCls, suffix = ConstantProfile, "_const_temp"
        case "time_dep":
            ProfileCls, suffix = TimeDepProfile, ""
        case "two_step":
            ProfileCls, suffix = TwoStepProfile, "_two_step"
        case _:
            raise ValueError(f"Unknown profile: {profile!r}")

    base_out = cfg.output.directory
    if seed_name != "fresh":
        base_out = f"{base_out}_seed-{seed_name}"
    output_dir = base_out
    reox_output_dir = output_dir
    total_simulations = len(times_h) * len(bake_C_list)
    threshold = 400

    for bake_C in bake_C_list:
        bake_K = bake_C + 273.15
        for time_hold in times_h:
            tic = time.perf_counter()

            t_h, temps_K, total_h = ProfileCls(
                cfg, start_K, bake_K, time_hold
            ).generate()

            print(
                f"Running {profile} profile @ {bake_C:.0f}°C, hold time: {time_hold:.2f}h, total time: {total_h:.2f}h"
            )

            solver = CNSolver(cfg, temps_K, total_h, civ_model, U_initial=U_initial)
            U_record = solver.get_oxygen_profile()
            o_total = U_record[-1]
            U_initial = None  # only apply custom seed to the first solve

            if reoxidize:
                for n in range(n_reoxidize):
                    pass_dir = f"{reox_output_dir}_reoxidize_{n+1}"
                    report = GenSimReport(
                        cfg,
                        x_grid,
                        o_total,
                        time_hold,
                        bake_K,
                        t_h,
                        temps_K,
                        profile,
                        pass_dir,
                    )
                    report.generate()
                    print(f"Re-oxidizing {n+1} pass of {n_reoxidize}...")
                    solver = CNSolver(
                        cfg, temps_K, total_h, civ_model, U_initial=o_total
                    )
                    U_record = solver.get_oxygen_profile()
                    o_total = U_record[-1]
                    print(f"Re-oxidization pass {n+1} complete. output → {pass_dir}")
                output_dir = f"{reox_output_dir}_reoxidized"
            else:
                output_dir = reox_output_dir
            report = GenSimReport(
                cfg,
                x_grid,
                o_total,
                time_hold,
                bake_K,
                t_h,
                temps_K,
                profile,
                output_dir,
            )
            report.generate()

            elapsed = time.perf_counter() - tic
            print(
                f"Done: {profile} profile @ {bake_C:.0f}°C, hold time: {time_hold:.2f}h, total_time={total_h:.2f}h,\n "
                f"Completed in {elapsed:.2f}s, output → {output_dir}"
            )

    print(f"Total simulations: {total_simulations}")
    if total_simulations > threshold:
        if reoxidize:
            CurrentDensityAnalyzer(output_dir, "reoxidized").run()
            for n in range(n_reoxidize):
                pass_dir = f"{reox_output_dir}_reoxidize_{n+1}"
                CurrentDensityAnalyzer(pass_dir, f"reoxidize_{n+1}").run()
        else:
            CurrentDensityAnalyzer(output_dir).run()


###############################################################################
# ─── Batch orchestrator ─────────────────────────────────────────────────────
###############################################################################


def run_batch(cfg, seed_yaml: Path, profile: str, reoxidize: bool, n_reoxidize: int):
    seeds = _load_seed_yaml(seed_yaml)
    seeds_dir = Path(cfg.output.directory).parent / "seeds"
    seeds_dir.mkdir(parents=True, exist_ok=True)

    for rec in seeds:
        seed_name = rec.get("name", f"seed_{rec.get('bake_C')}C_{rec.get('time_h')}h")
        seed_out_dir = seeds_dir / seed_name
        U0 = _simulate_one_seed(cfg, rec, seed_out_dir)
        run_simulation(
            cfg, profile, reoxidize, n_reoxidize, seed_name=seed_name, U_initial=U0
        )


###############################################################################
# ─── Command‑line interface ─────────────────────────────────────────────────
###############################################################################


def _resolve_cfg(cfg_arg: str | None, pos_arg: str | None) -> Path:
    fname = cfg_arg or pos_arg or "sim_config.yml"
    if not Path(fname).suffix:
        fname += ".yml"
    for p in (Path(fname), Path("config") / fname):
        if p.exists():
            return p
    raise FileNotFoundError(fname)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="sim", description="Nb heat‑treatment simulator"
    )
    subparsers = parser.add_subparsers(dest="mode", required=False)

    # default/legacy
    parser.add_argument("-c", "--config", metavar="CONFIG", help="YAML config file")
    parser.add_argument(
        "-p",
        "--profile",
        choices=["const", "time_dep", "two_step"],
        default="time_dep",
        help="Temperature profile to use",
    )
    parser.add_argument(
        "-r",
        "--reoxidize",
        metavar="N",
        type=int,
        nargs="?",
        const=1,
        default=0,
        help="Number of re‑oxidization passes (omit flag for 0)",
    )
    parser.add_argument(
        "pos_config", nargs="?", metavar="CONFIG", help="Positional YAML config"
    )

    # batch sub‑command
    batch = subparsers.add_parser("batch", help="Run seeds from YAML then sweep")
    batch.add_argument("seed_yaml", type=Path, help="YAML file listing seed recipes")
    batch.add_argument("-c", "--config", metavar="CONFIG", help="YAML config file")
    batch.add_argument(
        "-p",
        "--profile",
        choices=["const", "time_dep", "two_step"],
        default="time_dep",
        help="Temperature profile to use in sweep",
    )
    batch.add_argument(
        "-r",
        "--reoxidize",
        metavar="N",
        type=int,
        nargs="?",
        const=1,
        default=0,
        help="Number of re‑oxidization passes (omit flag for 0)",
    )
    batch.add_argument(
        "pos_config", nargs="?", metavar="CONFIG", help="Positional YAML config"
    )
    args = parser.parse_args()

    cfg_path = _resolve_cfg(args.config, args.pos_config)
    cfg = load_sim_config(cfg_path)

    stamp = datetime.now().strftime("%Y-%m-%d")
    run_root = Path("experiments") / f"{stamp}_{args.profile}_{_git_hash()}"
    cfg.output.directory = str(run_root / "results")
    _save_run_metadata(cfg_path, args, run_root, cfg)

    if args.mode == "batch":
        run_batch(
            cfg, args.seed_yaml, args.profile, bool(args.reoxidize), args.reoxidize
        )
    else:
        run_simulation(cfg, args.profile, bool(args.reoxidize), args.reoxidize)


if __name__ == "__main__":
    main()
