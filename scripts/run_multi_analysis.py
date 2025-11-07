#!/usr/bin/env python3
"""
Run the multi‑oxidation current‑density analysis without any CLI flags.

Edit the CONFIG SECTION below to point to the parent directory that
contains folders such as

    results_oxidized_1/
    results_oxidized_2/
    results_oxidized/        # final pass

The script will create an output folder named
    <parent>/analysis_multi_<N>step/
and store three PDF heat‑map figures there.
"""

# ─── CONFIG SECTION ─────────────────────────────────────────────────────
PARENT_DIR = "experiments/2025-05-18_time_dep_9db8969"  # <— edit me
STEP_GLOB = "results_reoxidize*"  # Pattern for pass directories
# ────────────────────────────────────────────────────────────────────────

from scripts.simulation_analyzer import (
    MultiOxidationCurrentDensityAnalyzer,
)


def main() -> None:
    analyzer = MultiOxidationCurrentDensityAnalyzer(
        parent_dir=PARENT_DIR,
        step_glob=STEP_GLOB,
    )
    analyzer.run()


if __name__ == "__main__":
    main()
