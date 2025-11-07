import pytest
import numpy as np
import os
import matplotlib.pyplot as plt
from simulation.cn_solver import CNSolver
from simulation.ciovati_model import CiovatiModel
from simulation.temp_profile import ConstantProfile
from config.sim_config import load_sim_config


def sum_of_squared_deviations(y_1, y_2):
    """
    Computes the sum of squared deviations (SSD) between two arrays.

    Parameters:
        y_1 (array-like): Numerical solution (from Crank-Nicolson solver).
        y_2 (array-like): Analytical solution (from Ciovati model).

    Returns:
        float: The sum of squared deviations between the two profiles.
    """
    return np.sum((y_1 - y_2) ** 2)


@pytest.fixture
def setup_cn_solver():
    """
    Loads the simulation configuration and sets up the solver and model.
    """
    # Load the simulation config using the function from sim_config.py
    # print("Loading configuration...")
    config = load_sim_config("config/sim_config.yml")

    # Access the config attributes using dot notation
    ciovati_config = config.ciovati  # Access ciovati config using dot notation
    # print(f"Ciovati model config: {ciovati_config}")

    # Initialize the Ciovati model with the config
    civ_model = CiovatiModel(ciovati_config)

    # Set up time grid for the solver (using start, stop, and step from config)
    hold_h = config.time.start_h  # Access time config using dot notation

    # Set up temperature grid for the solver (using start, stop, and step from config)
    temp_K = (
        config.temperature.start_C + 273.15
    )  # Access temperature config using dot notation

    # Generate time and temperature grids using the ConstantProfile class
    t_h, temps_K, total_h = ConstantProfile(config, temp_K, temp_K, hold_h).generate()
    # print(f"Generated time grid: {t_h}, temperature grid: {temps_K}")

    # Initialize the Crank-Nicolson solver
    cn_solver = CNSolver(config, temps_K, total_h, civ_model)

    return cn_solver, civ_model, temp_K, total_h, config


def test_oxygen_concentration(setup_cn_solver):
    """
    Test the oxygen concentration profile from the Crank-Nicolson solver against the Ciovati model.
    """
    cn_solver, civ_model, bake_K, total_h, config = setup_cn_solver
    # print("Setup complete. Running test...")

    time_sec = total_h * 3600.0  # Convert time to seconds
    x_grid = np.linspace(0, config.grid.x_max_nm, config.grid.n_x)  # Spatial grid

    # Compute the analytical solution using the Ciovati model
    # print("Computing Ciovati model profile...")
    ciovati_profile = [civ_model.c(x, time_sec, bake_K) for x in x_grid]

    # Solve using Crank-Nicolson
    # print("Solving with Crank-Nicolson solver...")
    cn_solver_results = cn_solver.get_oxygen_profile()
    cn_solver_profile = cn_solver_results[-1]  # Get the last profile

    # Check if the profiles are valid (debugging)
    # print(f"Simulation profile: {cn_solver_profile[:5]}...")  # Print first 5 values for checking
    # print(f"Ciovati profile: {ciovati_profile[:5]}...")  # Print first 5 values for checking

    # Plot the results
    plot_oxygen_profile(
        config, x_grid, total_h, bake_K, cn_solver_profile, ciovati_profile
    )

    # Compute SSD between the solver's result and the analytical solution
    ssd_value = sum_of_squared_deviations(cn_solver_profile, ciovati_profile)
    # print(f"SSD value: {ssd_value}")

    # Set a threshold for the acceptable deviation
    tolerance = 1e-3

    # Compare the SSD value with the tolerance
    assert (
        ssd_value < tolerance
    ), f"SSD between solver and analytical solution is too large: {ssd_value}"


def plot_oxygen_profile(cfg, x_grid, total_h, bake_K, o_total, c_model):
    """
    Create a plot comparing the simulation results with the Ciovati model and save it.

    Parameters:
        cfg (SimConfig): Configuration object with model parameters and output directory.
        x_grid (array-like): Spatial grid points (nm).
        total_h (float): Simulation time in hours.
        bake_K (float): Bake temperature in Kelvin.
        o_total (array-like): Simulated oxygen concentration profile.
        c_model (array-like): Analytical oxygen concentration profile from Ciovati model.
    """
    # Determine output directory (use 'tests/test_cn_solver_plots' if not specified)
    out_dir = os.path.join(
        "tests", "test_cn_solver_plots"
    )  # Change output directory to tests/test_cn_solver_plots
    os.makedirs(out_dir, exist_ok=True)  # Create directory if it doesn't exist

    # Plot both profiles
    fig, ax = plt.subplots()
    ax.plot(x_grid, o_total, "-", label="Simulation")
    ax.plot(x_grid, c_model, "-", label="Ciovati Model")
    ax.set_xlabel("Depth (nm)")
    ax.set_ylabel("Oxygen Concentration (at.%)")
    ax.set_title(f"Oxygen Profile at {total_h:.1f} h and {bake_K - 273.15:.1f} K")
    ax.set_xlim(0, 150)
    ax.legend()

    # Save plot in the desired directory
    filename = os.path.join(out_dir, "oxygen_profile_comparison.pdf")
    fig.savefig(filename)
    print(f"Saved plot to {filename}")
