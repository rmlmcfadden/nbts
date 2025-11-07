#!/usr/bin/env python3
"""
plot_ciovati.py

Compute & plot Ciovati oxygen-diffusion profile c(x) at a given time (in hours) & temp,
using the Ciovati (2006) model. Manual inputs: time_h and temp_C. All other simulation parameters (grid, ciovati) loaded from config.
"""

import os
import numpy as np
import matplotlib.pyplot as plt

# Import your configuration loader and Ciovati model classes
from config.sim_config import load_sim_config, SimConfig
from simulation.ciovati_model import CiovatiParams, CiovatiModel

# ─── USER CONFIGURATION ─────────────────────────────────────────────────────────
# Path to your simulation configuration file
config_path = "config/sim_config.yml"

# Output directory for Ciovati plots
out_dir = "experiments/ciovati"

# Manual inputs
time_h = 100.0  # time in hours
temp_C = 200.0  # temperature in °C
# ────────────────────────────────────────────────────────────────────────────────

# Load simulation config
cfg: SimConfig = load_sim_config(config_path)

# Extract grid parameters from config
x_max_nm = cfg.grid.x_max_nm
n_x = cfg.grid.n_x

# Extract Ciovati parameters
ciov_cfg = cfg.ciovati
params = CiovatiParams(
    D_0=ciov_cfg.D_0,
    E_A_D=ciov_cfg.E_A_D,
    k_A=ciov_cfg.k_A,
    E_A_k=ciov_cfg.E_A_k,
    u_0=ciov_cfg.u_0,
    v_0=ciov_cfg.v_0,
    c_0=getattr(ciov_cfg, "c_0", 0.0),
)
model = CiovatiModel(params)

# Ensure output directory exists
os.makedirs(out_dir, exist_ok=True)

# Convert manual inputs
time_s = time_h * 3600.0  # hours to seconds
T_K = temp_C + 273.15  # °C to K

# Generate depth grid and compute concentration profile
x = np.linspace(0, x_max_nm, n_x)
c_profile = np.array([model.c(xi, time_s, T_K) for xi in x])

# Plot the profile
plt.figure(figsize=(6, 4))
plt.plot(x, c_profile, lw=2)
plt.xlabel("Depth (nm)")
plt.ylabel("Oxygen concentration (at.%)")
plt.xlim(0, 150)
plt.title(f"Ciovati profile: T={temp_C:.1f}°C, t={time_h:.2f} h")
plt.tight_layout()

# Save the figure
filename = f"ciovati_profile_{int(temp_C)}C_{time_h}h.pdf"
out_path = os.path.join(out_dir, filename)
plt.savefig(out_path, dpi=300)
plt.close()

print(f"Saved Ciovati profile plot to {out_path}")
