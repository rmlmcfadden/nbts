import os
import glob
import pandas as pd
import numpy as np
from simulation.quantities import (
    B,
    J,
)  # Assuming B and J functions are imported from quantities

# Base directory containing simulation folders
base_dir = "sim_output"
subfolders = glob.glob(os.path.join(base_dir, "sim_t*_T*"))

# Loop through each subfolder to process the simulation data
for folder in subfolders:
    folder_name = os.path.basename(folder)  # e.g., "sim_t0.1_T100"

    # Path to the lambda_eff_val.csv (assuming it's in the data subfolder)
    lambda_eff_path = os.path.join(folder, "data", "penetration_depth.csv")
    if not os.path.exists(lambda_eff_path):
        print(f"File not found: {lambda_eff_path}")
        continue

    lambda_eff_path_corr = os.path.join(
        folder, "data", "penetration_depth_corrected.csv"
    )
    if not os.path.exists(lambda_eff_path_corr):
        print(f"File not found: {lambda_eff_path_corr}")
        continue

    # Read lambda_eff_val.csv
    df_lambda = pd.read_csv(lambda_eff_path)
    df_lambda_corr = pd.read_csv(lambda_eff_path_corr)

    # Ensure the 'lambda_eff_val' column exists
    if "penetration_depth" not in df_lambda.columns:
        print(f"Column 'penetration_depth' not found in {lambda_eff_path}")
        continue

    if "penetration_depth_corrected" not in df_lambda_corr.columns:
        print(
            f"Column 'penetration_depth_corrected' not found in {lambda_eff_path_corr}"
        )
        continue

    # Get the lambda_eff values
    lambda_eff_vals = df_lambda["penetration_depth"].values
    lambda_eff_vals_corr = df_lambda_corr["penetration_depth_corrected"].values

    # Compute the required values
    H0 = 100  # As per the provided formula
    lambda_eff_max = lambda_eff_vals.max()
    lambda_eff_min = lambda_eff_vals.min()

    # Compute B and J values using the formulas
    B_dirty = B(df_lambda["x"], H0, lambda_eff_max)
    B_clean = B(df_lambda["x"], H0, lambda_eff_min)
    J_dirty = J(df_lambda["x"], H0, lambda_eff_max)
    J_clean = J(df_lambda["x"], H0, lambda_eff_min)

    lambda_eff_max_corr = lambda_eff_vals_corr.max()
    lambda_eff_min_corr = lambda_eff_vals_corr.min()
    B_dirty_corr = B(df_lambda_corr["x"], H0, lambda_eff_max_corr)
    B_clean_corr = B(df_lambda_corr["x"], H0, lambda_eff_min_corr)
    J_dirty_corr = J(df_lambda_corr["x"], H0, lambda_eff_max_corr)
    J_clean_corr = J(df_lambda_corr["x"], H0, lambda_eff_min_corr)

    # -----------------------------
    # Update current_density.csv with J_dirty and J_clean
    current_density_path = os.path.join(folder, "data", "current_density.csv")
    if os.path.exists(current_density_path):
        df_current_density = pd.read_csv(current_density_path)
        # Add J_dirty and J_clean as new columns
        df_current_density["J_dirty"] = J_dirty
        df_current_density["J_clean"] = J_clean
        df_current_density.to_csv(current_density_path, index=False)

    current_density_path_corr = os.path.join(
        folder, "data", "current_density_corrected.csv"
    )
    if os.path.exists(current_density_path_corr):
        df_current_density_corr = pd.read_csv(current_density_path_corr)
        # Add J_dirty and J_clean as new columns
        df_current_density["J_dirty"] = J_dirty_corr
        df_current_density["J_clean"] = J_clean_corr
        df_current_density.to_csv(current_density_path_corr, index=False)

    # -----------------------------
    # Update screening_profile.csv with B_dirty and B_clean
    screening_profile_path = os.path.join(folder, "data", "screening_profile.csv")
    if os.path.exists(screening_profile_path):
        df_screening_profile = pd.read_csv(screening_profile_path)
        # Add B_dirty and B_clean as new columns
        df_screening_profile["B_dirty"] = B_dirty
        df_screening_profile["B_clean"] = B_clean
        df_screening_profile.to_csv(screening_profile_path, index=False)

    screening_profile_path_corr = os.path.join(
        folder, "data", "screening_profile_corrected.csv"
    )
    if os.path.exists(screening_profile_path_corr):
        df_screening_profile_corr = pd.read_csv(screening_profile_path_corr)
        # Add B_dirty and B_clean as new columns
        df_screening_profile["B_dirty"] = B_dirty_corr
        df_screening_profile["B_clean"] = B_clean_corr
        df_screening_profile.to_csv(screening_profile_path_corr, index=False)

    print(f"Processed folder: {folder_name}")
