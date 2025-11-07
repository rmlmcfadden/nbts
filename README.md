# NBTS: Numerical Simulation of Niobium Heat Treatments for SRF Cavity Optimization

**NBTS** (Niobium Bake Treatment Simulator) is a scientific computing framework for modeling the diffusion of interstitial oxygen in niobium under various heat treatment conditions. This tool supports the simulation and analysis of defect engineering for enhancing the performance of niobium superconducting radio frequency (SRF) accelerator cavities.

---

## Overview

This project numerically solves Fick’s second law of diffusion with a source term. The goal is to predict the depth-dependent oxygen concentration profile resulting from surface baking or heat treatment protocols, and to analyze the corresponding impact on Nb's superconducting properties.

---

## Physical Model

The governing equation is:

∂C(x, t)/∂t = ∂/∂x [ D(T) ∂C(x, t)/∂x ] + q(x, t, T)

Where:
- `C(x, t)` is the interstitial oxygen concentration
- `D(T)` is the temperature-dependent diffusion coefficient
- `q(x, t, T)` is a source term that models surface oxygen activity (optional)

Discretization is performed using the Crank–Nicolson finite difference method (FDM), which offers second-order accuracy in space and time with good numerical stability.

---

## Analysis Capabilities

Simulated concentration profiles are post-processed to compute:

- **Electron mean free path** `ℓ(x, t)`
- **Effective magnetic penetration depth** `λ_eff(x, t)`
- **Magnetic screening profile** `B(x, t)`
- **Supercurrent density** `J(x, t)`
- **Visualization** using Matplotlib (e.g., `pcolormesh` plots for depth–time–concentration)

These outputs allow analysis of treatment profiles and their impact on SRF cavity performance.

---


## Installation & Operation

```bash
cd nbts
pip3 install --user .
```


```bash
sim -h
```

```bash
sim -c sim_config.yml -p const
```

