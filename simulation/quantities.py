"""Application of the oxygen diffusion model in Nb.

This module contains functions that make use of the oxygen diffusion model in
Nb to simulate adulteration to its intrinsic properties caused by (spatially
inhomogeneous) interstitial doping.
"""

from typing import Annotated, Sequence
import numpy as np
from scipy import constants
from scipy.special import zeta


def ell(
    c: float,
    a_0: float = 4.5e-12,
    sigma_0: float = 0.37e-15,
) -> float:
    r"""Nb's electron mean-free-path.

    Calculate the mean-free-path l of Nb's electrons as a result of
    oxygen doping (e.g., from surface heat-treatments).

    Args:
        c: Impurity concentration (at. %).
        a_0: Proportionality coefficient between oxygen concentration and Nb's residual resistivity :math:`\rho_{0}` (\ :math:`\ohm` m ppma\ :sup:`-1`\ ).
        sigma_0: Proportionality coefficient between Nb's residual resistivity :math:`\rho_{0}` its electron mean-free-path (\ :math:`\ohm` m\ :sup:`2`\ ).

    Returns:
        The electron mean-free-path (nm).
    """
    # Define a small epsilon to avoid division by zero
    epsilon = 1e-10

    # Convert the oxygen concentration in at. % to ppma
    stoich_per_at_percent = 1e-2
    ppma_per_stoich = 1e6
    c_ppma = c * stoich_per_at_percent * ppma_per_stoich

    # Clip the denominator to avoid it being too small
    # denominator = np.clip(a_0 * c_ppma, epsilon, None)
    denominator = a_0 * c_ppma

    ell_m = sigma_0 / denominator  # in meters
    nm_per_m = 1e9
    ell_nm = ell_m * nm_per_m  # convert to nm
    return ell_nm


def lambda_eff(
    ell: float,
    lambda_L: float = 27.0,
    xi_0: float = 33,
) -> float:
    r"""Effective magnetic penetration depth.

    Effective magnetic penetration depth :math:`\lambda_{\mathrm{eff.}` for an
    impure superconductor (at 0 K).

    Args:
        ell: Electron mean-free-path :math:`\ell` (nm).
        lambda_L: London penetration depth :math:`\lambda_{L}` (nm).
        xi_0: Pippard/Bardeen-Cooper-Schrieffer coherence length (nm).

    Returns:
        The effective magnetic penetration depth (nm).
    """

    # Note: the factor pi/2 is necessary to be in accord with BCS theory
    # see, e.g.:
    #
    # P. B. Miller, "Penetration Depth in Impure Superconductors",
    # Phys. Rev. 113, 1209 (1959).
    # https://doi.org/10.1103/PhysRev.113.1209
    #
    # J. Halbritter, "On the penetration of the magnetic field into a
    # superconductor", Z. Phys. 243, 201–219 (1971).
    # https://doi.org/10.1007/BF01394851
    return lambda_L * np.sqrt(1.0 + (np.pi / 2.0) * (xi_0 / ell))


def B(
    z_nm: Sequence[float],
    applied_field_G: Annotated[float, 0:None],
    penetration_depth_nm: Annotated[float, 0:None],
    dead_layer_nm: Annotated[float, 0:None] = 0.0,
    demagnetization_factor: Annotated[float, 0:1] = 0.0,
) -> Sequence[float]:
    """Meissner screening profile for the simple London model.

    Args:
        z_nm: Depth below the surface (nm).
        penetration_depth_nm: Magnetic penetration depth (nm).
        applied_field_G: Applied magnetic field (G).
        dead_layer_nm: Non-superconducting dead layer thickness (nm).
        demagnetization_factor: Effective demagnetization factor.

    Returns:
        The magnetic field as a function of depth (G).
    """

    effective_field_G = applied_field_G / (1.0 - demagnetization_factor)

    return effective_field_G * np.exp(-z_nm / penetration_depth_nm)


def J(
    z_nm: Sequence[float],
    applied_field_G: Annotated[float, 0:None],
    penetration_depth_nm: Annotated[float, 0:None],
    dead_layer_nm: Annotated[float, 0:None] = 0.0,
    demagnetization_factor: Annotated[float, 0:1] = 0.0,
) -> Sequence[float]:
    """Meissner current density for the simple London model.

    Args:
        z_nm: Depth below the surface (nm).
        penetration_depth_nm: Magnetic penetration depth (nm).
        applied_field_G: Applied magnetic field (G).
        dead_layer_nm: Non-superconducting dead layer thickness (nm).
        demagnetization_factor: Effective demagnetization factor.

    Returns:
        The current density as a function of the depth (A m^-2).
    """

    # calculate the prefactor for the conversion
    G_per_T = 1e4
    # nm_per_m = 1e9
    m_per_nm = 1e-9
    mu_0 = constants.value("vacuum mag. permeability") * G_per_T

    j_0 = -1.0 / mu_0

    # correct the depth for the dead layer
    z_corr_nm = z_nm - dead_layer_nm

    return (
        j_0
        * (-1.0 / penetration_depth_nm / m_per_nm)
        * B(
            z_nm,
            applied_field_G,
            penetration_depth_nm,
            dead_layer_nm,
            demagnetization_factor,
        )
    )


def J_c(
    penetration_depth_nm: Annotated[float, 0:None],
    B_c: float = 199.3,  # Critical magnetic field in mT (default for Nb)
):
    """Critical current density for the simple London model.

    Args:
        penetration_depth_nm: Magnetic penetration depth (nm).
        B_c: Critical magnetic field (mT).

    Returns:
        The critical current density (A m^-2).
    """

    # calculate the prefactor for the conversion
    G_per_T = 1e4
    m_per_nm = 1e-9
    mu_0 = constants.value("vacuum mag. permeability") * G_per_T
    penetration_depth_m = penetration_depth_nm * m_per_nm

    return B_c / (mu_0 * penetration_depth_m)


def chi(a_imp, n_max=2000) -> float:
    """
    Gor'kov function χ(a_imp) ≈ (8 / [7 ζ(3)]) * sum_{n=0}^∞ [1 / ((2n+1)^2 (2n+1 + a_imp))].

    a_imp : float
        The impurity parameter a_imp.
    n_max : int
        Truncation index for the series (increase until convergence). Must be the same as Nx, space steps in simulation.

    Returns:
        The Gor'kov function χ(a_imp).
    """
    zeta3 = zeta(3, 1)  # ζ(3) via the Hurwitz zeta ζ(s,q) with q=1
    n = np.arange(n_max + 1)  # n = 0,1,2,…,n_max
    odd = 2 * n + 1
    terms = 1.0 / (odd**2 * (odd + a_imp))
    return (8.0 / (7.0 * zeta3)) * np.sum(terms)


def kappa(
    cfg,
    lambda_eff: float,
    lambda_L: float = 27.0,
    xi_0: float = 33.0,
) -> float:
    r"""Compute the impurity-corrected Ginzburg–Landau parameter κ.

    This implements the Gor'kov correction for κ in the presence of non-magnetic
    impurities:

        κ = κ_clean / χ(a_imp)
    Args:
        lambda_eff (float):
            Effective magnetic penetration depth λ_eff at finite impurity
            concentration, in nanometers (nm).
        lambda_L (float, optional):
            Clean-limit London penetration depth λ_L at zero temperature,
            in nanometers (nm).
        xi_0 (float, optional):
            BCS coherence length ξ_0 at zero temperature, in nanometers (nm).

    Returns:
        Impurity-corrected Ginzburg–Landau parameter κ (dimensionless).
    """

    # convert inputs to nanometers (factor cancels in the κ_clean ratio)
    nm_per_m = 1e9
    lambda_L_nm = lambda_L * nm_per_m
    xi_0_nm = xi_0 * nm_per_m

    kappa_clean = 0.957 * lambda_L_nm / xi_0_nm
    a_imp = 0.882 * xi_0_nm / lambda_eff
    n_max = cfg.grid.n_x - 1
    chi_a_imp = chi(a_imp, n_max)

    return kappa_clean / chi_a_imp


def lambda_eff_corr(
    cfg,
    lambda_eff: float,
    B_c: float = 199.3,
) -> float:
    r"""Corrected London penetration depth for an impure superconductor.
        Taking into acount the non-linear meissner effect

    λ_corr(B) = [1 + κ (κ + 2³ᐟ²) B² / (8 (κ + 2¹ᐟ²)² B_c²)] · λ_eff

    Args:
        lambda_eff: zero‐field penetration depth λ (nm)
        B_c:          thermodynamic critical field (default = 200mT)

    Returns:
        field‐corrected λ_corr (nm)
    """
    B_0 = B(0, 100, lambda_eff)
    kappa_val = kappa(cfg, lambda_eff)
    correction = 1.0 + (kappa_val * (kappa_val + 2**1.5) * B_0**2) / (
        8 * (kappa_val + 2**0.5) ** 2 * B_c**2
    )
    return correction * lambda_eff
