"""Oxygen diffusion in Nb.

This module contains functions implementing the oxygen dissolution and diffusion
model in Nb metal, as described in:
G. Ciovati, Appl. Phys. Lett. 89, 022507 (2006).
https://doi.org/10.1063/1.2220059
"""

import numpy as np
from scipy import constants, integrate
from dataclasses import dataclass


@dataclass
class CiovatiParams:
    """
    Parameter container for Ciovati model.
    """

    D_0: float  # Pre-exponential factor for diffusion (cm^2/s)
    E_A_D: float  # Activation energy for diffusion (J/mol·K)
    k_A: float  # Pre-exponential factor for surface dissolution (1/s)
    E_A_k: float  # Activation energy for dissolution (J/mol·K)
    u_0: float  # Total amount of surface oxygen (at.%·nm)
    v_0: float  # Initial interstitial amount (at.%·nm)
    c_0: float = 0.0  # Uniform bulk concentration (at.% )


class CiovatiModel:
    """
    Ciovati model for oxygen concentration in niobium.
    """

    def __init__(self, p: CiovatiParams):
        self.p = p

    def D(self, T: float) -> float:
        r"""Diffusion coefficient for oxygen in Nb.

        Diffusion coefficient :math:`D` for (interstitial) oxygen in Nb,
        assumed to follow an Arrhenius temperature dependence.

        Args:
            T: Temperature (K).

        Returns:
            The diffusion coefficient (nm^2 s^-1).
        """
        # unit conversion factor
        nm_per_cm = 1e7
        nm2_per_cm2 = nm_per_cm**2

        return self.p.D_0 * np.exp(-self.p.E_A_D / (constants.R * T)) * nm2_per_cm2

    def k(self, T: float) -> float:
        r"""Rate constant for oxygen dissolution from Nb's surface oxide.

        Rate constant :math:`k` for the dissolution of Nb's native surface oxide
        layer, assumed to follow an Arrhenius temperature dependence.

        Args:
            T: Temperature (K).

        Returns:
            The rate constant (s^-1).
        """
        # evaluate the rate constant
        return self.p.k_A * np.exp(-self.p.E_A_k / (constants.R * T))

    def q(self, t: float, T: float) -> float:
        r"""Reaction term in the diffusion equation.

        Reaction term in the diffusion equation :math:`q(x, t, T)` for the rate
        that interstitial oxygen is incorporated into the system
        [see Eqs. (5) and (6) in Ciovati (2006)].

        Args:
            t: Time (s).
            T: Temperature (K).

        Returns:
            The oxygen introduction rate (at.%/s).
        """
        # evaluate the rate of oxygen incorporation into the system
        kT = self.k(T)
        return self.p.u_0 * kT * np.exp(-kT * t)

    def v(self, x: float, t: float, T: float) -> float:
        r"""Solution to the diffusion eqn for interstitial oxygen.

        Solution to the diffusion equation :math:`v(x,t)` for interstitial oxygen
        initially present at the metal/surface oxide boundary
        [see Eq. (8) in Ciovati (2006)].

        Args:
            x: Depth (nm).
            t: Time (s).
            T: Temperature (K).

        Returns:
            The oxygen concentration (at.%).
        """
        D_T = self.D(T)
        # argument for the exponential in the Gaussian
        arg = np.square(x) / (4.0 * D_T * t)

        # prefactor for the Gaussian
        pre = self.p.v_0 / np.sqrt(np.pi * D_T * t)

        # evaluate the Gaussian
        return pre * np.exp(-arg) + self.p.c_0

    def _u_integrand(self, s: float, x: float, t: float, T: float) -> float:
        r"""Integrand from Eq. (7) in Ciovati (2006)."""
        # argument for the exponential
        arg = np.square(x) / (4.0 * self.D(T) * (t - s))

        # prefactor for the solution
        pre = self.k(T) * np.exp(-self.k(T) * s) / np.sqrt(t - s)

        return pre * np.exp(-arg)

    def u(self, x: float, t: float, T: float) -> float:
        r"""Solution to the diffusion eqn for dissolved oxygen.

        Solution to the diffusion equation :math:`u(x,t)` for oxygen that is
        thermally dissolved from Nb's native surface oxide layer
        [see Eq. (7) in Ciovati (2006)].

        Args:
            x: Depth (nm).
            t: Time (s).
            T: Temperature (K).

        Returns:
            The oxygen concentration (at.%).
        """
        # evaluate the integral
        integral, _ = integrate.quad(
            self._u_integrand,
            0,  # lower integration limit
            t,  # upper integration limit
            args=(x, t, T),
            full_output=False,
            epsabs=np.sqrt(np.finfo(float).eps),
            epsrel=np.sqrt(np.finfo(float).eps),
            limit=int(1e3),
        )

        # prefactor for the solution
        pre = self.p.u_0 / np.sqrt(np.pi * self.D(T))

        # evaluate the Gaussian
        return pre * integral

    def c(self, x: float, t: float, T: float) -> float:
        """Total interstitial oxygen concentration in Nb metal.

        Total oxygen concentration, given by the sum of contributions from oxygen
        thermally dissolved from Nb's surface oxide layer and interstitial oxygen
        initially present at the metal/oxide interface
        [see Eq. (10) in Ciovati (2006)].

        Args:
            x: Depth (nm).
            t: Time (s).
            T: Temperature (K).

        Returns:
            The total oxygen concentration (at.%).
        """
        # sum the u & v terms to get the total concentration
        return self.u(x, t, T) + self.v(x, t, T)
