from typing import Annotated, Sequence
import numpy as np
from scipy import constants, integrate, interpolate


class GLESolver:
    """Generalized London Equation (GLE) Solver.

    This class provides routines to numerically solve the GLE for a
    depth-dependent magnetic penetration depth.

    See, e.g.:
    M. Checchin et al., Appl. Phys. Lett. 117, 032601 (2020).
    https://doi.org/10.1063/5.0013698

    See also Eq. (1) in:
    M. S. Pamianchi at al., Phys. Rev. B 50, 13659 (1994).
    https://doi.org/10.1103/PhysRevB.50.13659

    Attributes:
        _x_nodes: x-values used as the initial mesh by the solver.
        _y_guess: y-values used as the guess for the function/derivative by the solver.
        _sol: Object encapsulating the solver's solution.
    """

    def __init__(
        self,
        x_vals: Sequence[float],
        lambda_vals: Sequence[float],
        x_nodes_min: float = 0.0,
        x_nodes_max: float = 1000.0,
        x_nodes_num: int = 1001,
    ) -> None:
        """Constructor for the GLE Solver.

        Args:
            x_vals: Depth values defining the... (nm).
            lambda_vals: Effective magnetic penetration depth... (nm).
            x_nodes_min: Minimum of the x-values used as the initial mesh by the solver.
            x_nodes_max: Maximum of the x-values used as the initial mesh by the solver.
            x_nodes_num: Number of x-values used as the initial mesh by the solver.
        """

        # make sure that x_vals is compatible with the chosen x_nodes
        if np.max(x_vals) < x_nodes_max:
            raise ValueError(
                f"""The maximum value in x_vals must be at least as large as x_nodes_max.
Current values are: max(x_vals) = {np.max(x_vals)}, x_nodes_max = {x_nodes_max}.

To resolve this, it is suggested that the range of x_vals be increased.
"""
            )

        # create the interpolator & its derivative
        self._lambda = interpolate.PchipInterpolator(
            x_vals,
            lambda_vals,
            extrapolate=False,
        )
        self._lambda_prime = self._lambda.derivative(nu=1)

        # create the x-nodes
        self._x_nodes = np.linspace(x_nodes_min, x_nodes_max, num=x_nodes_num)

        # solve the GLE
        self._solve()

    def _gle_derivs(
        self,
        t: Sequence[float],
        y: Sequence[float],
    ) -> Sequence[float]:
        """Right-hand side of the system of equations to solve.

        Right-hand side of the system of equations to solve, re-written as
        1st-order expressions.

        Indexes [0] refer to the function being solved for.
        Indexes [1] refer to the function's derivative.

        Args:
            t: x-values.
            y: y-values

        Returns:
            An array of the system of 1st-order equations to solve.
        """

        # variable substitution
        b, nu = y

        # evaluate the penetration depth and its derivative
        l = self._lambda(t)
        l_prime = self._lambda_prime(t)

        return np.vstack(
            [
                nu,
                -(2.0 / l) * l_prime * nu + (1.0 / l**2) * b,
            ]
        )

    def _bc(
        self,
        ya: Sequence[float],
        yb: Sequence[float],
    ) -> Sequence[float]:
        """Boundary conditions for the solver.

        Indexes [0] refer to the function being solved for.
        Indexes [1] refer to the function's derivative.

        Args:
            ya: Array of lower bounds.
            yb: Array of upper bounds.

        Returns:
            An array of penalties for the boundary conditions.
        """

        return np.array(
            [
                ya[0] - 1,  # B(x) / B_0 = 1
                yb[1],  # B'(np.inf) / B_0 = 0
            ]
        )

    def _solve(
        self,
        penetration_depth_nm: float = 40.0,
        tolerance: float = np.sqrt(np.finfo(float).eps),  # 1e3 * np.finfo(float).eps,
        max_x_nodes: int = np.iinfo(np.int32).max,
    ) -> None:
        """Solve the GLE numerically.

        Args:
            penetration_depth_nm: Magnetic penetration depth to use as initial guess (nm).
            tolerance: Convergence criteria for the solver.
            max_x_nodes: Maximum number of x nodes used by the solver.
        """

        # dynamically create the initial guesses for the solver
        self._y_guess = np.zeros((2, self._x_nodes.size))
        self._y_guess[0] = np.exp(-self._x_nodes / penetration_depth_nm)
        self._y_guess[1] = (-1.0 / penetration_depth_nm) * np.exp(
            -self._x_nodes / penetration_depth_nm
        )

        # solve the system of equations using boundary conditions and save the result
        self._sol = integrate.solve_bvp(
            self._gle_derivs,
            self._bc,
            self._x_nodes,
            self._y_guess,
            p=None,
            S=None,
            fun_jac=None,
            bc_jac=None,
            tol=tolerance,
            max_nodes=max_x_nodes,
            verbose=0,
            # bc_tol=np.sqrt(np.finfo(float).eps),
        )

    def screening_profile(
        self,
        z_nm: Sequence[float],
        applied_field_G: Annotated[float, 0:None],
        dead_layer_nm: Annotated[float, 0:None] = 0.0,
        demagnetization_factor: Annotated[float, 0:1] = 0.0,
    ) -> Sequence[float]:
        """Calculate the Meissner screening profile.

        Args:
            z_nm: Depth below the surface (nm).
            applied_field_G: Applied magnetic field (G).
            dead_layer_nm: Non-superconducting dead layer thickness (nm).
            demagnetization_factor: Effective demagnetization factor.

        Returns:
            The Meissner screening profile at depth z (G).
        """

        # solve the GLE
        # self._solve()

        # calculate the geometrically enhanced field value
        effective_field_G = applied_field_G / (1.0 - demagnetization_factor)

        # correct the depth for the dead layer
        z_corr_nm = z_nm - dead_layer_nm

        # return the current density
        return np.piecewise(
            z_corr_nm,
            [
                z_corr_nm < 0.0,
                z_corr_nm >= 0.0,
            ],
            [
                lambda x: np.full(x.shape, effective_field_G),
                lambda x: effective_field_G * self._sol.sol(x)[0],
            ],
        )

    def current_density(
        self,
        z_nm: Sequence[float],
        applied_field_G: Annotated[float, 0:None],
        dead_layer_nm: Annotated[float, 0:None] = 0.0,
        demagnetization_factor: Annotated[float, 0:1] = 0.0,
    ) -> Sequence[float]:
        """Calculate the current density profile.

        Args:
            z_nm: Depth below the surface (nm).
            applied_field_G: Applied magnetic field (G).
            dead_layer_nm: Non-superconducting dead layer thickness (nm).
            demagnetization_factor: Effective demagnetization factor.

        Returns:
            The current density profile at depth z (A m^-2).
        """

        # solve the GLE
        # self._solve()

        # calculate the geometrically enhanced field value
        effective_field_G = applied_field_G / (1.0 - demagnetization_factor)

        # calculate the prefactor for the conversion
        G_per_T = 1e4
        # nm_per_m = 1e9
        m_per_nm = 1e-9
        mu_0 = constants.value("vacuum mag. permeability") * G_per_T

        j_0 = -1.0 * effective_field_G / mu_0

        # correct the depth for the dead layer
        z_corr_nm = z_nm - dead_layer_nm

        # return the screening profile
        return np.piecewise(
            z_corr_nm,
            [
                z_corr_nm < 0.0,
                z_corr_nm >= 0.0,
            ],
            [
                lambda x: np.full(x.shape, 0.0 * j_0),
                lambda x: j_0 * self._sol.sol(x)[1] / m_per_nm,
            ],
        )

    def __call__(
        self,
        z_nm: Sequence[float],
        applied_field_G: Annotated[float, 0:None],
        dead_layer_nm: Annotated[float, 0:None] = 0.0,
        demagnetization_factor: Annotated[float, 0:1] = 0.0,
    ) -> Sequence[float]:
        """Calculate the Meissner screening profile (alias for self.screening_profile).

        Args:
            z_nm: Depth below the surface (nm).
            applied_field_G: Applied magnetic field (G).
            dead_layer_nm: Non-superconducting dead layer thickness (nm).
            demagnetization_factor: Effective demagnetization factor.

        Returns:
            The Meissner screening profile at depth z (G).
        """

        return self.screening_profile(
            z_nm,
            applied_field_G,
            dead_layer_nm,
            demagnetization_factor,
        )
