"""Crank-Nicolson example.

Adapted from: https://georg.io/2013/12/03/Crank_Nicolson

see also: https://math.stackexchange.com/a/3311598
"""

import numpy as np
from scipy import sparse

# from simulation.ciovati_model import D, q
from simulation.dissolution_species import dc_O_dt


class CNSolver:
    # TODO: fix docementation
    """Crank-Nicolson Solver for 1D Diffusion Problems.

    This class encapsulates the Crank-Nicolson method to solve diffusion equations
    in 1D with specified initial and boundary conditions.

    Attributes:
        D_u (float): Diffusion coefficient (in nm^2/s).
        u_0 (float): Initial concentration close to x = 0 (e.g., at. % nm).
        v_0 (float): Background concentration (e.g., at. % nm).
        t_h (float): Maximum time in hours.
        N_x (int): Number of spatial grid points.
        x_max (float): Maximum spatial boundary (nm).
        N_t (int): Number of time grid points.
        x_grid (np.ndarray): Spatial grid.
        t_grid (np.ndarray): Temporal grid.
        sigma_u (float): Proportionality term for Crank-Nicolson.
    """

    def __init__(self, cfg, temps_K, total_h, civ_model, U_initial=None):
        """Initialize the CNSolver with given parameters."""
        self.u_0 = cfg.initial.u0
        self.v_0 = cfg.initial.v0
        self.base_O = cfg.initial.base_O
        self.T = temps_K
        # TODO: fix D to use dissolution_species
        self.D = civ_model.D
        self.dc_o_dt = dc_O_dt
        self.q = civ_model.q

        # Spatial and temporal grids
        self.x_max = cfg.grid.x_max_nm
        self.N_x = cfg.grid.n_x
        self.t_h = total_h
        self.N_t = cfg.grid.n_t
        self.x_grid = np.linspace(0.0, self.x_max, self.N_x, dtype=np.double)
        self.dx = np.diff(self.x_grid)[0]

        self.s_per_h = 60.0 * 60.0
        self.t_max = self.t_h * self.s_per_h
        self.t_grid = np.linspace(0.0, self.t_max, self.N_t, dtype=np.double)
        self.dt = np.diff(self.t_grid)[0]

        # Initial concentration
        if U_initial is None:
            self.U_initial = sparse.csr_array(
                [self.v_0 / self.dx] + [self.base_O / self.dx] * (self.N_x - 1)
            )
        else:
            U_initial[0] += self.v_0 / self.dx
            self.U_initial = sparse.csr_array(U_initial)

        # Constants
        self.D_u_max = self.D(
            np.max(self.T)
        )  # Maximum diffusion coefficient (in nm^2/s)
        self.r_max = (np.max(self.D_u_max) * self.dt) / (self.dx * self.dx)
        self.stability = "STABLE" if self.r_max <= 0.5 else "POTENTIAL OSCILLATIONS"

        self.r = float
        self.sigma = float

    def gen_sparse_matrices(self, i):
        """Generate the sparse matrices "A" and "B" used by the Crank-Nicolson method.

        Args:
            N_x: Dimension of (square) matrices.
            sigma: The "nudging" parameter.

        Returns:
            The (sparse) matrices A and B.
        """
        # Initialize the diffusion coefficient for time t
        self.D_u = self.D(self.T[i])  # Diffusion coefficient (in nm^2/s)
        # Update the stability parameter
        self.r = (self.D_u * self.dt) / (self.dx * self.dx)
        # Crank-Nicolson proportionality term
        self.sigma = 0.5 * self.r

        # common sparse matrix parameters
        _offsets = [1, 0, -1]
        _shape = (self.N_x, self.N_x)
        _format = "csr"

        # define matrix A's elements
        _A_upper = [-self.sigma]
        _A_diag = (
            [1 + self.sigma] + [1 + 2 * self.sigma] * (self.N_x - 2) + [1 + self.sigma]
        )
        _A_lower = [-self.sigma]
        _A_elements = [_A_upper, _A_diag, _A_lower]

        # create matrix A
        _A = sparse.diags_array(
            _A_elements,
            offsets=_offsets,
            shape=_shape,
            format=_format,
        )

        # define matrix B's elements
        _B_upper = [self.sigma]
        _B_diag = (
            [1 - self.sigma] + [1 - 2 * self.sigma] * (self.N_x - 2) + [1 - self.sigma]
        )
        _B_lower = [self.sigma]
        _B_elements = [_B_upper, _B_diag, _B_lower]

        # create matrix A
        _B = sparse.diags_array(
            _B_elements,
            offsets=_offsets,
            shape=(self.N_x, self.N_x),
            format=_format,
        )

        # return both matrix A and B
        return _A, _B

    def get_oxygen_profile(self):
        """Solve the diffusion equation using the Crank-Nicolson method.

        Returns:
            np.ndarray: The solution record (time x space).
        """
        # Initial condition: Concentration is all in the first spatial bin
        # U_initial = sparse.csr_array([self.v_0 / self.dx] + [0] * (self.N_x - 1))
        U_record = np.zeros((self.N_t, self.N_x), dtype=np.double)

        for i, t in enumerate(self.t_grid):
            # if self.t_grid[i] == self.t_grid[-1]:
            #     print(f"Final time step: {i} / {self.N_t - 1} (t = {t:.2f} s), (T = {self.T[i]:.2f} K)")
            # if self.t_grid[i] == self.t_grid[100]:
            #     print(f"step time step : {i} / {self.N_t - 1} (t = {t:.2f} s), (T = {self.T[i]:.2f} K)")
            # if self.t_grid[i] == self.t_grid[1800]:
            #     print(f"bake time step : {i} / {self.N_t - 1} (t = {t:.2f} s), (T = {self.T[i]:.2f} K)")
            if i == 0:
                # Record the initial condition
                U_record[i] = self.U_initial.toarray()
            else:
                # Source term (plane source at x = 0)
                # f_vec = sparse.csr_array([self.q(t, self.T[i]) * (self.dt / self.dx)] + [0] * (self.N_x - 1))
                f_vec = sparse.csr_array(
                    [self.dc_o_dt(t, self.T[i], self.u_0, 0) * (self.dt / self.dx)]
                    + [0] * (self.N_x - 1)
                )

                # Generate matrices (could be precomputed if D_u is constant)
                A, B = self.gen_sparse_matrices(i)

                # Solve for the next time step
                U = sparse.csr_array(U_record[i - 1])
                U_new = sparse.linalg.spsolve(A, B @ U + f_vec)
                U_record[i] = U_new

        return U_record
