import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple


class LLGSolver:
    """
    Class for solving the Landau-Lifshitz-Gilbert (LLG) equation.

    Uses normalized quantities:
    - tau = gamma * M_s * t  (dimensionless time)
    - m = M / M_s            (dimensionless magnetization, norm(m)= 1)
    """

    def __init__(self, gamma: float = 1.76e7, alpha: float = 0.1,
                 Ms: float = 1707, K1: float = 4.2e5, K2: float = 1.5e5):
        """
        Initialize model parameters.

        Parameters:
        ----------
        gamma : float
            Gyromagnetic ratio [rad/(s·G)]
        alpha : float
            Dimensionless damping constant
        Ms : float
            Saturation magnetization [emu/cc]
        K1, K2 : float
            Cubic anisotropy constant for Fe [erg/cc]
        """
        self.gamma = gamma
        self.alpha = alpha
        self.Ms = Ms
        self.K1 = K1
        self.K2 = K2
        self.Ms_sq = Ms * Ms  # Precomputed value for optimization

    def compute_effective_field(self, m: np.ndarray) -> np.ndarray:
        """
        Compute the dimensionless effective field from cubic anisotropy.
        h_eff = H_eff / Ms

        Parameters:
        ----------
        m : np.ndarray
            Magnetization vector (normalized)

        Returns:
        -----------
        h_eff : np.ndarray
            Effective field vector
        """
        m1, m2, m3 = m
        h_eff = np.zeros(3)
        m1_sq, m2_sq, m3_sq = m1 * m1, m2 * m2, m3 * m3

        h_eff[0] = -2 * m1 * (self.K1 * (m2_sq + m3_sq) + self.K2 * m2_sq * m3_sq) / self.Ms_sq
        h_eff[1] = -2 * m2 * (self.K1 * (m1_sq + m3_sq) + self.K2 * m1_sq * m3_sq) / self.Ms_sq
        h_eff[2] = -2 * m3 * (self.K1 * (m1_sq + m2_sq) + self.K2 * m1_sq * m2_sq) / self.Ms_sq

        return h_eff

    def f(self, m: np.ndarray, h_eff: np.ndarray) -> np.ndarray:
        """
        LLG equation in dimensionless form: dm/dtau = -(m x h) - alpha * (m x (m x h))

        Parameters:
        ----------
        m : np.ndarray
            Magnetization vector
        h_eff : np.ndarray
            Effective field vector

        Returns:
        -----------
        np.ndarray
            Derivative dm/dtau
        """
        m_cross_h = np.cross(m, h_eff)
        return -(m_cross_h + self.alpha * np.cross(m, m_cross_h))

    def rk4_step(self, m: np.ndarray, step: float) -> np.ndarray:
        """
        Integration step using 4th order Runge-Kutta method with projection onto sphere.

        Parameters:
        ----------
        m : np.ndarray
            Current magnetization vector
        step : float
            Dimensionless time step

        Returns:
        -----------
        np.ndarray
            New magnetization vector after integration step
        """
        h_eff = self.compute_effective_field(m)
        k1 = self.f(m, h_eff)

        m2 = (m + 0.5 * step * k1)
        h_eff2 = self.compute_effective_field(m2)
        k2 = self.f(m2, h_eff2)

        m3 = (m + 0.5 * step * k2)
        h_eff3 = self.compute_effective_field(m3)
        k3 = self.f(m3, h_eff3)

        m4 = (m + step * k3)
        h_eff4 = self.compute_effective_field(m4)
        k4 = self.f(m4, h_eff4)

        m_new = m + (step / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

        # Projection on the unit sphere (euclidian_norm(m) = 1)
        m_new /= np.linalg.norm(m_new)

        return m_new

    def solve(self, m0: np.ndarray, t_max: float, dt: float,
              early_stopping: bool = True, tol: float = 1e-5) -> Tuple[np.ndarray, List[np.ndarray]]:
        """
        Solve the LLG equation for given initial conditions and time parameters.

        Parameters:
        ----------
        m0 : np.ndarray
            Initial magnetization vector (normalized)
        t_max : float
            Maximum time in seconds (solving problem in the interval [0, t_max]
        dt : float
            Integration time step in seconds
        early_stopping : bool
            Flag for early termination when reaching needed accuracy
        tol : float
            Tolerance for determining steady state

        Returns:
        -----------
        Tuple[np.ndarray, List[np.ndarray]]
            Tuple of time points array and list of magnetization vectors
        """
        # Convert to dimensionless form
        tau_max = self.gamma * self.Ms * t_max
        dtau = self.gamma * self.Ms * dt

        m = m0.copy()
        tau_points = []
        m_points = []
        tau = 0.0
        m_prev = m.copy()

        while tau <= tau_max:
            tau_points.append(tau)
            m_points.append(m.copy())

            m = self.rk4_step(m, dtau)
            tau += dtau

            if early_stopping and np.linalg.norm(m_prev - m) < tol:  # Do I really need this???
                print("Algorithm stopped at:\nm = ", m, "\ntau = ", tau)
                break

            m_prev = m.copy()

        return np.array(tau_points), m_points

    @staticmethod
    def plot_results(tau_points: np.ndarray, m_points: List[np.ndarray]):
        """
        Visualize the solution: Magnetization-Time graph.

        Parameters:
        ----------
        tau_points : np.ndarray
            Array of dimensionless time points
        m_trajectory : List[np.ndarray]
            List of magnetization vectors
        """

        plt.figure(figsize=(12, 6))
        for i, axis in enumerate(['x', 'y', 'z']):
            plt.plot(tau_points, np.array([m[i] for m in m_points]), label=f'$m_{axis}$', linewidth=2)

        plt.xlabel('Dimensionless time $\\tau = \\gamma M_s t$', fontsize=12)
        plt.ylabel('Magnetization components $m_i$', fontsize=12)
        plt.title('Magnetization dynamics', fontsize=14)
        plt.legend(fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    # Initialize solver with default parameters
    solver = LLGSolver()

    # Initial position of the magnetization vector
    theta = np.deg2rad(30)
    m0 = np.array([np.cos(theta), np.sin(theta), 0])  # normalized

    # Time parameters
    t_max = 1e-8  # Maximum time in seconds
    dt = 1e-12  # Time step in seconds

    # Solve the equation
    tau_points, m_trajectory = solver.solve(m0, t_max, dt)

    # Visualize results
    LLGSolver.plot_results(tau_points, m_trajectory)
