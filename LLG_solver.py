import numpy as np
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from typing import List, Tuple


class LLGSolver(ABC):
    """
    Abstract class with Landau-Lifshitz-Gilbert (LLG) equation solver.

    Uses normalized quantities:
    - tau = gamma * M_s * t  (dimensionless time)
    - m = M / M_s            (dimensionless magnetization, euclidian_norm(m) = 1)
    """
    def __init__(self, alpha: float, gamma: float = 1.76e7, Ms: float = 1707):
        """
        Initialize model parameters.

        Parameters:
        ----------
        alpha : float
            Dimensionless damping constant
        gamma : float, default=1.76e7
            Gyromagnetic ratio [rad/(s·G)] (default corresponds to electron)
        Ms : float, default=1707
            Saturation magnetization [emu/cc] (default corresponds to Fe in room temp.)
        """
        self.gamma = gamma
        self.alpha = alpha
        self.Ms = Ms

    @abstractmethod
    def compute_effective_field(self, m: np.ndarray) -> np.ndarray:
        """
        Compute the effective field (must be implemented by subclasses)
        """
        pass

    def f(self, m: np.ndarray, h_eff: np.ndarray) -> np.ndarray:
        """
        LLG equation in dimensionless form: dm/dtau = -(m x h) - alpha * (m x (m x h))

        Parameters:
        ----------
        m : Magnetization vector
        h_eff : Effective field vector

        Returns:
        -----------
        np.ndarray
            Derivative dm/dtau
        """
        m_cross_h = np.cross(m, h_eff)
        return -(m_cross_h + self.alpha * np.cross(m, m_cross_h))

    def merson_step(self, m: np.ndarray, step: float, tol: float) -> Tuple[np.ndarray, float, bool]:
        """
        Single step of Merson's RK method with adaptive step size control

        Returns:
        -----------
        (new_m, new_step, step_accepted(bool flag))
        """
        h_eff = self.compute_effective_field(m)
        k1 = step * self.f(m, h_eff)

        m2 = m + k1 / 3
        h_eff = self.compute_effective_field(m2)
        k2 = step * self.f(m2, h_eff)

        m3 = m + k1 / 6 + k2 / 6
        h_eff = self.compute_effective_field(m3)
        k3 = step * self.f(m3, h_eff)

        m4 = m + k1 / 8 + 3 * k3 / 8
        h_eff = self.compute_effective_field(m4)
        k4 = step * self.f(m4, h_eff)

        m5 = m + k1 / 2 - 3 * k3 / 2 + 2 * k4
        h_eff = self.compute_effective_field(m5)
        k5 = step * self.f(m5, h_eff)

        # Local error estimate
        error = (2 * k1 - 9 * k3 + 8 * k4 - k5) / 30
        error_norm = np.linalg.norm(error)
        m_norm = np.linalg.norm(m)

        if error_norm >= tol * m_norm:
            return m, step / 2, False  # Reject step, reduce h

        # Accept step
        m_new = m + (k1 + 4 * k4 + k5) / 6
        m_new /= np.linalg.norm(m_new)  # Project to unit sphere

        if error_norm <= tol * m_norm / 32:
            step *= 2  # Can increase step

        return m_new, step, True

    def solve(self, m0: np.ndarray, t_max: float = 1e-8, dt: float = 1e-12,
              tol: float = 1e-5) -> Tuple[np.ndarray, List[np.ndarray]]:
        """
        Solve Cauchy problem with adaptive Merson's RK method

        Parameters:
        ----------
        m0 : Initial solution m(t=0)
        t_max : Upper limit for the time (find solution for t in [0, t_max])
        dt : Initial step size
        tol : Desired relative accuracy

        Returns:
        ----------
        Tuple:
        (tau_points: array of dimensionless time values, m_points: list of magnetization vectors at each time step)
        """
        # Transform to dimensionless time
        tau_max = self.gamma * self.Ms * t_max
        dtau = self.gamma * self.Ms * dt

        m = m0.copy()
        tau = 0.0

        tau_points = [tau]
        m_points = [m.copy()]

        while tau < tau_max:
            m_new, dtau_new, step_accepted = self.merson_step(m, dtau, tol)

            if step_accepted:
                m = m_new
                tau += dtau
                tau_points.append(tau)
                m_points.append(m.copy())

            dtau = dtau_new

        return np.array(tau_points), m_points

    @staticmethod
    def plot_results(tau_points: np.ndarray, m_points: List[np.ndarray]):
        """
        Plot magnetization dynamics over time.

        Parameters:
        ----------
            tau_points: Array of dimensionless time values.
            m_points: List of magnetization vectors at each time step.
        """

        # Not showing an empty graph if there are almost no points
        if len(m_points) <= 2:
            print("\nInitial point was an optimal point.")
        else:
            # Create plot
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


class AnisotropyLLGSolver(LLGSolver):
    """
    An inherited class that only takes into account the effects of anisotropy.
    """
    def __init__(self, K1: float = 4.2e5, K2: float = 1.5e5, **kwargs):
        """
        Parameters:
        ----------
        K1, K2 : float
            Cubic anisotropy constant for Fe [erg/cc]
        """
        super().__init__(**kwargs)
        self.K1 = K1
        self.K2 = K2
        self.Ms_sq = self.Ms * self.Ms  # Precomputed value for optimization

    def compute_effective_field(self, m: np.ndarray) -> np.ndarray:
        """
        Compute the dimensionless effective field from cubic anisotropy.
        h_eff[i] = H_eff[i]/M_s = - 1/M_s^2 dF/dm[i]

        Parameters:
        ----------
        m : Magnetization vector (normalized)

        Returns:
        -----------
        h_eff : Effective field vector
        """
        m1, m2, m3 = m
        h_eff = np.zeros(3)
        m1_sq, m2_sq, m3_sq = m1 * m1, m2 * m2, m3 * m3

        h_eff[0] = -2 * m1 * (self.K1 * (m2_sq + m3_sq) + self.K2 * m2_sq * m3_sq) / self.Ms_sq
        h_eff[1] = -2 * m2 * (self.K1 * (m1_sq + m3_sq) + self.K2 * m1_sq * m3_sq) / self.Ms_sq
        h_eff[2] = -2 * m3 * (self.K1 * (m1_sq + m2_sq) + self.K2 * m1_sq * m2_sq) / self.Ms_sq

        return h_eff


class ExternalFieldLLGSolver(LLGSolver):
    """
    An inherited class that only takes into account the influence of external field.
    """
    def __init__(self, H_ext: np.ndarray, **kwargs):
        """
        Parameters:
        ----------
        H_ext : np.ndarray
            External field value [G]
        """
        super().__init__(**kwargs)
        self.H_ext = H_ext
        # Directions to compare results to
        print("H_ext direction: ", H_ext / np.linalg.norm(H_ext))

    def compute_effective_field(self, m: np.ndarray) -> np.ndarray:
        """
        Compute the dimensionless effective field.
        h_eff[i] = H_eff[i]/M_s = - 1/M_s^2 dF/dm[i] = H_ext / M_s
        """
        return self.H_ext / self.Ms


class DemagFieldLLGSolver(LLGSolver):
    """
    An inherited class that only takes into account the influence demagnetization field.
    """
    def __init__(self, N : np.ndarray, **kwargs):
        """
        Parameters:
        ----------
        N : np.ndarray
            Demagnetizing factors [dimensionless], sum(N[i]) = 1
        """
        super().__init__(**kwargs)
        self.N = N

    def compute_effective_field(self, m: np.ndarray) -> np.ndarray:
        """
        Compute the dimensionless effective field from demagnetization.
        h_eff[i] = H_eff[i]/M_s = - 1/M_s^2 dF/dm[i] = - 4 * pi * N[i] * m[i]
        """
        return -4 * np.pi * self.N * m
