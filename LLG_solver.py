import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Callable
import os

class LLGSolver:
    """
    Base class with common Landau-Lifshitz-Gilbert (LLG) equation solver methods.

    Uses normalized quantities:
    - tau = gamma * M_s * t  (dimensionless time)
    - m = M / M_s            (dimensionless magnetization, norm(m)= 1)
    """
    def __init__(self, alpha=0.01, gamma=1.76e7, Ms=1707):
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
        Ms_sq : float
            Precomputed value for optimization (Ms^2) [emu^2/c^6]
        """
        self.gamma = gamma
        self.alpha = alpha
        self.Ms = Ms

    def compute_effective_field(self, m: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Subclasses must implement this method.")

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

    def merson_step(self, m: np.ndarray, step: float, eps: float) -> Tuple[np.ndarray, float, bool]:
        """
        Single step of Merson's RK method with adaptive step size control
        Returns: (new_m, new_h, step_accepted)
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

        if error_norm >= eps * m_norm:
            return m, step / 2, False  # Reject step, reduce h

        # Accept step
        m_new = m + (k1 + 4 * k4 + k5) / 6
        m_new /= np.linalg.norm(m_new)  # Project to unit sphere

        if error_norm <= eps * m_norm / 32:
            step *= 2  # Can increase step

        return m_new, step, True

    def solve(self, m0: np.ndarray, t_max: float, dt: float,
              tol: float = 1e-5) -> Tuple[np.ndarray, List[np.ndarray]]:
        """
        Solve with adaptive Merson's RK method
        eps - desired relative accuracy
        h_init - initial step size (dimensionless)
        """
        tau_max = self.gamma * self.Ms * t_max
        m = m0.copy()
        tau = 0.0
        dtau = self.gamma * self.Ms * dt

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
    def plot_results(name: str, tau_points: np.ndarray, m_points: List[np.ndarray], save_dir="results", dpi=300):
        """
        Plot and save the results.

        Parameters:
        -----------
        save_dir : str
            Directory to save the plots
        show_plot : bool
            Whether to display the plot
        dpi : int
            Image resolution
        """

        # Not showing an empty graph if there are almost no points
        if len(m_points) <= 2:
            print("\nInitial point was an optimal point.")
        else:
            # # Create directory if needed
            # os.makedirs(save_dir, exist_ok=True)

            # Create filename from experiment name
            # filename = f"{name.replace(' ', '_')}.png"
            # filepath = os.path.join(save_dir, filename)

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
            # # Save
            # plt.savefig(filepath, dpi=dpi, bbox_inches='tight')
            # print(f"Saved: {filepath}")


class AnisotropyLLGSolver(LLGSolver):
    """Solver with cubic anisotropy."""
    def __init__(self, K1=4.2e5, K2=1.5e5, **kwargs):
        super().__init__(**kwargs)
        self.K1 = K1
        self.K2 = K2
        self.Ms_sq = self.Ms * self.Ms

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


class ExternalFieldLLGSolver(LLGSolver):
    """Solver with external field."""
    def __init__(self, H_ext=np.zeros(3), **kwargs):
        super().__init__(**kwargs)
        self.H_ext = H_ext
        # Print to compare results
        print("H_ext direction: ", H_ext / np.linalg.norm(H_ext))

    def compute_effective_field(self, m: np.ndarray) -> np.ndarray:
        return self.H_ext / self.Ms


class DemagFieldLLGSolver(LLGSolver):
    def __init__(self, N : np.ndarray, **kwargs):
        super().__init__(**kwargs)
        self.N = N

    def compute_effective_field(self, m: np.ndarray) -> np.ndarray:
        h_eff = np.zeros(3)
        for i in range(3):
            h_eff[i] = - 4 * np.pi * self.N[i] * m[i]

        return h_eff
