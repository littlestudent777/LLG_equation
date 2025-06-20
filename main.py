import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Callable


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
        # Create directory if needed
        os.makedirs(save_dir, exist_ok=True)

        # Create safe filename from experiment name
        filename = f"{name.replace(' ', '_')}.png"
        filepath = os.path.join(save_dir, filename)

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

        # Save
        plt.savefig(filepath, dpi=dpi, bbox_inches='tight')
        print(f"Saved: {filepath}")


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


import os

class Experiment:
    def __init__(self, name, solver_class, solver_params, m0, t_max, dt):
        self.name = name
        self.solver = solver_class(**solver_params)
        self.m0 = m0
        self.t_max = t_max
        self.dt = dt

    def run(self):
        self.tau_points, self.m_points = self.solver.solve(self.m0, self.t_max, self.dt)
        return self

    def plot(self):
        LLGSolver.plot_results(self.name, self.tau_points, self.m_points)


if __name__ == "__main__":

    # Initial position of the magnetization vector (randomly chosen)
    theta = np.deg2rad(30)
    m0 = np.array([np.cos(theta), np.sin(theta), 0])  # normalized

    # Time parameters
    t_max = 1e-7  # Maximum time in seconds
    dt = 1e-11  # Time step in seconds

    # Initialize solvers with different parameters
    experiments = [
        Experiment("External Field", ExternalFieldLLGSolver,
                   {"H_ext": np.array([800, 250, 1000])}, m0, t_max, dt),
        # Experiment("Demagnetization field (Cylinder)", DemagFieldLLGSolver,
        #            {"N": np.array([0.5, 0.5, 0])}, m0, t_max, dt),
        # Experiment("Anisotropy", AnisotropyLLGSolver,
        #            {}, m0, t_max, dt)
    ]

    for exp in experiments:
        exp.run().plot()
