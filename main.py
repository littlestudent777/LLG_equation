import numpy as np
from LLG_solver import LLGSolver, AnisotropyEffect, ExternalFieldEffect, DemagFieldEffect


class Experiment:
    """
    Handles execution and visualization of experiment.
    """
    def __init__(self, name, effects, solver_params, m0, m_eq=None, t_max=1e-8, dt=1e-12):
        self.name = name
        self.effects = effects
        self.solver_params = solver_params
        self.m0 = m0
        self.m_eq = m_eq  # optional
        self.t_max = t_max
        self.dt = dt

    def run(self, compute_freq=False):
        print("\n", self.name)
        solver = LLGSolver(
            effects=self.effects,
            **self.solver_params
        )
        print(f"Using t_max = {self.t_max:.2e} s, alpha = {solver.alpha}")
        self.tau_points, self.m_points = solver.solve(self.m0, self.t_max, self.dt)
        final_m = self.m_points[-1]
        # print(f"Final magnetization: {final_m}")

        if compute_freq:
            print("Frequency:")
            m_eq = self.m_eq
            # If the equilibrium magnetization is not given, we use the final value of m
            if (m_eq is None):
                m_eq = final_m
                print("Frequency will be calculated using final value of m.\n "
                      "If it didn't converge, results will not be accurate.")
            # Find the component along which the magnetization is directed the least
            min_ind = np.argmin(np.abs(m_eq))
            m_i = np.array([m[min_ind] for m in self.m_points])

            freq_i = solver.compute_frequency(self.tau_points, m_i)

            # Convert frequency from a dimensionless quantity to a quantity rad/s
            term = 2 * np.pi * solver.gamma * solver.Ms
            print(f"Frequency (dimensionless): {freq_i:.2f}, \nangular frequency in [rad/s]: {freq_i * term:.3e}")

    def plot(self):
        LLGSolver.plot_results(self.tau_points, self.m_points)


def get_orthogonal_vector(v):
    # Select a vector that is not parallel to v
    if abs(v[0]) < 0.9:
        w0 = np.array([1.0, 0.0, 0.0])
    elif abs(v[1]) < 0.9:
        w0 = np.array([0.0, 1.0, 0.0])
    else:
        w0 = np.array([0.0, 0.0, 1.0])
    w = w0 - np.dot(w0, v) * v
    return w / np.linalg.norm(w)

def rotate_vector(v, angle_deg):
    theta = np.radians(angle_deg)
    w = get_orthogonal_vector(v)
    u = (v * np.cos(theta) +
         np.cross(w, v) * np.sin(theta) +
         w * np.dot(w, v) * (1 - np.cos(theta)))
    u = u / np.linalg.norm(u)
    print(np.arccos(np.dot(u, v)))
    return u


if __name__ == "__main__":
    # # Initial position of the magnetization vector
    # theta = np.deg2rad(20)
    # m0 = np.array([np.cos(theta), 0, np.sin(theta)])  # normalized
    # print("Initial point: ", m0)

    # Define common solver parameters
    base_params = {
        "alpha": 0.015,
        "gamma": 1.76e7,
        "Ms": 1707
    }

    # Equilibrium magnetization for the following experiments
    m1 = np.array([0.50306325, 0.20961179, 0.83844515])
    m2 = np.array([0.60259486, 0.75324358, 0.26363525])
    m3 = np.array([1.0, 0.0, 0.0])
    m4 = np.array([0.57735, 0.57735, 0.57735])
    m5 = m3
    m6 = np.array([0.97852257, 0.19680969, 0.0613149])

    # Initialize experiments with different effect combinations
    experiments = [
        Experiment(
            "External Field Only",
            effects=[ExternalFieldEffect(H_ext=np.array([600, 250, 1000]))],
            solver_params={**base_params},
            m0=rotate_vector(m1, 1),
            m_eq=m1
        ),
        Experiment(
            "External Field Only",
            effects=[ExternalFieldEffect(H_ext=np.array([800, 1000, 350]))],
            solver_params={**base_params},
            m0=rotate_vector(m2, 1),
            m_eq=m2
        ),
        Experiment(
            "Anisotropy Only",
            effects=[AnisotropyEffect(K1=4.2e5, K2=1.5e5)],
            solver_params={**base_params},
            m0=rotate_vector(m3, 1),
            m_eq=m3
        ),
        Experiment(
            "Anisotropy Only",
            effects=[AnisotropyEffect(K1=-4.2e5, K2=1.0e5)],
            solver_params={**base_params},
            m0=rotate_vector(m4, 1),
            m_eq=m4,
            t_max=3e-8
        ),
        Experiment(
            "Demagnetization Field (Cylinder, x is inf)",
            effects=[DemagFieldEffect(N=np.array([0.0, 0.5, 0.5]))],
            solver_params={**base_params, "alpha": 0.001},
            m0=rotate_vector(m5, 1),
            m_eq=m5
        ),
        Experiment(
            "Combined Effects",
            effects=[
                AnisotropyEffect(K1=4.2e5, K2=1.5e5),
                ExternalFieldEffect(H_ext=np.array([800, 250, 1000])),
                DemagFieldEffect(N=np.array([0.1, 0.1, 0.8]))
            ],
            solver_params={**base_params, "alpha": 0.005},
            m0=rotate_vector(m6, 1),
            m_eq=m6
        )
    ]

    # Run all experiments
    for exp in experiments:
        exp.run(compute_freq=True)
        exp.plot()
