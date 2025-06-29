import numpy as np
from LLG_solver import LLGSolver, AnisotropyEffect, ExternalFieldEffect, DemagFieldEffect


class Experiment:
    """
    Handles execution and visualization of experiment.
    """
    def __init__(self, name, effects, solver_params, m0, t_max=1e-8, dt=1e-12):
        self.name = name
        self.effects = effects
        self.solver_params = solver_params
        self.m0 = m0
        self.t_max = t_max
        self.dt = dt

    def run(self):
        print("\n", self.name)
        solver = LLGSolver(
            effects=self.effects,
            **self.solver_params
        )
        print(f"Using t_max = {self.t_max:.2e} s, alpha = {solver.alpha}")
        self.tau_points, self.m_points = solver.solve(self.m0, self.t_max, self.dt)
        print(f"Final magnetization: {self.m_points[-1]}")

    def plot(self):
        LLGSolver.plot_results(self.tau_points, self.m_points)


if __name__ == "__main__":
    # Initial position of the magnetization vector
    theta = np.deg2rad(30)
    m0 = np.array([np.cos(theta), 0, np.sin(theta)])  # normalized
    print("Initial point: ", m0)

    # Define common solver parameters
    base_params = {
        "alpha": 0.1,
        "gamma": 1.76e7,
        "Ms": 1707
    }

    # Initialize experiments with different effect combinations
    experiments = [
        Experiment(
            "External Field Only",
            effects=[ExternalFieldEffect(H_ext=np.array([800, 250, 1000]))],
            solver_params={**base_params},
            m0=m0
        ),
        Experiment(
            "Demagnetization Field (Cylinder along Z)",
            effects=[DemagFieldEffect(N=np.array([0.5, 0.5, 0.0]))],
            solver_params={**base_params, "alpha": 0.01},
            m0=m0,
            t_max=5e-9
        ),
        Experiment(
            "Anisotropy Only",
            effects=[AnisotropyEffect(K1=4.2e5, K2=1.5e5)],
            solver_params={**base_params},
            m0=m0
        ),
        Experiment(
            "Combined Effects",
            effects=[
                AnisotropyEffect(K1=4.2e5, K2=1.5e5),
                ExternalFieldEffect(H_ext=np.array([800, 250, 1000])),
                DemagFieldEffect(N=np.array([0.1, 0.1, 0.8]))
            ],
            solver_params={**base_params, "alpha": 0.1},
            m0=m0,
            t_max=1e-9
        )
    ]

    # Run all experiments
    for exp in experiments:
        exp.run()
        exp.plot()
