import numpy as np
from LLG_solver import LLGSolver, ExternalFieldLLGSolver, DemagFieldLLGSolver, AnisotropyLLGSolver


class Experiment:
    """
    Handles execution and visualization of experiment.
    """
    def __init__(self, name, solver_class, solver_params, m0, t_max=1e-8, dt=1e-12):
        self.name = name
        self.solver = solver_class(**solver_params)
        self.m0 = m0
        self.t_max = t_max
        self.dt = dt

    def run(self):
        print("\n", self.name)
        print(f"Using t_max = {self.t_max:.2e} s, alpha = {self.solver.alpha}")
        self.tau_points, self.m_points = self.solver.solve(self.m0, self.t_max, self.dt)
        print(f"Final magnetization: {self.m_points[-1]}")

    def plot(self):
        LLGSolver.plot_results(self.tau_points, self.m_points)


if __name__ == "__main__":
    # Initial position of the magnetization vector (randomly chosen)
    theta = np.deg2rad(30)
    m0 = np.array([np.cos(theta), 0, np.sin(theta)])  # normalized
    print("Initial point: ", m0)

    # Initialize solvers with different parameters
    experiments = [
        Experiment("External Field", ExternalFieldLLGSolver,
                   {"H_ext": np.array([800, 250, 1000]), "alpha": 0.1}, m0),
        Experiment("Demagnetization field (Cylinder along Z)", DemagFieldLLGSolver,
                   {"N": np.array([0.5, 0.5, 0.0]), "alpha": 0.01}, m0, t_max=5e-9),
        Experiment("Anisotropy", AnisotropyLLGSolver,
                   {"alpha": 0.1}, m0)
    ]

    for exp in experiments:
        exp.run()
        exp.plot()

    # Study of dependence on alpha
    dif_alpha_exp = [
        Experiment("Demagnetization field (Plane XY)", DemagFieldLLGSolver,
                   {"N": np.array([0.0, 0.0, 1.0]), "alpha": 0.1}, m0, t_max=3e-10),
        Experiment("Demagnetization field (Plane XY)", DemagFieldLLGSolver,
                   {"N": np.array([0.0, 0.0, 1.0]), "alpha": 0.01}, m0, t_max=3e-9),
        Experiment("Demagnetization field (Plane XY)", DemagFieldLLGSolver,
                   {"N": np.array([0.0, 0.0, 1.0]), "alpha": 0.001}, m0, t_max=3e-8),
        Experiment("Demagnetization field (Plane XY)", DemagFieldLLGSolver,
                   {"N": np.array([0.0, 0.0, 1.0]), "alpha": 0.0001}, m0, t_max=3e-7),
    ]

    for exp in dif_alpha_exp:
        exp.run()
        exp.plot()
