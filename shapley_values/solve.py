import pathlib

import numpy as np
import pandas as pd
from desdeo_emo.EAs import NSGAIII
from desdeo_problem.problem import MOProblem
from desdeo_problem.testproblems import test_problem_builder


def compute_pareto_front(problem: MOProblem, file_path: str, file_name: str):
    """Compute a representation of the Pareto optimal front for an MOProblem and save it to a file."""
    evolver = NSGAIII(problem, interact=False, population_size=2500)

    while evolver.continue_evolution():
        evolver.iterate()

    xs, fs = evolver.end()

    numpy_data = np.hstack((fs, xs))
    df = pd.DataFrame(data=numpy_data, columns=problem.get_objective_names()[0] + problem.get_variable_names())

    df.to_csv(file_path + "/" + file_name, index=False)

    print(f"Computed a total of {fs.shape[0]} solutions.")
    return


def main():
    problem = test_problem_builder("DTLZ2", 5, 3)

    file_path = str(pathlib.Path(__file__).parent.resolve())
    file_name = "../data/DTLZ2_5x_3f.csv"

    compute_pareto_front(problem, file_path=file_path, file_name=file_name)
    return 0


if __name__ == "__main__":
    main()
