import pathlib
from desdeo_emo import population

import numpy as np
import pandas as pd
from desdeo_emo.EAs import NSGAIII, MOEA_D, RVEA, PPGA
from desdeo_emo.population import Population
from desdeo_problem.problem import MOProblem
from desdeo_problem.testproblems import test_problem_builder
from shapley_values.problems import river_pollution_problem, car_crash_problem


def compute_pareto_front(
    problem: MOProblem, file_path: str, file_name: str, pop_size: int = 2500
):
    """Compute a representation of the Pareto optimal front for an MOProblem and save it to a file."""
    evolvers = []

    evolver_nsga3 = NSGAIII(problem, interact=False, population_size=pop_size)
    evolvers.append(evolver_nsga3)

    evolver_moead = MOEA_D(
        problem, interact=False, population_params={"pop_size": pop_size}
    )
    evolvers.append(evolver_moead)

    evolver_rvea = RVEA(problem, interact=False, population_size=pop_size)
    evolvers.append(evolver_rvea)

    xss, fss = [], []
    for evolver in evolvers:
        print(f"Starting evolver: {type(evolver)}")
        while evolver.continue_evolution():
            evolver.iterate()

        xs, fs = evolver.end()
        xss.append(xs)
        fss.append(fs)

    xss_stack = np.vstack(xss)
    fss_stack = np.vstack(fss)

    population = Population(problem, pop_size=len(evolvers) * pop_size)
    population.individuals = xss_stack
    population.objectives = fss_stack

    non_dom = population.non_dominated_objectives()

    xs_non_dom = population.individuals[non_dom]
    fs_non_dom = population.objectives[non_dom]

    numpy_data = np.hstack((fs_non_dom, xs_non_dom))
    df = pd.DataFrame(
        data=numpy_data,
        columns=list(np.squeeze(problem.get_objective_names()))
        + problem.get_variable_names(),
    )

    fhandle = file_path + "/" + file_name + f"{fs_non_dom.shape[0]}" + ".csv"
    df.to_csv(fhandle, index=False)

    print(f"Computed a total of {fs.shape[0]} solutions.")

    return 0


def main():
    # problem_x = test_problem_builder("DTLZ2", 8, 5)
    # problem = test_problem_builder("DTLZ2", 5, 3)
    problem = river_pollution_problem()

    file_path = str(pathlib.Path(__file__).parent.resolve())
    file_name = "../data/river_pollution_"

    compute_pareto_front(
        problem, file_path=file_path, file_name=file_name, pop_size=10000
    )
    return 0


if __name__ == "__main__":
    main()
