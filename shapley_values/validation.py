import numpy as np
from numpy.core.fromnumeric import squeeze
import pandas as pd
import shap
from typing import List

from shapley_values.explanations import why_best, why_objective_i, why_worst, largest_conflict, how_to_improve_objective_i
from shapley_values.utilities import generate_missing_data, generate_black_box
from desdeo_problem.problem import DiscreteDataProblem
# from desdeo_tools.scalarization import DiscreteScalarizer
# from desdeo_tools.solver import DiscreteMinimizer
from desdeo_tools.scalarization import SimpleASF


def generate_validation_data_global(df: pd.DataFrame, variable_names: List[str], objective_names: List[str], n_missing_data: int = 200, n_runs: int = 10, ref_delta: float = 0.1):
    pareto_f = df.to_numpy()

    n_objectives = len(objective_names)
    ideal = np.min(pareto_f[:, 0:n_objectives], axis=0)
    nadir = np.max(pareto_f[:, 0:n_objectives], axis=0)

    problem = DiscreteDataProblem(df, variable_names, objective_names, nadir, ideal)

    asf = SimpleASF(np.array([1 for _ in range(n_objectives)]))

    bb = generate_black_box(problem, asf)

    # start generating data here
    run_i = 0
    # Datum format:
    # original ref point,
    # original solution,
    # index of objective DM wants to improve,
    # suggestion based on computed Shapley values,
    # index of objective to be improved in the reference point (-1 if none suggested),
    # index of objective to be worsened in the reference point (-1 if none suggested),
    # new reference point generated based on suggestion,
    # new solution based on new reference point,
    data = pd.DataFrame(
        columns=(
            list(f"Original ref point f_{i}" for i in range(1, n_objectives+1)) +
            list(f"Original solution f_{i}" for i in range(1, n_objectives+1)) +
            ["Index of solution DM wishes to improve"] +
            ["Suggestion with explanation"] +
            ["Index of objective to be improved in the reference point"] +
            ["Index of objective to be worsened in the reference point"] +
            list(f"New ref point based on suggestion f_{i}" for i in range(1, n_objectives+1)) +
            list(f"New solution based on new ref point f_{i}" for i in range(1, n_objectives+1)) +
            ["Was the desired effect achieved?"]
        )
    )

    fail_count = 0

    while run_i < n_runs:
        print(f"Run {run_i+1} our of {n_runs}...")

        # generate a random reference point between the ideal and nadir
        ref_point = np.array([np.random.uniform(low=ideal[i], high=nadir[i]) for i in range(ideal.shape[0])])
        
        # TODO: for now, the missing data is always generated between the ideal and nadir, for a problem with
        # a non-convex, non-linear Pareto front, the missing data is best generated around the
        # given reference point to capture the local nature of the front... Doing what is done now, if fine for a problem
        # like DTLZ2.
        low = ideal
        high = nadir
        missing_data = generate_missing_data(n_missing_data, low, high)

        # we need a new explainer since in the general case the missing_data may change based on the given reference point.
        explainer = shap.KernelExplainer(bb, missing_data)

        # the computed solution by the black box
        solution = bb(np.atleast_2d(ref_point))

        # comput the Shapley values for the given reference point
        shap_values = np.array(explainer.shap_values(ref_point))

        # check the Shapley values and figure out how to improve a random objective (zero indexed)
        to_be_improved = np.random.randint(0, n_objectives)

        explanation, improve_i, worsen_i, _ = how_to_improve_objective_i(shap_values, to_be_improved, ref_point, solution)

        # save the original ref_point before modifying it
        original_ref_point = np.copy(ref_point)

        # check if something is to be improved and improve it (notice that we assume minimization)
        if improve_i > -1:
            # change the ref_point accordingly
            ref_point[improve_i] -= ref_delta * (nadir[improve_i] - ideal[improve_i])
            # if the new value is less than the ideal value, then set the new value to the ideal value
            if ref_point[improve_i] < ideal[improve_i]:
                ref_point[improve_i] = ideal[improve_i]


        # check if something is to be worsened (notice that we assume minimization)
        if worsen_i > -1:
            # change the ref_point accordingly
            # ref_point[worsen_i] += ref_delta * ref_point[worsen_i]
            ref_point[worsen_i] += ref_delta * (nadir[worsen_i] - ideal[worsen_i])
            # if the new value is more than the nadir value, then set the new value to the nadir value
            if ref_point[worsen_i] > nadir[worsen_i]:
                ref_point[worsen_i] = nadir[worsen_i]


        # compute the new solution based on the modified ref_point
        new_solution = bb(np.atleast_2d(ref_point))

        # check if the desired effect was achieved
        if new_solution.squeeze()[to_be_improved] < solution.squeeze()[to_be_improved]:
            # print(f"Effect was achieved! To be improved {to_be_improved}\nOld solution {solution.squeeze()[to_be_improved]}\nNew solution {new_solution.squeeze()[to_be_improved]}")
            effect_was_achieved = 1
        elif np.equal(new_solution.squeeze()[to_be_improved], solution.squeeze()[to_be_improved]):
            # print(f"Effect was _NOT_ achieved due to no change! To be improved {to_be_improved}\nOld solution {solution.squeeze()[to_be_improved]}\nNew solution {new_solution.squeeze()[to_be_improved]}")
            effect_was_achieved = 0
        else:
            # print(f"Effect was _NOT_ achieved! To be improved {to_be_improved}\nOld solution {solution.squeeze()[to_be_improved]}\nNew solution {new_solution.squeeze()[to_be_improved]}")
            fail_count += 1
            effect_was_achieved = -1


        # compile row to be added to the data and add it
        datum = (
            original_ref_point.squeeze().tolist() +
            solution.squeeze().tolist() +
            [to_be_improved + 1] +
            [explanation] +
            [improve_i + 1 if improve_i != -1 else -1] +
            [worsen_i + 1 if worsen_i != -1 else -1] +
            ref_point.squeeze().tolist() +
            new_solution.squeeze().tolist() +
            [effect_was_achieved]
        )

        data.loc[run_i] = datum

        run_i += 1
    
    print(f"Out of {n_runs} runs {n_runs - fail_count} succeeded.")

    # data.to_excel("/home/kilo/Downloads/run_DTLZ2_5_objectives_n_5000_missing-data_500_delta_30.xlsx")
    


if __name__ == "__main__":
    # df = pd.read_csv("./data/DTLZ2_8x_5f.csv")
    # generate_validation_data_global(df, ["x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8"], ["f1", "f2", "f3", "f4", "f5"], n_runs=500, n_missing_data=200, ref_delta=0.2)

    df = pd.read_csv("./data/river_pollution_2500.csv")
    # OBS! this is with _global_ missing data!!!
    generate_validation_data_global(df, ["x_1", "x_2"], ["f_1", "f_2", "f_3", "f_4", "f_5"], n_runs=500, n_missing_data=200, ref_delta=0.1)