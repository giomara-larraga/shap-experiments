import numpy as np
from numpy.core.fromnumeric import squeeze
import pandas as pd
import shap
from typing import List

from shapley_values.explanations import why_best, why_objective_i, why_worst, largest_conflict, how_to_improve_objective_i
from shapley_values.utilities import generate_missing_data,generate_missing_data_even, generate_black_box, Normalizer
from desdeo_problem.problem import DiscreteDataProblem
# from desdeo_tools.scalarization import DiscreteScalarizer
# from desdeo_tools.solver import DiscreteMinimizer
from desdeo_tools.scalarization import SimpleASF, PointMethodASF


def generate_validation_data_global(df: pd.DataFrame, variable_names: List[str], objective_names: List[str], n_missing_data: int = 200, n_runs: int = 10, ref_delta: float = 0.1, file_name: str = ""):
    pareto_f = df.to_numpy()

    n_objectives = len(objective_names)
    ideal = np.min(pareto_f[:, 0:n_objectives], axis=0)
    nadir = np.max(pareto_f[:, 0:n_objectives], axis=0)

    problem = DiscreteDataProblem(df, variable_names, objective_names, nadir, ideal)

    # asf = SimpleASF(np.array([1 for _ in range(n_objectives)]))
    asf = PointMethodASF(nadir, ideal)

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
            ["Was the desired effect achieved?"] +
            ["Case identifier"] +
            ["Change in reference point values (multiplier w.r.t. nadir - ideal)"]
        )
    )

    fail_count = 0

    low = ideal
    high = nadir
    # nadir always > ideal
    delta = ref_delta * (nadir - ideal)

    normalizer = Normalizer(ideal, nadir)

    while run_i < n_runs:
        print(f"Run {run_i+1} out of {n_runs}...")

        # generate a random reference point between the ideal and nadir
        ref_point = np.array([np.random.uniform(low=ideal[i], high=nadir[i]) for i in range(ideal.shape[0])])
        
        # We generate data also outside the area dicated by the ideal and nadir points to have more accurate explanations
        # for reference point that reside on the edge of the area.
        # missing_data = generate_missing_data(n_missing_data, low-delta, high+delta)
        missing_data = generate_missing_data_even(n_missing_data, low-delta, high+delta)

        explainer = shap.KernelExplainer(bb, missing_data)

        # the computed solution by the black box
        solution = bb(np.atleast_2d(ref_point))

        # compute the Shapley values for the given reference point
        shap_values = normalizer.scale(np.array(explainer.shap_values(ref_point)))

        # check the Shapley values and figure out how to improve a random objective (zero indexed)
        to_be_improved = np.random.randint(0, n_objectives)

        explanation, improve_i, worsen_i, case_i = how_to_improve_objective_i(shap_values, to_be_improved, ref_point, solution)

        # save the original ref_point before modifying it
        original_ref_point = np.copy(ref_point)

        # check if something is to be improved and improve it (notice that we assume minimization)
        if improve_i > -1:
            # change the ref_point accordingly
            ref_point[improve_i] -= delta[improve_i]


        # check if something is to be worsened (notice that we assume minimization)
        if worsen_i > -1:
            # change the ref_point accordingly
            # ref_point[worsen_i] += ref_delta * ref_point[worsen_i]
            ref_point[worsen_i] += delta[worsen_i]


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
            [effect_was_achieved] +
            [case_i] +
            [delta]
        )

        data.loc[run_i] = datum

        run_i += 1
    
    print(f"Out of {n_runs} runs {n_runs - fail_count} succeeded.")

    data.to_excel(file_name)
    


if __name__ == "__main__":
    # df = pd.read_csv("./data/DTLZ2_8x_5f.csv")
    # generate_validation_data_global(df, ["x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8"], ["f1", "f2", "f3", "f4", "f5"], n_runs=500, n_missing_data=200, ref_delta=0.2)

    df = pd.read_csv("./data/river_pollution_5000.csv")
    # OBS! this is with _global_ missing data!!!
    # fname ="/home/kilo/workspace/shap-experiments/_results/run_river_n_1000_missing_16807_even_delta_10.xlsx" 
    # generate_validation_data_global(df, ["x_1", "x_2"], ["f_1", "f_2", "f_3", "f_4", "f_5"], n_runs=1000, n_missing_data=7, ref_delta=0.10, file_name=fname)

    deltas = [0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.14, 0.16, 0.18, 0.20]
    missings = [2, 3, 4, 5, 6, 7]
    n = 1000
    for m in missings:
        for d in deltas:
            fname =f"/home/kilo/workspace/shap-experiments/_results/run_river_5000_n_{n}_missing_{m**5}_even_delta_{int(d*100)}.xlsx" 
            generate_validation_data_global(df, ["x_1", "x_2"], ["f_1", "f_2", "f_3", "f_4", "f_5"], n_runs=n, n_missing_data=m, ref_delta=d, file_name=fname)
