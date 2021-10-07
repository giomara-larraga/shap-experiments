from desdeo_problem.problem.Problem import MOProblem
import numpy as np
from numpy.core.fromnumeric import squeeze
import pandas as pd
import shap
from typing import List, Optional
import matplotlib.pyplot as plt

from shapley_values.explanations import (
    why_best,
    why_objective_i,
    why_worst,
    largest_conflict,
    how_to_improve_objective_i,
)
from shapley_values.utilities import (
    generate_missing_data,
    generate_missing_data_even,
    generate_black_box,
    Normalizer,
)
from shapley_values.problems import car_crash_problem, river_pollution_problem
from desdeo_problem.problem import DiscreteDataProblem
from desdeo_tools.scalarization import Scalarizer
from desdeo_tools.solver import ScalarMinimizer
from desdeo_tools.scalarization import SimpleASF, PointMethodASF


def solve_with_ref_point(problem: MOProblem, ref_point: np.ndarray, asf=PointMethodASF):
    scalarizer = Scalarizer(
        lambda x: problem.evaluate(x).objectives,
        asf,
        scalarizer_args={"reference_point": np.atleast_2d(ref_point)},
    )
    minimizer = ScalarMinimizer(
        scalarizer, problem.get_variable_bounds(), method="scipy_de"
    )

    res = minimizer.minimize(
        (problem.get_variable_lower_bounds() + problem.get_variable_upper_bounds()) / 2
    )

    print(res["x"])
    objectives = problem.evaluate(res["x"]).objectives

    return objectives


def generate_validation_data_global(
    df: pd.DataFrame,
    variable_names: List[str],
    objective_names: List[str],
    n_missing_data: int = 200,
    n_runs: int = 10,
    ref_delta: float = 0.1,
    file_name: str = "",
    original_problem: Optional[MOProblem] = None,
    improve_target: bool = True,
    pareto_as_missing: bool = False,
    worsen_random: bool = False,
):
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
            list(f"Original ref point f_{i}" for i in range(1, n_objectives + 1))
            + list(f"Original solution f_{i}" for i in range(1, n_objectives + 1))
            + ["Index of solution DM wishes to improve"]
            + ["Suggestion with explanation"]
            + ["Index of objective to be improved in the reference point"]
            + ["Index of objective to be worsened in the reference point"]
            + list(
                f"New ref point based on suggestion f_{i}"
                for i in range(1, n_objectives + 1)
            )
            + list(
                f"New solution based on new ref point f_{i}"
                for i in range(1, n_objectives + 1)
            )
            + ["Was the desired effect achieved?"]
            + ["Case identifier"]
            + ["Change in reference point values (multiplier w.r.t. nadir - ideal)"]
        )
    )

    fail_count = 0

    low = ideal
    high = nadir
    # nadir always > ideal
    delta = ref_delta * (nadir - ideal)

    while run_i < n_runs:
        print(f"Run {run_i+1} out of {n_runs}...")

        # generate a random reference point between the ideal and nadir
        ref_point = np.array(
            [
                np.random.uniform(low=ideal[i], high=nadir[i])
                for i in range(ideal.shape[0])
            ]
        )

        if not pareto_as_missing:
            # We generate data also outside the area dicated by the ideal and nadir points to have more accurate explanations
            # for reference point that reside on the edge of the area.
            # missing_data = generate_missing_data(n_missing_data, low-delta, high+delta)
            missing_data = generate_missing_data_even(
                n_missing_data, low - delta, high + delta
            )
        else:
            # sample the PO for missing data
            missing_data = shap.sample(pareto_f[:, 0:n_objectives], nsamples=200)

        explainer = shap.KernelExplainer(bb, missing_data)

        # the computed solution by the black box
        if original_problem is None:
            solution = bb(np.atleast_2d(ref_point))
        else:
            solution = solve_with_ref_point(original_problem, ref_point, asf)

        # compute the Shapley values for the given reference point
        shap_values = np.array(explainer.shap_values(ref_point))

        # check the Shapley values and figure out how to improve a random objective (zero indexed)
        to_be_improved = np.random.randint(0, n_objectives)

        explanation, improve_i, worsen_i, case_i = how_to_improve_objective_i(
            shap_values, to_be_improved, ref_point, solution
        )

        if worsen_random and worsen_i > -1:
            # choose something else to worsen at random, except the original worsen_i
            worsen_candidates = list(range(0, worsen_i)) + list(
                range(worsen_i + 1, n_objectives)
            )
            print(
                f"Worsening originally {worsen_i}; picking from {list(worsen_candidates)}"
            )
            worsen_i = np.random.choice(worsen_candidates)
            print(f"New candidate is {worsen_i}")

        # save the original ref_point before modifying it
        original_ref_point = np.copy(ref_point)

        # check if something is to be improved and improve it (notice that we assume minimization)
        if improve_target and improve_i > -1:
            # change the ref_point accordingly
            ref_point[improve_i] -= delta[improve_i]

        # check if something is to be worsened (notice that we assume minimization)
        if worsen_i > -1:
            # change the ref_point accordingly
            # ref_point[worsen_i] += ref_delta * ref_point[worsen_i]
            ref_point[worsen_i] += delta[worsen_i]

        # compute the new solution based on the modified ref_point
        if original_problem is None:
            new_solution = bb(np.atleast_2d(ref_point))
        else:
            new_solution = solve_with_ref_point(original_problem, ref_point, asf)

        # check if the desired effect was achieved
        if new_solution.squeeze()[to_be_improved] < solution.squeeze()[to_be_improved]:
            # print(f"Effect was achieved! To be improved {to_be_improved}\nOld solution {solution.squeeze()[to_be_improved]}\nNew solution {new_solution.squeeze()[to_be_improved]}")
            effect_was_achieved = 1
        elif np.equal(
            new_solution.squeeze()[to_be_improved], solution.squeeze()[to_be_improved]
        ):
            # print(f"Effect was _NOT_ achieved due to no change! To be improved {to_be_improved}\nOld solution {solution.squeeze()[to_be_improved]}\nNew solution {new_solution.squeeze()[to_be_improved]}")
            effect_was_achieved = 0
        else:
            # print(f"Effect was _NOT_ achieved! To be improved {to_be_improved}\nOld solution {solution.squeeze()[to_be_improved]}\nNew solution {new_solution.squeeze()[to_be_improved]}")
            fail_count += 1
            effect_was_achieved = -1

        # compile row to be added to the data and add it
        datum = (
            original_ref_point.squeeze().tolist()
            + solution.squeeze().tolist()
            + [to_be_improved + 1]
            + [explanation]
            + [improve_i + 1 if improve_i != -1 else -1]
            + [worsen_i + 1 if worsen_i != -1 else -1]
            + ref_point.squeeze().tolist()
            + new_solution.squeeze().tolist()
            + [effect_was_achieved]
            + [case_i]
            + [delta]
        )

        data.loc[run_i] = datum

        run_i += 1

    print(f"Out of {n_runs} runs {n_runs - fail_count} succeeded.")

    data.to_excel(file_name)


def plot_3d(
    df: pd.DataFrame, n_objectives: int, dims: Optional[List[int]] = None
) -> None:
    pareto_f = df.to_numpy()[:, 0:n_objectives]

    if dims is None:
        dims = list(range(n_objectives))

    print(dims)
    print(pareto_f.shape)

    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")

    ax.scatter(pareto_f[:, dims[0]], pareto_f[:, dims[1]], pareto_f[:, dims[2]])
    plt.show()


if __name__ == "__main__":
    problem_name = "car_crash"
    var_names = ["x_1", "x_2"]
    obj_names = ["f_1", "f_2", "f_3"]
    n_solutions = 10112
    n_runs = 1000

    use_original_problem = True
    # OBS! Check me!
    mop = car_crash_problem()

    pareto_as_missing = True
    improve_target = False
    worsen_random = True

    df = pd.read_csv(f"./data/{problem_name}_{n_solutions}.csv")

    plot_3d(df, 3)

    # deltas = [0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.14, 0.16, 0.18, 0.20]
    deltas = [0.05, 0.10, 0.15]
    # missings = [2, 3, 4, 5, 6, 7]
    missings = [3]
    for m in missings:
        for d in deltas:
            fname = (
                f"/home/kilo/workspace/shap-experiments/_results/run_{problem_name}_{n_solutions}_{n_runs}_missing_200_delta_"
                f"{int(d*100)}{'_original_' if use_original_problem else '_'}{'pfmissing_' if pareto_as_missing else ''}"
                f"{'nochange_' if not improve_target else ''}{'random' if worsen_random else ''}.xlsx"
            )
            generate_validation_data_global(
                df,
                var_names,
                obj_names,
                n_runs=n_runs,
                n_missing_data=m,
                ref_delta=d,
                file_name=fname,
                original_problem=mop,
                pareto_as_missing=pareto_as_missing,
                improve_target=improve_target,
                worsen_random=worsen_random,
            )
