import numpy as np
import pandas as pd
import shap
from typing import List

from shapley_values.explanations import why_best, why_objective_i, why_worst, largest_conflict, how_to_improve_objective_i
from shapley_values.utilities import generate_missing_data, generate_black_box
from desdeo_problem.problem import DiscreteDataProblem
from desdeo_tools.scalarization import DiscreteScalarizer
from desdeo_tools.solver import DiscreteMinimizer
from desdeo_tools.scalarization import SimpleASF


def generate_validation_data_global(df: pd.DataFrame, variable_names: List[str], objective_names: List[str]):
    pareto_f = df.to_numpy()

    n_objectives = len(objective_names)
    ideal = np.min(pareto_f[:, 0:n_objectives], axis=0)
    nadir = np.max(pareto_f[:, 0:n_objectives], axis=0)

    problem = DiscreteDataProblem(df, variable_names, objective_names, nadir, ideal)

    asf = SimpleASF(np.array([1,1,1]))

    missing_data = generate_missing_data(200, ideal, nadir)

    bb = generate_black_box(problem, asf)

    explainer = shap.KernelExplainer(bb, missing_data)

    ref_point = np.array([0.65, 0.44, 0.66])
    result = bb(np.atleast_2d(ref_point))
    shap_values = np.array(explainer.shap_values(ref_point))

    print(f"Original ref point: {ref_point} with result: {result}")
    print(how_to_improve_objective_i(shap_values, 1)[0])

    ref_point = np.array([0.80, 0.44, 0.66])
    result = bb(np.atleast_2d(ref_point))
    shap_values = np.array(explainer.shap_values(ref_point))

    print(f"New reference point {ref_point} with result: {result}")
    


if __name__ == "__main__":
    df = pd.read_csv("./data/DTLZ2_5x_3f.csv")
    generate_validation_data_global(df, ["x1", "x2", "x3", "x4", "x5"], ["f1", "f2", "f3"])