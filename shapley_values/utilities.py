import numpy as np
import pandas as pd
import shap
from shapley_values.explanations import why_best, why_objective_i, why_worst, largest_conflict, how_to_improve_objective_i
from desdeo_problem.problem import DiscreteDataProblem
from desdeo_tools.scalarization import DiscreteScalarizer
from desdeo_tools.solver import DiscreteMinimizer
from desdeo_tools.scalarization import SimpleASF


def generate_missing_data(n: int, ideal: np.ndarray, nadir: np.ndarray) -> np.ndarray:
    """ Generate n missing data by randomly sampling a uniform distribution of points between
    the ideal and nadir.
    """
    return np.hstack(tuple(np.random.uniform(low=ideal[i], high=nadir[i], size=(n, 1)) for i in range(ideal.shape[0])))


def generate_black_box(problem: DiscreteDataProblem, asf: SimpleASF) -> np.ndarray:
    """Given a 2D array of reference points, a problem, and an achivevement scalarizing function,
    finds a set of solutions minimizing the achievement scalarizing function for each given
    reference point.

    TODO: add minimizer_args as kwarg
    """
    def black_box(ref_points: np.ndarray, problem: DiscreteDataProblem = problem, asf: SimpleASF = asf) -> np.ndarray:
        res = np.zeros(ref_points.shape)

        for (i, ref_point) in enumerate(ref_points):
            scalarizer = DiscreteScalarizer(asf, scalarizer_args={"reference_point": np.atleast_2d(ref_point)})
            solver = DiscreteMinimizer(scalarizer)
            index = solver.minimize(problem.objectives)["x"]

            res[i] = problem.objectives[index]

            return res

    return black_box


if __name__ == "__main__":
    """
    ideal = np.array([-4, -1, -3])
    nadir = np.array([4, 1, 3])
    n = 200

    res = generate_missing_data(n, ideal, nadir)
    print(res)
    print(res.shape)
    """

    # root me from project root
    df = pd.read_csv("./data/DTLZ2_5x_3f.csv")
    pareto_f = df.to_numpy()

    ideal = np.min(pareto_f[:, 0:3], axis=0)
    nadir = np.max(pareto_f[:, 0:3], axis=0)

    problem = DiscreteDataProblem(df, ["x1", "x2", "x3", "x4", "x5"], ["f1", "f2", "f3"], nadir, ideal)

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


