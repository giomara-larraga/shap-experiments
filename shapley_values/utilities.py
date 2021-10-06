import numpy as np
import pandas as pd
import shap
from typing import Optional
from shapley_values.explanations import why_best, why_objective_i, why_worst, largest_conflict, how_to_improve_objective_i
from sklearn.preprocessing import MinMaxScaler
from desdeo_problem.problem import DiscreteDataProblem
from desdeo_tools.scalarization import DiscreteScalarizer
from desdeo_tools.solver import DiscreteMinimizer
from desdeo_tools.scalarization import SimpleASF


class Normalizer:
    """Used to transform objective vectors to normalized form and back to original form.
    """
    def __init__(self, low_limits: np.ndarray, high_limits: np.ndarray, scaler_class = MinMaxScaler):
        self.scaler = scaler_class()
        self.scaler.fit(np.stack((low_limits, high_limits)))

    
    def scale(self, values: np.ndarray) -> np.ndarray:
        return self.scaler.transform(np.atleast_2d(values))

    def inverse_scale(self, values: np.ndarray) -> np.ndarray:
        return self.scaler.inverse_transform(np.atleast_2d(values))


def generate_missing_data(n: int, low: np.ndarray, high: np.ndarray) -> np.ndarray:
    """Generate n missing data by randomly sampling a uniform distribution of points between
    a lower and an higher limit.

    Args:
        n (int): The number of samples to be generated.
        low (np.ndarray): An array with the lower limit for each dimension.
        high (np.ndarray): An array with the higher limit for each dimension.

    Returns:
        np.ndarray: A 2D array of vectors representing the missing data generated.
    """    
    return np.hstack(tuple(np.random.uniform(low=low[i], high=high[i], size=(n, 1)) for i in range(low.shape[0])))


def generate_missing_data_even(n: int, low: np.ndarray, high: np.ndarray) -> np.ndarray:
    """Generate n missing data by evenly sampling the points between
    a lower and an higher limit for each dimension.

    Args:
        n (int): The number of samples to be generated (this is an
            approximation, n^k points will be generated where k is the dimension
            of the low and high arrays.
        low (np.ndarray): An array with the lower limit for each dimension.
        high (np.ndarray): An array with the higher limit for each dimension.

    Returns:
        np.ndarray: A 2D array of vectors representing the missing data generated.
    """    
    # return np.hstack(tuple(np.random.uniform(low=low[i], high=high[i], size=(n, 1)) for i in range(low.shape[0])))
    steps = (high - low) / (n-1)
    return np.mgrid[[slice(l, h+step/2, step) for l, h, step in zip(low, high, steps)]].reshape(low.shape[0], -1).T


def generate_black_box(problem: DiscreteDataProblem, asf: SimpleASF, normalizer: Optional[Normalizer] = None) -> np.ndarray:
    """Given a 2D array of reference points, a problem, and an achivevement scalarizing function,
    finds a set of solutions minimizing the achievement scalarizing function for each given
    reference point.

    TODO: add minimizer_args as kwarg
    """
    def black_box(ref_points: np.ndarray, problem: DiscreteDataProblem = problem, asf: SimpleASF = asf, normalizer: Optional[Normalizer] = normalizer) -> np.ndarray:
        res = np.zeros(ref_points.shape)

        for (i, ref_point) in enumerate(ref_points):
            scalarizer = DiscreteScalarizer(asf, scalarizer_args={"reference_point": np.atleast_2d(ref_point)})
            solver = DiscreteMinimizer(scalarizer)
            index = solver.minimize(problem.objectives)["x"]

            res[i] = problem.objectives[index]

        if normalizer is None:
            # original scale
            return res
        else:
            return normalizer.scale(res)

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


