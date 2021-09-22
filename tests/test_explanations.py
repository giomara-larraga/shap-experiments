from shapley_values.explanations import largest_conflict, why_worst, why_best, why_objective_i, how_to_improve_objective_i
import numpy as np


def test_largest_conflict():
    data = np.array([
        [0, 2, 4],
        [-2, 0, 5],
        [1, 4, 0]
    ], dtype=float)
    _, p1, p2 = largest_conflict(data)

    assert p1 != p2
    assert p1 in [0, 1]
    assert p2 in [0, 1]

    data_all_pos = np.array([
        [0, 2, 4],
        [2, 0, 5],
        [1, 4, 0]
    ], dtype=float)

    _, p1_pos, p2_pos = largest_conflict(data_all_pos)

    assert (p1_pos, p2_pos) == (-1, -1)

    data_all_neg = np.array([
        [0, -2, -4],
        [-2, 0, -5],
        [-1, -4, 0]
    ], dtype=float)

    _, p1_neg, p2_neg = largest_conflict(data_all_neg)

    assert (p1_neg, p2_neg) == (-1, -1)

def test_why_worst():
    target = np.array([-5, 2, 0], dtype=float)

    actual_all_better = np.array([-6, -1, -2], dtype=float)
    shap_values = np.array([
        [-1, 2, 3],
        [-2, -1, 2],
        [-2, -2, -1]
    ], dtype=float)

    _, what, why = why_worst(shap_values, target, actual_all_better)

    assert what == why == -1

    actual_all_worse = np.array([-3, 3, 1], dtype=float)
    _, what_worse, why_worse = why_worst(shap_values, target, actual_all_worse)

    assert what_worse == 0
    assert why_worse == 2

    actual_mixed = np.array([-3, 1, -3], dtype=float)
    shap_values_mixed = np.array([
        [-1, 10, 3],
        [-2, -1, 2],
        [-2, -2, -1]
    ], dtype=float)

    _, what_mixed, why_mixed = why_worst(shap_values_mixed, target, actual_mixed)

    assert what_mixed == 0
    assert why_mixed == 1

def test_why_best():
    target = np.array([-5, 2, 0], dtype=float)

    actual_all_better = np.array([-6, -1, -2], dtype=float)
    shap_values = np.array([
        [-1, 2, 3],
        [-2, -1, 2],
        [-2, -2, -1]
    ], dtype=float)

    _, what, why = why_best(shap_values, target, actual_all_better)

    assert what == 1
    assert why == 0

    actual_all_worse = np.array([-3, 3, 1], dtype=float)
    _, what_worse, why_worse = why_best(shap_values, target, actual_all_worse)

    assert what_worse == why_worse == -1

    actual_mixed = np.array([-3, 1, -3], dtype=float)
    shap_values_mixed = np.array([
        [-1, 10, 3],
        [-2, -1, 2],
        [-2, -5, -1]
    ], dtype=float)

    _, what_mixed, why_mixed = why_best(shap_values_mixed, target, actual_mixed)

    assert what_mixed == 2
    assert why_mixed == 1


def test_why_objective_i():
    shap_values = np.array([
        [1, 2, 3],
        [-2, -1, 2],
        [-2, -3, -1]
    ], dtype=float)

    target_0 = 0
    _, best_0, worst_0 = why_objective_i(shap_values, target_0)

    assert best_0 == -1
    assert worst_0 == 2

    target_1 = 1
    _, best_1, worst_1 = why_objective_i(shap_values, target_1)

    assert best_1 == 0
    assert worst_1 == 2

    target_2 = 2
    _, best_2, worst_2 = why_objective_i(shap_values, target_2)

    assert best_2 == 1
    assert worst_2 == -1


def test_how_to_improve_objective_i():
    target = np.array([5, -2, 0, 3], dtype=float)

    shap_values = np.array([
        [-1, 2, -3, 1],
        [-2, -3, -1, 2],
        [2, 1, 5, 3],
        [0, 0, 0, -2]
    ], dtype=float)

    # Case nothing has improved
    actual_worse = np.array([6, -1, 1,5])

    ## Impairement not i
    objective_i = 0
    _, improve, impair = how_to_improve_objective_i(shap_values, objective_i, target, actual_worse)

    assert improve == 0
    assert impair == 1

    ## impairement is i
    objective_i = 2

    _, improve, impair = how_to_improve_objective_i(shap_values, objective_i, target, actual_worse)

    assert improve == 2
    assert impair == 3

    # Case everything has improved
    actual_better = np.array([2, -3, -1, 2])

    ## Objective i not least positive cause
    objective_i = 0
    _, improve, impair = how_to_improve_objective_i(shap_values, objective_i, target, actual_better)

    assert improve == 0
    assert impair == 1

    ## Objective i is least positive cause
    objective_i = 2
    _, improve, impair = how_to_improve_objective_i(shap_values, objective_i, target, actual_better)

    assert improve == 2
    assert impair == 3

    # Case: objective i is neither the cause of the best effect nor the worst effect
    shap_values_missing = np.array([
        [-1, -2, -3, -4],
        [-2, -1, -1, 2],
        [2, 1, 5, 3],
        [1, 5, 2, 1]
    ], dtype=float)

    actual_neutral = np.array([3, -1, -2, 0])

    ## No objective had positive effect
    objective_i = 3

    _, improve, impair = how_to_improve_objective_i(shap_values_missing, objective_i, target, actual_neutral)

    assert improve == objective_i
    assert impair == 1

    ## No objective had a negative effect
    objective_i = 0

    _, improve, impair = how_to_improve_objective_i(shap_values_missing, objective_i, target, actual_neutral)

    assert improve == objective_i
    assert impair == 1

    ## Some objective (that is not i) had a positive effect and some had a negative
    objective_i = 1

    _, improve, impair = how_to_improve_objective_i(shap_values_missing, objective_i, target, actual_neutral)

    assert improve == objective_i
    assert impair == 3

    # Case: objective i is neither the cause of the best effect nor the worst effect
    shap_values_itself = np.array([
        [-10, -2, 3, -4],
        [-2, -5, -1, 2],
        [2, 1, 5, 3],
        [-1, 5, -2, 8]
    ], dtype=float)

    ## objective i is the cause of the best effect
    objective_i = 0

    _, improve, impair = how_to_improve_objective_i(shap_values_itself, objective_i, target, actual_neutral)

    assert improve == objective_i
    assert impair == 2

    ## objective i is the cause of the worst effect
    objective_i = 3

    _, improve, impair = how_to_improve_objective_i(shap_values_itself, objective_i, target, actual_neutral)

    assert improve == objective_i
    assert impair == 1