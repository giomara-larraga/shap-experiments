from shapley_values.explanations import largest_conflict, why_worst, why_best
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