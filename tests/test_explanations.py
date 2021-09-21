from shapley_values.explanations import largest_conflict
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