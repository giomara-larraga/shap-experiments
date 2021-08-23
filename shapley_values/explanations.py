from typing import Tuple

import numpy as np


def why_worst(svalues: np.ndarray, target: np.ndarray, actual: np.ndarray) -> Tuple[str, int, int]:
    # compute diff between target and actual values, and find the largest positive
    # discrepancy
    # only look at positive values in diff (i.e., objectives that were worse than desired)
    # here we care only about resulting objective values worse than the target and wish to
    # explain the probable reason the value is so bad.

    diff = actual - target
    if np.all(diff < 0):
        return "All objectives were improved compared to the desired value.", -1, -1
    mask = diff > 0
    diff[~mask] = -np.inf

    max_i = np.argmax(diff)

    # find the reason for the above from the SHAP values
    reason_i = np.argmax(svalues[max_i])

    return (
        (
            f"Objective {max_i+1} is farthest from the desired value. The value of objective {reason_i+1} "
            "has affected most significantly its deviance from the desired value."
        ),
        max_i,
        reason_i,
    )


def why_best(svalues: np.ndarray, target: np.ndarray, actual: np.ndarray) -> Tuple[str, int, int]:
    # compute diff between target and actual values, and find the smallest negative
    # discrepancy
    # only look at negative values in diff (i.e., objectives that were better than desired)
    # Here we just care finding the value which was improved the most from the target value.
    # We assume that when a value is better than the target it does not matter if it is far from the
    # desired value.

    diff = actual - target
    if np.all(diff > 0):
        return "All objectives were impaired compared to the desired value.", -1, -1
    mask = diff < 0
    diff[~mask] = np.inf

    print(diff)

    min_i = np.argmin(diff)

    # find the reason for the above from the SHAP values
    reason_i = np.argmin(svalues[min_i])

    return (
        (
            f"Objective {min_i+1} was improved the most compared to the desired value. The value of objective {reason_i+1} "
            "has affected most significantly its agreement with the desired value."
        ),
        min_i,
        reason_i,
    )


def why_objective_i(svalues: np.ndarray, objective_i: int) -> Tuple[str, int, int]:
    # given SHAP values and the index of an objective desired to be improved,
    # look at the SHAP values for hints on which objective values have had the
    # best and worst effects on objective_i

    best_effect_i = np.argmin(svalues[objective_i])
    worst_effect_i = np.argmax(svalues[objective_i])

    return (
        (
            f"Objective {objective_i+1} was improved most by the value given for objective {best_effect_i+1} and "
            f"impaired most by the value given to objective {worst_effect_i + 1}."
        ),
        best_effect_i,
        worst_effect_i,
    )


def largest_conflict(svalues: np.ndarray) -> Tuple[str, int, int]:
    # look at the off-diagonal elements in the SHAP values and compare them symmetrically. Find the two elements
    # with different signs and largest difference.
    diff = svalues + svalues.T
    np.fill_diagonal(diff, -np.inf)
    conflict_pair = np.unravel_index(np.argmax(diff), diff.shape)

    msg = f"The largest conflict seems to be between objectives {conflict_pair[0]+1} and {conflict_pair[1]+1}."

    return msg, conflict_pair[0], conflict_pair[1]


def how_to_improve_objective_i(svalues: np.ndarray, objective_i: int) -> Tuple[str, int, int]:
    # Look at the largest conflicts. If objective_i is among these, suggest to worsen the other.
    conflicting = largest_conflict(svalues)[1:]

    if objective_i in conflicting:
        to_worsen = conflicting[0] if conflicting[0] != objective_i else conflicting[1]
        msg = (
            f"Since objective {objective_i+1} and objective {to_worsen+1} are in great conflict, try worsening the value "
            f"of objective {to_worsen+1} for a better result for objective {objective_i+1}."
        )

        return msg, -1, to_worsen
    else:
        # objective_i not in greatest conflict, look for other causes
        msg, to_keep, to_worsen = why_objective_i(svalues, objective_i)
        # if objective_i is the cause of the worst effect, then suggest to improve it, otherwise, suggest
        # to worsen the cause of worst effect
        if to_worsen == objective_i:
            msg = (
                (
                    f"Objective {objective_i+1} was the cause of worst effect on itself. Since objective {to_keep+1} "
                    f"had the best effect, try to keep its value as is while improving the value of objective {objective_i+1}"
                ),
            )
            return msg, objective_i, -1
        msg = msg + (
            f"Therefore, try keeping the value of objective {to_keep+1} as is while worsening the value "
            f"of objective {to_worsen + 1} to improve the value of objective {objective_i+1}."
        )

        return msg, to_keep, to_worsen
