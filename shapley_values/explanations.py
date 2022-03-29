from typing import Tuple, Union

import numpy as np


def why_worst(
    svalues: np.ndarray, target: np.ndarray, actual: np.ndarray
) -> Tuple[str, int, int]:
    """Compute the difference between target and actual values, and find the largest positive
    discrepancy. Only look at positive values in the difference (i.e., objectives that were worse
    than desired). We care only about resulting objective values worse than the target and wish
    to explain the probable reason for the bad value.

    Args:
        svalues (np.ndarray): A square matrix (2D array) with SHAP values.
        target (np.ndarray): An array with target objective values (i.e., the reference vector)
        actual (np.ndarray): An array with actual objective values (i.e., a projection of the references on the Pareto front)

    Returns:
        Tuple[str, int, int]: A tuple containing a textual explanation (str), an index (int)
        representing the objective with the highest (positive) deviation from the target, an
        index (int) representing the objective, which was the probable reason for this deviation.
        A value of -1 for both indices signifies that all objectives were improved.
    """
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


def why_best(
    svalues: np.ndarray, target: np.ndarray, actual: np.ndarray
) -> Tuple[str, int, int]:
    """Compute the difference between target and actual values, and find the largest negative
    discrepancy. Only look at negative values in the difference (i.e., objectives that were better
    than desired). We care only about resulting objective values better than the target and wish
    to explain the probable reason for the good value.

    Args:
        svalues (np.ndarray): A square matrix (2D array) with SHAP values.
        target (np.ndarray): An array with target objective values (i.e., the reference vector)
        actual (np.ndarray): An array with actual objective values (i.e., a projection of the references on the Pareto front)

    Returns:
        Tuple[str, int, int]: A tuple containing a textual explanation (str), an index (int)
        representing the objective with the lowest (negative) deviation from the target, an
        index (int) representing the objective, which was the probable reason for this deviation.
        A value of -1 for both indices signifies that all objectives were impaired.
    """
    diff = actual - target
    if np.all(diff > 0):
        return "All objectives were impaired compared to the desired value.", -1, -1
    mask = diff < 0
    diff[~mask] = np.inf

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
    """Given SHAP values and the index of an objective to be improved, look at the SHAP values for
    hints on which objectives values have had the best and worst effects on the desired objective.

    Args:
        svalues (np.ndarray): A square matrix (2D array) with SHAP values.
        objective_i (int): The index of the objective to be improved.

    Returns:
        Tuple[str, int, int]: A tuple containing a textual explanations (str), the index (int) of the objective with
        the best effect, and the index (int) of the objective with the worst effect. An idex value of -1 indicates
        that no objective had a good/bad effect.

    Note:
        It is assumed that each row has at least one element with a non-zero element.
    """
    # check that an objective exists that had a positive effect
    if np.any(svalues[objective_i] <= 0):
        best_effect_i = np.argmin(svalues[objective_i])
    else:
        best_effect_i = -1

    # check that an objective exists that had a negative effect
    if np.any(svalues[objective_i] > 0):
        worst_effect_i = np.argmax(svalues[objective_i])
    else:
        worst_effect_i = -1

    if best_effect_i == -1:
        msg = (
            f"None of the objectives had a positive effect on objective {objective_i+1}. Objective {objective_i+1} "
            f"was impaired most by the value given to objective {worst_effect_i+1}"
        )
    elif worst_effect_i == -1:
        msg = (
            f"All of the objectives had a positive effect on objective {objective_i+1}. Objective {objective_i+1} "
            f"was improved most by the value given to objective {best_effect_i+1}"
        )
    else:
        msg = (
            f"Objective {objective_i+1} was improved most by the value given for objective {best_effect_i+1} and "
            f"impaired most by the value given to objective {worst_effect_i+1}."
        )

    return (msg, best_effect_i, worst_effect_i)


def largest_conflict(svalues: np.ndarray) -> Tuple[str, int, int]:
    """Look at the off-diagonal elements in the SHAP values and compare them symmetrically. Find
    two elements with different signs and largest absolute difference. These two elements
    are taken to be mutually in a 'great conflict'.

    Args:
        svalues (np.ndarray): A square matrix (2D array) with SHAP values.

    Returns:
        Tuple[str, int, int]: A tuple containing a textual (str) explanation of the 'great conflict',
        an index (int) indicating the first element in the pair of conflict, an index (int) indicating the
        second pair. If both indices are -1, then no great conflict was found.
    """

    diff = np.abs(svalues - svalues.T)
    sign_mask = np.sign(svalues) == np.sign(svalues.T)

    if np.all(sign_mask):
        # nothing is conflicting, or everything is conflicting
        # all signs are the same, so this is safe
        sign = np.sign(svalues)[0, 0]

        if sign == -1:
            # everything improves everything!
            msg = f"No largest conflict found. Everything improves everything. You are too pessimistic!"
            return msg, -1, -1
        else:
            # 1, everything impairs everything!
            msg = f"No largest conflict found. Everything impairs everything. You are too optimistic!"
            return msg, -1, -1

    diff[sign_mask] = np.nan
    np.fill_diagonal(diff, -np.inf)
    conflict_pair = np.unravel_index(np.nanargmax(diff), diff.shape)

    msg = f"The largest conflict seems to be between objectives {conflict_pair[0]+1} and {conflict_pair[1]+1}."

    return msg, conflict_pair[0], conflict_pair[1]


def how_to_improve_objective_i_old(
    svalues: np.ndarray, objective_i: int
) -> Tuple[str, int, int]:
    """Determines a strategy on how a reference point, for which SHAP values have been computed for some black-box,
    should change so that an improvement in a desired objective may result when the black-box is invoked again with
    the changed reference point.

    Args:
        svalues (np.ndarray): A square matrix (2D array) with SHAP values.
        objective_i (int): The index of the objective that we wish to improve.

    Returns:
        Tuple[str, int, int]: A tuple containing: a textual explanation (str),
        an index to the reference point pointing to the objective value that
        should be improved for the desired effect, an index to the reference
        point pointing to the objective value which should be impaired for the
        desired effect.

    Note:
        Minimization is assumed for all objective. I.e., by 'improvement', a decrement in
        the related value is expected. Vice-versa for 'impairement'.
    """
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
        if to_worsen == objective_i:
            # if objective_i is the cause of the worst effect, then suggest to improve it, otherwise, suggest
            # to worsen the cause of worst effect
            msg = (
                f"Objective {objective_i+1} was the cause of worst effect on itself. Since objective {to_keep+1} "
                f"had the best effect, try to keep its value as is while improving the value of objective {objective_i+1}"
            )
            return msg, objective_i, -1
        else:
            msg = msg + (
                f"Therefore, try keeping the value of objective {to_keep+1} as is while worsening the value "
                f"of objective {to_worsen + 1} to improve the value of objective {objective_i+1}."
            )

            return msg, -1, to_worsen


def how_to_improve_objective_i(
    svalues: np.ndarray,
    objective_i: int,
    target: np.ndarray,
    actual: np.ndarray,
    objective_names: str = None,
) -> Tuple[str, int, int]:
    """Determines a strategy on how a reference point, for which SHAP values have been computed for some black-box,
    should change so that an improvement in a desired objective may result when the black-box is invoked again with
    the changed reference point.

    Args:
        svalues (np.ndarray): A square matrix (2D array) with SHAP values.
        target (np.ndarray): The objective values we wish to attain (i.e., reference point).
        actual (np.ndarray): The actual objective values we got (i.e., a projection from the
        reference point to the Pareto front).
        objective_i (int): The index of the objective that we wish to improve.

    Returns:
        Tuple[str, int, int, int]: A tuple containing: a textual explanation (str),
        an index to the reference point pointing to the objective value that
        should be improved for the desired effect, an index to the reference
        point pointing to the objective value which should be impaired for the
        desired effect, and a reference (int) used to recognize an outcome (for debug).

    Note:
        Minimization is assumed for all objective. I.e., by 'improvement', a decrement in
        the related value is expected. Vice-versa for 'impairement'.

    TODO:
        Write the textual explanations.
    """
    # Set default objective names if not provided.
    if objective_names is None:
        objective_names = [
            f"Objective {i}" for i in range(1, np.squeeze(target).shape[0] + 1)
        ]

    _, best_effect, worst_effect = why_objective_i(svalues, objective_i)
    diff = target - actual

    # Case: nothing has improved (everything in actual is greater than in target)
    if np.all(diff <= 0):
        # Impair the value of the objective with the worst effect and improve
        # the value for objective i. If i has had the worst effect, find the next
        # probable cause.
        if worst_effect != objective_i:
            # impair worst_effect, improve i
            msg = (
                "Explanation: Each objective value in the solution is worse when compared to the reference point. "
                "The reference point given was too demanding. "
                f"The component {objective_names[worst_effect]} in the reference point had the most impairing effect "
                f"on objective {objective_names[objective_i]} in the solution."
                f"\nSuggestion: Try improving the component {objective_names[objective_i]} and impairing the component "
                f"{objective_names[worst_effect]}."
            )
            return msg, objective_i, worst_effect, 0
        else:
            # impair second worst effect, improve i
            # find second effect
            row = svalues[objective_i]
            row[objective_i] = -np.inf
            second_worst = np.argmax(row)

            msg = (
                f"Explanation: Each objective value in the solution is worse when compared to the reference point. "
                f"The reference point was too demanding. The component {objective_names[objective_i]} in the reference point "
                f"had the most impairing effect on {objective_names[objective_i]} in the solution. The component "
                f"{objective_names[second_worst]} had the second most impairing effect on the objective {objective_names[objective_i]}."
                f"\nSuggestion: Try improving the component {objective_names[objective_i]} and impairing the component {objective_names[second_worst]}."
            )
            return msg, objective_i, second_worst, 1

    # Case: everything has improved (everything in actual is less than in target)
    if np.all(diff > 0):
        # Improve the value for objective i and impair the value for the objective
        # which had the least derisable effect on objective i (this effect can be also
        # negative (i.e., good))

        # Find the objective with the least desirable effect
        row = svalues[objective_i]
        first_cause = np.argmax(row)

        # Check if first_cause is objective i
        if first_cause == objective_i:
            # improve i, impair second_cause
            # find the second least desirable effect
            row[objective_i] = -np.inf
            second_cause = np.argmax(row)
            msg = (
                f"Explanation: Each objective value in the solution had a better value when compared to the reference point."
                f"The reference point was pessimistic. The component {objective_names[objective_i]} in the reference point "
                f"had the least improving effect on objective {objective_names[objective_i]} in the solution. The component "
                f"{objective_names[second_cause]} had the second least improving effect on the objective {objective_names[objective_i]}."
                f"\nSuggestion: Try improving the component {objective_names[objective_i]} and impairing the component {objective_names[second_cause]}."
            )

            return msg, objective_i, second_cause, 2
        else:
            # improve i, impair first_cause
            msg = (
                f"Explanation: Each objective value in the solution had a bettern value when compared to the reference point. "
                f"The reference point was pessimistic. The component {objective_names[first_cause]} in the refence point "
                f"had the least improving effect on the objective {objective_names[objective_i]} in the solution."
                f"\nSuggestion: Try improving the component {objective_names[objective_i]} and impairing the component {objective_names[first_cause]}."
            )
            return msg, objective_i, first_cause, 3

    # Note: best_effect and worst_effect being -1 at the same time is not possible
    # due to how why_objective_i is defined.

    # Case: objective i is neither the cause of the best effect nor the worst effect
    if objective_i != best_effect and objective_i != worst_effect:
        # Improve the value for i, keep the value
        # for j, and impair the value for k.
        if best_effect == -1:
            # no objective had a positive effect on i
            # improve i, impair objective with greatest negative effect
            msg = (
                f"Explanation: None of the component in the reference point had an improving effect on the objective {objective_names[objective_i]} "
                f"in the solution. The component {objective_names[worst_effect]} in the reference point had the most impairing effect on "
                f"objective {objective_names[objective_i]} in the solution."
                f"\nSuggestion: Try improving the component {objective_names[objective_i]} and impairing the component {objective_names[worst_effect]}."
            )
            return msg, objective_i, worst_effect, 4

        elif worst_effect == -1:
            # no objective had a negative effect on i
            # improve i, impair objective with least positive effect (which is not i)
            row = svalues[objective_i]
            row[objective_i] = -np.inf
            least_positive = np.argmax(row)

            msg = (
                f"Explanation: None of the objectives in the reference point had an impairing effect on objective {objective_names[objective_i]} "
                f"in the solution. Objective {objective_names[least_positive]} in the reference point had the least improving effect on objective "
                f"{objective_names[objective_i]} in the solution."
                f"\nSuggestion: Try improving the component {objective_names[objective_i]} and impairing the component {objective_names[least_positive]}."
            )

            return msg, objective_i, least_positive, 5

        else:
            # some objective had a positive and some objective had a negative effect on i
            # improve i, impair objective with worst effect
            msg = (
                f"Explanation: The objective {objective_names[objective_i]} was most improved in the solution by the component "
                f"{objective_names[best_effect]} and most impaired by the component {objective_names[worst_effect]} in the reference point."
                f"\nSuggestion: Try improving the component {objective_names[objective_i]} and impairing the component {objective_names[worst_effect]}."
            )

            return msg, objective_i, worst_effect, 6

    # Case: objective i is the cause of the worst effect
    if objective_i == worst_effect:
        # improve i, impair objective with second most negative effect
        # TODO: we might want to rethink this one
        row = svalues[objective_i]
        row[objective_i] = -np.inf
        second_worst = np.argmax(row)

        msg = (
            f"Explanation: The objective {objective_names[objective_i]} was most impaired in the solution by its component in "
            f"the reference point. The component {objective_names[second_worst]} had the second most impairing effect on the "
            f"objective {objective_names[objective_i]}."
            f"\nSuggestion: Try improving the component {objective_names[objective_i]} and impairing the component {objective_names[second_worst]}."
        )
        return msg, objective_i, second_worst, 7

    # Case: no worst effect exists (i.e., no positive values in svalues relevant to objective_i)
    if worst_effect == -1:
        # improve i, impair the objective with the least positive effect on i
        # i cannot be least_positive
        row = svalues[objective_i]
        row[objective_i] = -np.inf
        least_positive = np.argmax(row)

        msg = (
            f"Explanation: None of the objectives in the reference point had an impairing effect on objective {objective_names[objective_i]} "
            f"in the solution. Objective {objective_names[least_positive]} in the reference point had the least improving effect on objective "
            f"{objective_names[objective_i]} in the solution."
            f"\nSuggestion: Try improving the component {objective_names[objective_i]} and impairing the component {objective_names[least_positive]}."
        )
        return msg, objective_i, least_positive, 8

    # Case: objective i is the cause of the best effect
    if objective_i == best_effect:
        # improve i, impair objective with worst effect
        msg = (
            f"Explanation: The objective {objective_names[objective_i]} was most improved in the solution by its component in "
            f"the reference point. The component {objective_names[worst_effect]} had the most impairing effect of objective {objective_names[objective_i]}."
            f"\nSuggestion: Try improving the component {objective_names[objective_i]} and impairing the component {objective_names[worst_effect]}."
        )
        return msg, objective_i, worst_effect, 9

    """
    # Case: no best effect exists (i.e., no negative values in svalues relevant to objective_i)
    if best_effect == -1:
        # improve i, impair the objective with the least positive effect i
        # i cannot be least_positive
        row = svalues[objective_i]
        row[objective_i] = -np.inf
        least_positive = np.argmax(row)

        msg = (
            f"None of the objectives had an improving effect. To improve objective {objective_names[objective_i]}, "
            f"try to improve its value while impairing the value of objective {objective_names[least_positive]}, which had "
            f"the least positive effect on objective {objective_names[objective_i]}."
        )
        return msg, objective_i, least_positive, 10

    return "Impossible outcome", -1, -1
    """


def split_suggestion_explanation(s: str) -> Tuple[str, str]:
    """Splits a string output returned by 'how_to_improve_objective_i' into a suggestion and explanation parts. The
    parts should be separated by a newline character. If no newline character is found, then the suggestion and
    explanation parts returned will be identical.

    Args:
        s (str): the string containing an explanation and suggestion part to be split.

    Returns:
        Tuple[str, str]: the tuple containing the suggestion and explanations parts, respectively.

    Note:
        If no newline if found in the input s, then the suggestion and explanation part in the output
        Tuple will be the same as the input s.
    """
    # split at the newline
    ind = s.find("\n")
    if ind != -1:
        explanation = s[:ind]
        suggestion = s[ind + 1 :]
    else:
        suggestion = s
        explanation = s

    return (suggestion, explanation)


if __name__ == "__main__":
    pass
