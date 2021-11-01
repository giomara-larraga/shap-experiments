from numpy.lib.function_base import diff
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import List, SupportsComplex
from shapley_values.utilities import Normalizer

DATA_DIR = "/home/kilo/workspace/shap-experiments/_results"
N = 1000
DELTAS = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
MISSINGS = [32, 243, 1024, 3125, 7776, 16807]


def get_original_rps(
    dframe: pd.DataFrame, objective_prefix: str, n_objectives: int
) -> np.ndarray:
    """Fetch the original reference points from a data frame.

    Args:
        dframe (pd.DataFrame): The data frame.
        objective_prefix (str): The prefix of the objectives names. E.g., 'f_' for "f_1, f_2, f_3, etc..."
        n_objectives (int): The number of objectives.

    Returns:
        np.ndarray: A numpy array containing the reference points on each row.
    """
    table_prefix = "Original ref point "
    stack_me = list(
        dframe.loc[:, f"{table_prefix}{objective_prefix}{i}"].to_numpy()
        for i in range(1, n_objectives + 1)
    )
    return np.vstack(stack_me).T


def get_original_solutions(
    dframe: pd.DataFrame, objective_prefix: str, n_objectives: int
) -> np.ndarray:
    """Fetch the original solutions from a data frame.

    Args:
        dframe (pd.DataFrame): The data frame.
        objective_prefix (str): The prefix of the objective names. E.g., 'f_' for "f_1, f_2, f_3, etc..."
        n_objectives (int): The number of objectives

    Returns:
        np.ndarray: A numpy array containing the solutions on each row.
    """
    table_prefix = "Original solution "
    stack_me = list(
        dframe.loc[:, f"{table_prefix}{objective_prefix}{i}"].to_numpy()
        for i in range(1, n_objectives + 1)
    )
    return np.vstack(stack_me).T


def get_new_solutions(
    dframe: pd.DataFrame, objective_prefix: str, n_objectives: int
) -> np.ndarray:
    """Feth the new solutions from a data frame.

    Args:
        dframe (pd.DataFrame): The data frame.
        objective_prefix (str): The prefix of the objective names. E.g., 'f_' for "f_1, f_2, f_3, etc..."
        n_objectives (int): The number of objectives.

    Returns:
        np.ndarray: A numpy array containing the new solutions on each row.
    """
    table_prefix = "New solution based on new ref point "
    stack_me = list(
        dframe.loc[:, f"{table_prefix}{objective_prefix}{i}"].to_numpy()
        for i in range(1, n_objectives + 1)
    )
    return np.vstack(stack_me).T


def get_new_rps(
    dframe: pd.DataFrame, objective_prefix: str, n_objectives: int
) -> np.ndarray:
    """Fetch the new solutions from a data frame.

    Args:
        dframe (pd.DataFrame): The data frame.
        objective_prefix (str): The prefix of the objective names. E.g., 'f_' for "f_1, f_2, f_3, etc..."
        n_objectives (int): The number of objectives

    Returns:
        np.ndarray: A numpy array containing the new solutions on each row.
    """
    table_prefix = "New ref point based on suggestion "
    stack_me = list(
        dframe.loc[:, f"{table_prefix}{objective_prefix}{i}"].to_numpy()
        for i in range(1, n_objectives + 1)
    )
    return np.vstack(stack_me).T


def get_effect_result(dframe: pd.DataFrame) -> np.ndarray:
    """Return the effect result for each run (1: success, 0: no change, -1: failure) from a data frame.

    Args:
        dframe (pd.DataFrame): The data frame.

    Returns:
        np.ndarray: A numpy array with the effect results.
    """
    return dframe["Was the desired effect achieved?"].to_numpy()


def get_target_indices(dframe: pd.DataFrame) -> np.ndarray:
    target_key = "Index of solution DM wishes to improve"
    # start from 1
    target_indices = dframe.loc[:, target_key].to_numpy() - 1

    return target_indices


def get_changes_in_solutions(
    dframe: pd.DataFrame, objective_prefix: str, n_objectives: int
):

    originals = get_original_solutions(dframe, objective_prefix, n_objectives)
    news = get_new_solutions(dframe, objective_prefix, n_objectives)

    # target_originals = originals[np.arange(target_indices.shape[0]), target_indices]
    # target_news = news[np.arange(target_indices.shape[0]), target_indices]

    abs_diffs = np.abs(news - originals)
    diffs = np.where(news < originals, -abs_diffs, abs_diffs)

    return diffs


def print_effects(dframe: pd.DataFrame) -> None:
    effects = get_effect_result(dframe)

    success = sum([1 if t == 1 else 0 for t in effects])
    no_change = sum([1 if t == 0 else 0 for t in effects])
    fail = sum([1 if t == -1 else 0 for t in effects])

    s = (
        f"Out of {df.shape[0]}: {success} were successful, {no_change} had no change, {fail} were a failure.\n"
        f"Success rate: {100*(success/df.shape[0]):.{1}f}%; No change rate: {100*(no_change/df.shape[0]):.{1}f}%; Failure rate: {100*(fail/df.shape[0]):.{1}f}%"
    )

    return s


def compute_relative_improvements_of_targets(
    dframe: pd.DataFrame, objective_prefix: str, n_objectives: int
):
    target_i = get_target_indices(dframe)
    originals = get_original_solutions(dframe, objective_prefix, n_objectives)
    news = get_new_solutions(dframe, objective_prefix, n_objectives)

    relative_changes_per_target = {str(i): [] for i in range(n_objectives)}
    for (i, t) in enumerate(target_i):
        # iterate the targets
        relative_change = (news[i, t] - originals[i, t]) / np.abs(originals[i, t]) * 100
        relative_changes_per_target[str(t)].append(relative_change)

    relative_means = np.array(
        [
            np.mean(relative_changes_per_target[key])
            for key in relative_changes_per_target
        ]
    )

    relative_stds = np.array(
        [
            np.std(relative_changes_per_target[key])
            for key in relative_changes_per_target
        ]
    )

    relative_maxes = np.array(
        [
            np.max(relative_changes_per_target[key])
            for key in relative_changes_per_target
        ]
    )

    relative_mins = np.array(
        [
            np.min(relative_changes_per_target[key])
            for key in relative_changes_per_target
        ]
    )

    return {
        "mean": relative_means,
        "std": relative_stds,
        "max": relative_maxes,
        "min": relative_mins,
    }


def compute_relative_improvements_of_targets_median_and_mad(
    dframe: pd.DataFrame, objective_prefix: str, n_objectives: int
):
    k = 1.4826
    target_i = get_target_indices(dframe)
    originals = get_original_solutions(dframe, objective_prefix, n_objectives)
    news = get_new_solutions(dframe, objective_prefix, n_objectives)

    relative_changes_per_target = {str(i): [] for i in range(n_objectives)}
    for (i, t) in enumerate(target_i):
        # iterate the targets
        relative_change = (news[i, t] - originals[i, t]) / np.abs(originals[i, t]) * 100
        relative_changes_per_target[str(t)].append(relative_change)

    relative_medians = np.array(
        [
            np.median(relative_changes_per_target[key])
            for key in relative_changes_per_target
        ]
    )

    relative_mads = np.array(
        [
            np.median(
                np.abs(
                    relative_changes_per_target[key]
                    - np.median(relative_changes_per_target[key])
                )
            )
            for key in relative_changes_per_target
        ]
    )

    relative_maxes = np.array(
        [
            np.max(relative_changes_per_target[key])
            for key in relative_changes_per_target
        ]
    )

    relative_mins = np.array(
        [
            np.min(relative_changes_per_target[key])
            for key in relative_changes_per_target
        ]
    )

    relative_std_mads = np.array(k * relative_mads)

    relative_changes = np.array(
        [relative_changes_per_target[key] for key in relative_changes_per_target]
    )
    no_outliers = np.where(
        np.logical_and(
            np.less_equal(relative_medians - 2 * relative_std_mads, relative_changes.T),
            np.greater_equal(
                relative_medians + 2 * relative_std_mads, relative_changes.T
            ),
        ),
        relative_changes.T,
        np.nan,
    )

    no_outliers_mask = np.isnan(no_outliers).any(axis=1)

    mean_95s = np.mean(no_outliers[~no_outliers_mask], axis=0)

    n_outliers = np.sum(np.count_nonzero(~no_outliers_mask))

    return {
        "median": relative_medians,
        "mad": relative_mads,
        "max": relative_maxes,
        "min": relative_mins,
        "std_mad": relative_std_mads,
        "mean95": mean_95s,
        "outliers": n_outliers,
    }


def gaussian(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.0) / (2 * np.power(sig, 2.0)))


def compute_target_success_rates(
    df: pd.DataFrame,
    n_objectives: int,
):
    effects = get_effect_result(df)

    # effects_per_target = {"0": [], "1": [], "2": [], "3": [], "4": []}
    effects_per_target = {str(i): [] for i in range(n_objectives)}
    t_indices = get_target_indices(df)

    for (i, t) in enumerate(t_indices):
        effects_per_target[str(t)].append(effects[i])

    stats_strs = []
    success_rates = {str(i): [] for i in range(n_objectives)}

    for key in effects_per_target:
        suc = sum([1 if t == 1 else 0 for t in np.array(effects_per_target[key])])
        neu = sum([1 if t == 0 else 0 for t in np.array(effects_per_target[key])])
        fai = sum([1 if t == -1 else 0 for t in np.array(effects_per_target[key])])
        n = len(effects_per_target[key])

        s = f"For objective {int(key)+1}: Success: {suc}/{n}; Neutral: {neu}/{n}; Fail: {fai}/{n}"

        stats_strs.append(s)

        success_rates[key] = [
            suc / n * 100,
            neu / n * 100,
            fai / n * 100,
        ]

    return np.array([success_rates[key] for key in success_rates])


def plot_and_save_basic_target_stats(
    df: pd.DataFrame,
    title: str,
    n_objectives: int,
    save_dir: str = "/home/kilo/workspace/shap-experiments/_numerical_results/",
):
    float_precision = 4
    float_width = 5
    effects = get_effect_result(df)

    # effects_per_target = {"0": [], "1": [], "2": [], "3": [], "4": []}
    effects_per_target = {str(i): [] for i in range(n_objectives)}
    t_indices = get_target_indices(df)

    for (i, t) in enumerate(t_indices):
        effects_per_target[str(t)].append(effects[i])

    stats_strs = []

    for key in effects_per_target:
        suc = sum([1 if t == 1 else 0 for t in np.array(effects_per_target[key])])
        neu = sum([1 if t == 0 else 0 for t in np.array(effects_per_target[key])])
        fai = sum([1 if t == -1 else 0 for t in np.array(effects_per_target[key])])
        n = len(effects_per_target[key])

        s = f"For objective {int(key)+1}: Success: {suc}/{n}; Neutral: {neu}/{n}; Fail: {fai}/{n}"

        stats_strs.append(s)

    stats_strs.append(print_effects(df))

    diffs_per_target = {str(i): [] for i in range(n_objectives)}
    # diffs_per_target = {"0": [], "1": [], "2": [], "3": [], "4": []}

    diffs = get_changes_in_solutions(df, "f_", n_objectives)
    t_indices = get_target_indices(df)

    for (i, t) in enumerate(t_indices):
        diffs_per_target[str(t)].append(diffs[i])

    mus_and_sigmas = {}

    for key in diffs_per_target:
        mu = np.mean(np.array(diffs_per_target[key])[:, int(key)])
        sig = np.std(np.array(diffs_per_target[key])[:, int(key)])
        mus_and_sigmas[f"{key}"] = (mu, sig)
        # print(f"Objective {key}: Mean diff: {mu}; Std diff: {sig}")
        x = np.arange(-5, 2.5, 0.01)

        plt.plot(
            x,
            gaussian(x, mu, sig),
            label=f"Objective {int(key)+1}: mean={mu:{float_width}.{float_precision}}, std={sig:{float_width}.{float_precision}}",
        )

    plt.text(
        x[0],
        1.075,
        "\n".join(stats_strs),
        bbox={"facecolor": "grey", "alpha": 0.3, "pad": 10},
    )

    plt.title(title, y=1.0, pad=-14, fontweight="bold")
    plt.legend(loc="upper right", bbox_to_anchor=(1, 1.125))
    plt.show()


if __name__ == "__main__":
    n_objectives = 5
    problem_name = "river_pollution"
    n_missing = 200
    n_runs = 200
    n_solutions = 10171
    asf_name = "pointmethodasf"
    use_original_problem = True
    pareto_as_missing = True

    improve_target = True
    worsen_random = False
    naive = False

    data = pd.DataFrame(
        columns=(
            "Delta",
            "ASF",
            "Strategy",
            "Success",
            "Neutral",
            "Failure",
            "median",
            "MAD",
            "min",
            "max",
            "mean_95",
            "std_mad",
        )
    )

    formatters = {
        "Success": lambda x: f"{x:.2f}",
        "Neutral": lambda x: f"{x:.2f}",
        "Failure": lambda x: f"{x:.2f}",
        "median": lambda x: f"{x:.2f}",
        "MAD": lambda x: f"{x:.2f}",
        "min": lambda x: f"{x:.2e}",
        "max": lambda x: f"{x:.2e}",
        "std_mad": lambda x: f"{x:.2f}",
    }

    asf_names = ["pointmethodasf", "guessasf", "stomasf"]
    deltas = [0.05, 0.10, 0.15, 0.20]

    def strategy_resolver(improve_target, worsen_random, naive):
        if naive:
            # return "BAU"
            return "E"
        elif improve_target and not worsen_random:
            # return "Modus operandi"
            return "A"
        elif improve_target and worsen_random:
            # return "Worsen random"
            return "B"
        elif not improve_target and not worsen_random:
            # return "Only worsen suggested"
            return "C"
        elif not improve_target and worsen_random:
            # return "No improving worsen random"
            return "D"
        else:
            return "WTF?"

    def asf_resolver(asf_name):
        if asf_name == "pointmethodasf":
            return "RPM"
        elif asf_name == "guessasf":
            return "GUESS"
        elif asf_name == "stomasf":
            return "STOM"
        else:
            return "WTF?"

    i = 0

    for asf_name in asf_names:
        for d in deltas:
            for improve_target in [True, False]:
                for worsen_random in [False, True]:
                    f_name_ = (
                        f"run_{problem_name}_{n_solutions}_per_objective_{n_runs}_{asf_name}_delta_"
                        f"{int(d*100)}{'_original_' if use_original_problem else '_'}{'pfmissing_' if pareto_as_missing else ''}"
                        f"{'nochange_' if not improve_target else ''}{'random' if worsen_random else ''}{'naive' if naive else ''}"
                    )
                    f_name = f_name_ + ".xlsx"

                    df = pd.read_excel(f"{DATA_DIR}/{f_name}", engine="openpyxl")
                    # title = f"{problem_name} - N=1000 - delta=0.{d:1d} - PF as missing - impair random (but not suggested)"
                    title = f"{problem_name} - N=1000 - delta={int(d*100)} - PF as missing - {f'{n_runs} per objective - '}{'improve target - ' if improve_target else 'do not improve target - '}{'worsen random' if worsen_random else 'do not worsen random'}"

                    relative_imprvs = (
                        compute_relative_improvements_of_targets_median_and_mad(
                            df, "f_", n_objectives
                        )
                    )
                    success_rates = compute_target_success_rates(df, n_objectives)

                    """
                    for i in range(n_objectives):
                        datum = (
                            f"f_{i+1}",
                            success_rates[i, 0],
                            success_rates[i, 1],
                            success_rates[i, 2],
                            relative_imprvs["median"][i],
                            relative_imprvs["mad"][i],
                            relative_imprvs["min"][i],
                            relative_imprvs["max"][i],
                        )
                        data.loc[i] = datum
                    """

                    data.loc[i] = (
                        f"{int(d*100)}",
                        asf_resolver(asf_name),
                        strategy_resolver(improve_target, worsen_random, naive),
                        np.mean(success_rates[:, 0]),
                        np.mean(success_rates[:, 1]),
                        np.mean(success_rates[:, 2]),
                        np.mean(relative_imprvs["median"]),
                        np.mean(relative_imprvs["mad"]),
                        np.mean(relative_imprvs["min"]),
                        np.mean(relative_imprvs["max"]),
                        f"{np.mean(relative_imprvs['mean95']):.2f}({relative_imprvs['outliers']/(n_objectives*n_runs)*100:.2f})",
                        np.mean(relative_imprvs["std_mad"]),
                    )

                    print(data)

                    latex_fname = DATA_DIR + "/../_tables/" + f_name_ + ".tex"
                    table_tex = data.to_latex(
                        formatters=formatters,
                        index=False,
                    )
                    data_tex = "\n".join(
                        map(lambda x: x.strip(), table_tex.splitlines()[4:-2])
                    )

                    i += 1

    # naive once
    improve_target = True
    worsen_random = False
    naive = True
    for d in deltas:
        for asf_name in asf_names:
            f_name_ = (
                f"run_{problem_name}_{n_solutions}_per_objective_{n_runs}_{asf_name}_delta_"
                f"{int(d*100)}{'_original_' if use_original_problem else '_'}{'pfmissing_' if pareto_as_missing else ''}"
                f"{'nochange_' if not improve_target else ''}{'random' if worsen_random else ''}{'naive' if naive else ''}"
            )
            f_name = f_name_ + ".xlsx"

            df = pd.read_excel(f"{DATA_DIR}/{f_name}", engine="openpyxl")
            # title = f"{problem_name} - N=1000 - delta=0.{d:1d} - PF as missing - impair random (but not suggested)"
            title = f"{problem_name} - N=1000 - delta={int(d*100)} - PF as missing - {f'{n_runs} per objective - '}{'improve target - ' if improve_target else 'do not improve target - '}{'worsen random' if worsen_random else 'do not worsen random'}"

            relative_imprvs = compute_relative_improvements_of_targets_median_and_mad(
                df, "f_", n_objectives
            )
            success_rates = compute_target_success_rates(df, n_objectives)

            """
            for i in range(n_objectives):
                datum = (
                    f"f_{i+1}",
                    success_rates[i, 0],
                    success_rates[i, 1],
                    success_rates[i, 2],
                    relative_imprvs["median"][i],
                    relative_imprvs["mad"][i],
                    relative_imprvs["min"][i],
                    relative_imprvs["max"][i],
                )
                data.loc[i] = datum
            """

            data.loc[i] = (
                f"{int(d*100)}",
                asf_resolver(asf_name),
                strategy_resolver(improve_target, worsen_random, naive),
                np.mean(success_rates[:, 0]),
                np.mean(success_rates[:, 1]),
                np.mean(success_rates[:, 2]),
                np.mean(relative_imprvs["median"]),
                np.mean(relative_imprvs["mad"]),
                np.mean(relative_imprvs["min"]),
                np.mean(relative_imprvs["max"]),
                f"{np.mean(relative_imprvs['mean95']):.2f}({relative_imprvs['outliers']/(n_objectives*n_runs)*100:.2f})",
                np.mean(relative_imprvs["std_mad"]),
            )

            print(data)

            i += 1

            # with open(latex_fname, "w") as f:
            # f.write(data_tex)

    table_tex = data.to_latex(
        DATA_DIR + "/../_tables/big_table_river.tex",
        formatters=formatters,
        index=False,
    )

"""
    # missing vs delta
    d = 10
    fig, axs = plt.subplots(2, 5)

    for i, d in enumerate(DELTAS):
        successes = []
        neutrals = []
        fails = []
        for m in MISSINGS:
            f_name = f"run_river_5000_n_1000_missing_{m}_even_delta_{d}.xlsx"
            df = pd.read_excel(f"{DATA_DIR}/{f_name}", engine="openpyxl")
            effects = get_effect_result(df)
            success = sum([1 if t == 1 else 0 for t in effects])
            no_change = sum([1 if t == 0 else 0 for t in effects])
            fail = sum([1 if t == -1 else 0 for t in effects])

            successes.append(success)
            neutrals.append(no_change)
            fails.append(fail)

        axs[int(i / 5), i % 5].plot(MISSINGS, successes, c="green", label="Ok")
        axs[int(i / 5), i % 5].plot(MISSINGS, neutrals, c="orange", label="No change")
        axs[int(i / 5), i % 5].plot(MISSINGS, fails, c="red", label="Fail")
        axs[int(i / 5), i % 5].set_title(f"delta={d}")

    plt.show()


    pf = pd.read_csv("./data/river_pollution_5000.csv").to_numpy()[:, 0:n_objectives]
    fig, axs = plt.subplots(2, 2)

    colors = ["red", "orange", "green"]

    axs[0, 0].scatter(pf[:, 0], pf[:, 1], s=4,c="grey")
    axs[0, 0].scatter(original_rps[:, 0], original_rps[:, 1], s=4, c=[colors[i+1] for i in effects])

    axs[0, 1].scatter(pf[:, 0], pf[:, 2], s=4,c="grey")
    axs[0, 1].scatter(original_rps[:, 0], original_rps[:, 2], s=4)

    axs[0, 1].scatter(pf[:, 0], pf[:, 3], s=4,c="grey")
    axs[1, 0].scatter(original_rps[:, 0], original_rps[:, 3], s=4)
    
    axs[0, 1].scatter(pf[:, 0], pf[:, 4], s=4,c="grey")
    axs[1, 1].scatter(original_rps[:, 0], original_rps[:, 4], s=4)
    plt.show()
"""
