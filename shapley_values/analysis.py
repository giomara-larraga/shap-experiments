import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import List

DATA_DIR = "/home/kilo/workspace/shap-experiments/_results"
N = 1000
DELTAS = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
MISSINGS = [32, 243, 1024, 3125, 7776, 16807]


def get_original_rps(dframe: pd.DataFrame, objective_prefix: str, n_objectives: int) -> np.ndarray:
    """Fetch the original reference points from a data frame.

    Args:
        dframe (pd.DataFrame): The data frame.
        objective_prefix (str): The prefix of the objectives names. E.g., 'f_' for "f_1, f_2, f_3, etc..."
        n_objectives (int): The number of objectives.

    Returns:
        np.ndarray: A numpy array containing the reference points on each row.
    """    
    table_prefix = "Original ref point "
    stack_me = list(dframe.loc[:, f"{table_prefix}{objective_prefix}{i}"].to_numpy() for i in range(1, n_objectives+1))
    return np.vstack(stack_me).T


def get_original_solutions(dframe: pd.DataFrame, objective_prefix: str, n_objectives: int) -> np.ndarray:
    """Fetch the original solutions from a data frame.

    Args:
        dframe (pd.DataFrame): The data frame.
        objective_prefix (str): The prefix of the objective names. E.g., 'f_' for "f_1, f_2, f_3, etc..."
        n_objectives (int): The number of objectives

    Returns:
        np.ndarray: A numpy array containing the solutions on each row.
    """    
    table_prefix = "Original solution "
    stack_me = list(dframe.loc[:, f"{table_prefix}{objective_prefix}{i}"].to_numpy() for i in range(1, n_objectives+1))
    return np.vstack(stack_me).T


def get_new_solutions(dframe: pd.DataFrame, objective_prefix: str, n_objectives: int) -> np.ndarray:
    """Feth the new solutions from a data frame.

    Args:
        dframe (pd.DataFrame): The data frame.
        objective_prefix (str): The prefix of the objective names. E.g., 'f_' for "f_1, f_2, f_3, etc..."
        n_objectives (int): The number of objectives.

    Returns:
        np.ndarray: A numpy array containing the new solutions on each row.
    """    
    table_prefix = "New solution based on new ref point "
    stack_me = list(dframe.loc[:, f"{table_prefix}{objective_prefix}{i}"].to_numpy() for i in range(1, n_objectives+1))
    return np.vstack(stack_me).T


def get_new_rps(dframe: pd.DataFrame, objective_prefix: str, n_objectives: int) -> np.ndarray:
    """Fetch the new solutions from a data frame.

    Args:
        dframe (pd.DataFrame): The data frame.
        objective_prefix (str): The prefix of the objective names. E.g., 'f_' for "f_1, f_2, f_3, etc..."
        n_objectives (int): The number of objectives

    Returns:
        np.ndarray: A numpy array containing the new solutions on each row.
    """    
    table_prefix = "New ref point based on suggestion "
    stack_me = list(dframe.loc[:, f"{table_prefix}{objective_prefix}{i}"].to_numpy() for i in range(1, n_objectives+1))
    return np.vstack(stack_me).T

def get_effect_result(dframe: pd.DataFrame) -> np.ndarray:
    """Return the effect result for each run (1: success, 0: no change, -1: failure) from a data frame.

    Args:
        dframe (pd.DataFrame): The data frame.

    Returns:
        np.ndarray: A numpy array with the effect results.
    """    
    return dframe["Was the desired effect achieved?"].to_numpy()

"""
successess = []
no_changes = []
fails = []

# for d in DELTAS:
d = 2
f_name = f"run_river_5000_n_1000_missing_1024_even_delta_{d}_original.xlsx"
df = pd.read_excel(f"{DATA_DIR}/{f_name}", engine="openpyxl")

slice = df.loc[:, "Was the desired effect achieved?"]

success = sum([1 if t == 1 else 0 for t in slice])
no_change = sum([1 if t == 0 else 0 for t in slice])
fail = sum([1 if t == -1 else 0 for t in slice])

successess.append(success)
no_changes.append(no_change)
fails.append(fail)

print(f"Out of 1000 run (delta={d}): {success} were successful, {no_change} had no change, {fail} were a failure.")


plt.plot(DELTAS, successess, label="OK")
plt.plot(DELTAS, no_changes, label="Neutral")
plt.plot(DELTAS, fails, label="Fail")
plt.legend()

plt.show()
"""

if __name__ == "__main__":
    n_objectives = 5
    d = 20
    f_name = f"run_river_5000_n_1000_missing_1024_even_delta_{d}_original.xlsx"
    df = pd.read_excel(f"{DATA_DIR}/{f_name}", engine="openpyxl")

    original_rps = get_original_rps(df, "f_", 5)
    original_solutions = get_original_solutions(df, "f_", 5)
    new_ref_points = get_new_rps(df, "f_", 5)
    new_solutions = get_new_solutions(df, "f_", 5)
    effects = get_effect_result(df)

    success = sum([1 if t == 1 else 0 for t in effects])
    no_change = sum([1 if t == 0 else 0 for t in effects])
    fail = sum([1 if t == -1 else 0 for t in effects])

    print(f"Out of 1000 run (delta={d}): {success} were successful, {no_change} had no change, {fail} were a failure.")
    exit()

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

        axs[int(i/5), i%5].plot(MISSINGS, successes, c="green", label="Ok")
        axs[int(i/5), i%5].plot(MISSINGS, neutrals, c="orange", label="No change")
        axs[int(i/5), i%5].plot(MISSINGS, fails, c="red", label="Fail")
        axs[int(i/5), i%5].set_title(f"delta={d}")

    plt.show()
        






"""
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