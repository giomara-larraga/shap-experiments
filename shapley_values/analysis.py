import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import List

DATA_DIR = "/home/kilo/workspace/shap-experiments/_results"
N = 1000
DELTAS = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]


def get_original_rp(fname: str, objective_prefix: List[str], n_objectives: int, data_dir: str = DATA_DIR):
    return 


successess = []
no_changes = []
fails = []

for d in DELTAS:
    f_name = f"run_river_5000_n_1000_missing_1024_even_delta_{d}.xlsx"
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