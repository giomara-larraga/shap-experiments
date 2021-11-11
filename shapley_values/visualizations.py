import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def plot_mean_rates_per_strategy(data, problem_name):
    cols = data[["Strategy", "Success", "Neutral", "Failure"]]
    grouped_mean = cols.groupby("Strategy").mean()
    grouped_std = cols.groupby("Strategy").std()

    labels = list(grouped_mean.index)
    values = grouped_mean.to_numpy()
    success = values[:, 0]
    neutral = values[:, 1]
    failure = values[:, 2]

    errors = grouped_std.to_numpy()
    success_err = errors[:, 0]
    neutral_err = errors[:, 1]
    failure_err = errors[:, 2]

    fig, ax = plt.subplots()

    width = 0.65
    capsize = 3
    success_color = "#68A828"
    neutral_color = "#E56D31"
    failure_color = "#C8331B"
    success_ecolor = "#66ff00"
    neutral_ecolor = "#ffff00"
    failure_ecolor = "#ff160c"
    ax.bar(
        labels,
        success,
        yerr=success_err,
        label="Success",
        color=success_color,
        ecolor=success_ecolor,
        capsize=capsize,
        width=width,
    )
    ax.bar(
        labels,
        neutral,
        yerr=neutral_err,
        bottom=success,
        label="Neutral",
        color=neutral_color,
        ecolor=neutral_ecolor,
        capsize=capsize,
        width=width,
    )
    ax.bar(
        labels,
        failure,
        yerr=failure_err,
        bottom=success + neutral,
        label="Failure",
        color=failure_color,
        ecolor=failure_ecolor,
        capsize=capsize,
        width=width,
    )

    ax.set_ylabel("Rate")
    ax.set_xlabel("Strategy")
    ax.set_title(
        f"Average success rates for each strategy for the {problem_name} problem"
    )

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    ax.grid(axis="y", linestyle="-", linewidth=0.5)

    plt.show()


def plot_mean_changes_per_strategy_and_delta(data, problem_name, what):
    data["Mean_95"] = data["Mean_95"].apply(lambda x: float(x[: x.rfind("(")]))
    cols = data[["Strategy", "Delta", what]]

    grouped_mean = cols.groupby(["Strategy", "Delta"]).mean()

    label_names = ["A", "B", "C", "D", "E"]
    labels = np.arange(len(label_names))

    values = grouped_mean.to_numpy()
    values_5 = values[np.arange(0, 20, 4), 0]
    values_10 = values[np.arange(1, 20, 4), 0]
    values_15 = values[np.arange(2, 20, 4), 0]
    values_20 = values[np.arange(3, 20, 4), 0]

    fig, ax = plt.subplots()

    width = 0.20
    color_5 = "#a8e6cf"
    color_10 = "#dcedc1"
    color_15 = "#ffd3b6"
    color_20 = "#ffaaa5"
    ax.bar(
        labels,
        values_5,
        # yerr=values_5_err,
        label="$\delta = 5\%$",
        color=color_5,
        # ecolor=success_ecolor,
        # capsize=capsize,
        width=width,
    )
    ax.bar(
        labels + width,
        values_10,
        # yerr=values_5_err,
        label="$\delta = 10\%$",
        # bottom=values_5,
        color=color_10,
        # ecolor=success_ecolor,
        # capsize=capsize,
        width=width,
    )
    ax.bar(
        labels + 2 * width,
        values_15,
        # yerr=values_5_err,
        label="$\delta = 15\%$",
        # bottom=values_5 + values_10,
        color=color_15,
        # ecolor=success_ecolor,
        # capsize=capsize,
        width=width,
    )
    ax.bar(
        labels + 3 * width,
        values_20,
        # yerr=values_5_err,
        label="$\delta = 20\%$",
        # bottom=values_5 + values_10 + values_15,
        color=color_20,
        # ecolor=success_ecolor,
        # capsize=capsize,
        width=width,
    )

    ax.set_ylabel("Change (%)")
    ax.set_xlabel("Strategy")
    ax.set_title(
        (
            "The average of the median absolute deviations observed in "
            f"the \ntarget for each strategy and perturbation $\delta$ for the {problem_name} problem"
        )
    )
    ax.set_xticks(labels + 3 * width / 2)
    ax.set_xticklabels(label_names)

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    ax.grid(axis="y", linestyle="-", linewidth=0.5)

    print(grouped_mean)

    # plt.gca().invert_yaxis()
    plt.show()


if __name__ == "__main__":
    data = pd.read_excel(
        "/home/kilo/workspace/shap-experiments/_results/car_excel.ods",
    )
    problem_name = "car"
    plot_mean_rates_per_strategy(data, problem_name)
    # plot_mean_changes_per_strategy_and_delta(data, problem_name, "MAD")
