from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


METRICS = ["Accuracy", "Precision", "Recall"]

EXPERIMENTS = {
    "hidden-layer dimension": {
        "values": ["48", "96", "192"],
        "PD": [
            [0.7073, 0.7065, 0.7121],
            [0.6956, 0.7346, 0.7413],
            [0.7025, 0.7084, 0.6918],
        ],
        "ADHD": [
            [0.7056, 0.7234, 0.7316],
            [0.7292, 0.7533, 0.7648],
            [0.7133, 0.7295, 0.7482],
        ],
    },
    "learn rate": {
        "values": ["0.0001", "0.0002", "0.0004"],
        "PD": [
            [0.7063, 0.7261, 0.7149],
            [0.6956, 0.7346, 0.7413],
            [0.6827, 0.7078, 0.7132],
        ],
        "ADHD": [
            [0.7165, 0.7083, 0.7144],
            [0.7292, 0.7533, 0.7648],
            [0.7393, 0.7274, 0.7283],
        ],
    },
    "dropout": {
        "values": ["0.2", "0.25", "0.3"],
        "PD": [
            [0.7034, 0.7161, 0.7085],
            [0.6956, 0.7346, 0.7413],
            [0.6993, 0.7247, 0.7245],
        ],
        "ADHD": [
            [0.7147, 0.7258, 0.7342],
            [0.7292, 0.7533, 0.7648],
            [0.7014, 0.7127, 0.7176],
        ],
    },
}

COLORS = ["#ff3f00", "#ffa726", "#20a486"]
HATCHES = ["....", "++++", "xxxx"]


def configure_style() -> None:
    """Use a clean style close to the reference figure."""
    plt.rcParams["font.sans-serif"] = [
        "SimHei",
        "Microsoft YaHei",
        "Noto Sans CJK SC",
        "Noto Sans CJK JP",
        "Noto Sans CJK HK",
        "Droid Sans Fallback",
        "AR PL UMing CN",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = 120
    plt.rcParams["savefig.dpi"] = 300


def add_value_labels(ax: plt.Axes, bars) -> None:
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + 0.0025,
            f"{height:.4f}",
            ha="center",
            va="bottom",
            fontsize=6,
            color="#444444",
        )


def draw_parameter_experiment() -> Path:
    configure_style()

    fig, axes = plt.subplots(
        nrows=len(EXPERIMENTS),
        ncols=2,
        figsize=(13, 10),
        sharey=False,
        constrained_layout=True,
    )

    x = np.arange(len(METRICS))
    width = 0.22
    dataset_names = ["PD", "ADHD"]

    for row, (param_name, experiment) in enumerate(EXPERIMENTS.items()):
        labels = experiment["values"]

        for col, dataset_name in enumerate(dataset_names):
            ax = axes[row, col]
            scores = np.asarray(experiment[dataset_name])

            for idx, label in enumerate(labels):
                offset = (idx - (len(labels) - 1) / 2) * width
                bars = ax.bar(
                    x + offset,
                    scores[idx],
                    width=width,
                    label=f"{param_name} {label}",
                    color=COLORS[idx],
                    edgecolor="white",
                    linewidth=0.9,
                    hatch=HATCHES[idx],
                    alpha=0.95,
                )
                add_value_labels(ax, bars)

            ax.set_title(f"{dataset_name} Dataset", fontsize=11, pad=9)
            ax.set_xticks(x)
            ax.set_xticklabels(METRICS)
            ax.set_ylabel("Performance")
            ax.grid(axis="both", linestyle="--", linewidth=0.6, color="#d9d9d9", alpha=0.55)
            ax.set_axisbelow(True)

            y_min = max(0.60, float(scores.min()) - 0.035)
            y_max = min(0.82, float(scores.max()) + 0.055)
            ax.set_ylim(y_min, y_max)

            ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.27), ncol=3, fontsize=8)

    fig.suptitle("GCN 参数实验结果对比", fontsize=16, y=1.02)

    output_path = Path(__file__).with_name("参数实验结果对比.png")
    fig.savefig(output_path, bbox_inches="tight")
    plt.show()
    return output_path


if __name__ == "__main__":
    saved_path = draw_parameter_experiment()
    print(f"Figure saved to: {saved_path}")
