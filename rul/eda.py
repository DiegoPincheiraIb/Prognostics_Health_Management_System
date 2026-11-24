"""Create a simple, reproducible FD001 sensor exploration report."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from rul.data import FEATURES, load_cycles


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="data/CMAPSSData/train_FD001.txt")
    parser.add_argument("--output", default="artifacts/eda")
    args = parser.parse_args()
    data = load_cycles(Path(args.input))
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    data.describe().to_csv(output / "summary.csv")
    data.groupby("unit").cycle.max().describe().to_csv(output / "lifetimes.csv")
    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    for unit, group in data.groupby("unit"):
        for ax, feature in zip(axes, FEATURES[3:6], strict=True):
            ax.plot(group.cycle, group[feature], alpha=0.25)
            ax.set_ylabel(feature)
    axes[-1].set_xlabel("Cycle")
    fig.tight_layout()
    fig.savefig(output / "sensor_evolution.png", dpi=120)
    plt.close(fig)
    print(f"EDA saved to {output}")


if __name__ == "__main__":
    main()