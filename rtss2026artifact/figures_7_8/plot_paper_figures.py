#!/usr/bin/env python3
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import pandas as pd
import seaborn as sns


def read_csv(path, label, green=False):
    frame = pd.read_csv(path)
    if green:
        # The original GC plot takes the max over each ten-repetition chunk.
        frame["Chunk"] = (frame["Experiment"] - 1) // 10
        frame = frame.groupby(["Threads", "Chunk"], as_index=False)["MaxExecTime"].max()
    frame["Label"] = label
    return frame


def plot(files, output):
    data = pd.concat([read_csv(*args) for args in files], ignore_index=True)
    visible_max = data.groupby(["Threads", "Label"])["MaxExecTime"].quantile(0.95).max()
    fig, axis = plt.subplots(figsize=(7.2, 4.2))
    sns.boxplot(data=data, x="Threads", y="MaxExecTime", hue="Label", palette="Paired",
                showfliers=False, whis=(0, 95), linewidth=0.5, width=0.72, ax=axis)
    axis.set(xlabel="Concurrent Host Threads", ylabel="Latency (ms)", ylim=(0, visible_max * 1.1))
    axis.yaxis.set_major_locator(MaxNLocator(integer=True))
    axis.grid(True, axis="y", linestyle="--", alpha=0.45)
    axis.grid(True, axis="x", linestyle="--", alpha=0.35)
    axis.legend(ncol=2, frameon=True, fancybox=False, edgecolor="0.8", loc="upper left")
    sns.despine(axis=axis)
    fig.tight_layout()
    fig.savefig(output, dpi=600)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")
    baseline = [(args.results / "baseline" / f"malloc_{n}.csv", f"Malloc: {n}", False) for n in (512, 1024, 2048)]
    baseline += [(args.results / "baseline" / f"matmul_{n}.csv", f"MatMul: {n}", False) for n in (512, 1024, 2048)]
    green = [(args.results / "green" / f"malloc_sm{n}.csv", f"Malloc: 1024, SM: {n}", True) for n in (4, 6, 8, 16, 32)]
    green += [(args.results / "green" / f"matmul_sm{n}.csv", f"MatMul: 512, SM: {n}", True) for n in (4, 6, 8, 16, 32)]
    plot(baseline, args.output / "malloc_vs_matmul_max_ops100.png")
    plot(green, args.output / "gc_malloc_vs_matmul_max_ops100.png")


if __name__ == "__main__":
    main()
