# SPEED: six dimensions (no-erosionX4, particle-erosion, cellular-erosion) X 2 plots: actual time (ms) + speedup
# ->
## one plot: top section - 2x2 pure algo, bottom section - 1x2 erosion. Actual time and speedup can be fitted on the same plot. Just use different scales.

import argparse

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

import numpy as np
import matplotlib.pyplot as plt


def plot_cpu_gpu_speedup(
    cpu_mean,
    cpu_std,
    gpu_mean,
    gpu_std,
    x_values=None,
    title: str = "CPU vs GPU Performance",
):
    save_path = "_cpu_gpu_speedup.png"
    cpu_mean = np.asarray(cpu_mean)
    cpu_std = np.asarray(cpu_std)
    gpu_mean = np.asarray(gpu_mean)
    gpu_std = np.asarray(gpu_std)

    if x_values is None:
        x_values = np.arange(len(cpu_mean))

    speedup = cpu_mean / gpu_mean

    fig, ax1 = plt.subplots(figsize=(5.5, 3.5))

    # ======================
    # LEFT AXIS
    # ======================
    ax1.plot(x_values, cpu_mean, label="CPU Mean Runtime", linestyle="solid")
    ax1.fill_between(
        x_values,
        np.maximum(0.0, cpu_mean - cpu_std),
        cpu_mean + cpu_std,
        alpha=0.2,
    )

    ax1.plot(x_values, gpu_mean, label="GPU Mean Runtime", linestyle="dashed")
    ax1.fill_between(
        x_values,
        np.maximum(0.0, gpu_mean - gpu_std),
        gpu_mean + gpu_std,
        alpha=0.2,
    )

    ax1.set_xlabel("Heightmap side size (texels)")
    ax1.set_ylabel("Runtime log(sec)")
    ax1.set_xticks(x_values)
    ax1.set_yscale("log")
    ax1.grid(True, linestyle="dotted", alpha=0.4)

    ax2 = ax1.twinx()

    ax2.plot(
        x_values,
        speedup,
        label="Speedup (CPU/GPU)",
        linestyle="dotted",
    )

    ax2.set_ylabel("Speedup (× faster)")

    # ======================
    # LEGEND
    # ======================
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()

    ax1.legend(
        lines_1 + lines_2,
        labels_1 + labels_2,
        loc="lower right",
        fontsize=8,
    )

    plt.title(title)
    plt.tight_layout()
    plt.savefig("_".join(title.lower().split()) + save_path, dpi=300)
    plt.close()


def collect_data(
    dir: str,
    mode_postfix: str,
    column_to_agg: int,
    resolutions: list = [32, 64, 128, 256, 512],
    stopwatch_freq: int = 10000000,
    discard_first: bool = False
):
    mean_runtimes = []
    std_runtimes = []
    for resolution in resolutions:
        try:
            speed_data = np.loadtxt(
                Path.joinpath(Path(dir), Path(str(resolution) + mode_postfix)),
                delimiter="\t",
                skiprows=1,
            )[:, column_to_agg] / float(stopwatch_freq)

            if discard_first:
                speed_data = speed_data[5:]

            mean_runtimes.append(np.mean(speed_data))
            std_runtimes.append(np.std(speed_data))
        except Exception as e:
            print(f"Failed to find a file. {e}")
    return (mean_runtimes, std_runtimes)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Speedup evaluation script")
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="The path to a directory with speed measurements.",
    )
    parser.add_argument(
        "--stopwatch-freq",
        type=int,
        required=True,
        help="The frequency of C# stopwatch used for profiling (system-dependent).",
    )
    parser.add_argument(
        "--plot-title",
        type=str,
        required=True,
        help="The title to add.",
    )

    args = parser.parse_args()
    resolutions = [
        32,
        64, 128, 256, 512
        #1024
        ]

    cpu_runtime_mean, cpu_runtime_std = collect_data(
        args.data_dir,
        "_cpu_performance_evaluation.txt",
        1,
        resolutions,
        args.stopwatch_freq,
    )
    gpu_runtime_mean, gpu_runtime_std = collect_data(
        args.data_dir,
        "_sync_gpu_performance_evaluation.txt",
        1,
        resolutions,
        args.stopwatch_freq,
        True
    )

    plot_cpu_gpu_speedup(
        cpu_runtime_mean,
        cpu_runtime_mean,
        gpu_runtime_mean,
        gpu_runtime_std,
        title=args.plot_title,
        x_values=resolutions,
    )
