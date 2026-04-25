import argparse

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

import numpy as np
import matplotlib.pyplot as plt


def plot_cpu_gpu_comparison(
    cpu_data,
    gpu_data,
    algos_names,
    x_values,
    x_label,
    title,
    save_path="_cpu_gpu_compare.png",
):
    fig, ax_cpu = plt.subplots(figsize=(5.5, 3.5))

    ax_gpu = ax_cpu.twinx()

    colors = plt.cm.tab10.colors

    cpu_lines = []
    gpu_lines = []

    for i, (algo_name, cpu_pair, gpu_pair) in enumerate(
        zip(algos_names, cpu_data, gpu_data)
    ):
        cpu_mean, cpu_std = np.asarray(cpu_pair[0]), np.asarray(cpu_pair[1])
        gpu_mean, gpu_std = np.asarray(gpu_pair[0]), np.asarray(gpu_pair[1])

        color = colors[i % len(colors)]

        # ======================
        # CPU
        # ======================
        (line_cpu,) = ax_cpu.plot(
            x_values,
            cpu_mean,
            linestyle="solid",
            color=color,
        )
        # ax_cpu.fill_between(
        #     x_values,
        #     np.maximum(0.0, cpu_mean - cpu_std),
        #     cpu_mean + cpu_std,
        #     color=color,
        #     alpha=0.15,
        # )

        # ======================
        # GPU
        # ======================
        (line_gpu,) = ax_gpu.plot(
            x_values,
            gpu_mean,
            linestyle="dashed",
            color=color,
        )
        # ax_gpu.fill_between(
        #     x_values,
        #     np.maximum(0.0, gpu_mean - gpu_std),
        #     gpu_mean + gpu_std,
        #     color=color,
        #     alpha=0.15,
        # )

        cpu_lines.append(line_cpu)
        gpu_lines.append(line_gpu)

    # ======================
    # Axes labeling
    # ======================
    ax_cpu.set_xlabel(x_label)
    ax_cpu.set_ylabel("CPU Runtime log(sec)")
    ax_gpu.set_ylabel("GPU Runtime log(sec)")

    ax_cpu.set_xticks(x_values)
    ax_cpu.grid(True, linestyle="dotted", alpha=0.4)

    # ======================
    # Legends
    # ======================

    algo_legend = ax_cpu.legend(
        cpu_lines,
        algos_names,
        title="Algorithms",
        loc="lower right",
        fontsize=8,
    )

    from matplotlib.lines import Line2D

    device_legend_lines = [
        Line2D([0], [0], color="black", linestyle="solid"),
        Line2D([0], [0], color="black", linestyle="dashed"),
    ]

    device_legend = ax_cpu.legend(
        device_legend_lines,
        ["CPU", "GPU"],
        title="Device",
        loc="upper left",
        fontsize=8,
    )

    ax_cpu.add_artist(algo_legend)
    
    ax_cpu.set_yscale("log")
    ax_gpu.set_yscale("log")

    # ======================
    # Final touches
    # ======================
    plt.title(title)
    plt.tight_layout()
    plt.savefig("_".join(title.lower().split()) + save_path, dpi=300)
    plt.close()


def collect_data(
    dir: str,
    mode_postfix: str,
    column_to_agg: int,
    prefixes: list = [32, 64, 128, 256, 512],
    stopwatch_freq: int = 10000000,
):
    mean_runtimes = []
    std_runtimes = []
    for prefix in prefixes:
        try:
            speed_data = np.loadtxt(
                Path.joinpath(Path(dir), Path(str(prefix) + mode_postfix)),
                delimiter="\t",
                skiprows=1,
            )[:, column_to_agg] / float(stopwatch_freq)
            mean_runtimes.append(np.mean(speed_data))
            std_runtimes.append(np.std(speed_data))
        except Exception as e:
            print(f"Failed to find a file. {e}")
    return (mean_runtimes, std_runtimes)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Speedup evaluation script")
    parser.add_argument(
        "--data-dirs",
        type=str,
        required=True,
        help="The paths to directories with speed measurements.",
    )

    parser.add_argument(
        "--algos-names",
        type=str,
        required=True,
        help="The names of algos that are compared.",
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

    parser.add_argument(
        "--x-axis-title",
        type=str,
        required=True,
        help="The title to add to x axis.",
    )

    parser.add_argument(
        "--x-axis-ticks",
        type=str,
        required=True,
        help="The ticks to add to x axis.",
    )

    args = parser.parse_args()
    ticks = [int(tick) for tick in args.x_axis_ticks.split()]
    algos_names = [name for name in args.algos_names.split()]

    cpu_data = []
    gpu_data = []

    for data_dir in args.data_dirs.split():

        cpu_runtime_mean, cpu_runtime_std = collect_data(
            data_dir,
            "_cpu_performance_evaluation.txt",
            1,
            ticks,
            args.stopwatch_freq,
        )

        cpu_data.append((cpu_runtime_mean, cpu_runtime_std))

        gpu_runtime_mean, gpu_runtime_std = collect_data(
            data_dir,
            "_sync_gpu_performance_evaluation.txt",
            1,
            ticks,
            args.stopwatch_freq,
        )

        gpu_data.append((gpu_runtime_mean, gpu_runtime_std))

    plot_cpu_gpu_comparison(
        cpu_data,
        gpu_data,
        algos_names,
        ticks,
        args.x_axis_title,
        args.plot_title,
    )
