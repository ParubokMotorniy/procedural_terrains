# AESTHETICS: 2 groups (tuned, random) X 2 classifiers (LLM, custom-trained) X 12 dimensions (4 algos X 3 erosion modes)
# ->
## Custom-model class X tuned : 3(erosion modes)X4(algos) boxplots
## Custom-model class X random : 3(erosion modes)X4(algos) boxplots
## LLM class X tuned : 3(erosion modes)X4(algos) boxplots
## LLM class X random : 3(erosion modes)X4(algos) boxplots

import argparse

import classificationlib as classlib
import evaluationlib as elib
import experiments.llm_baseline_provider as llmlib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import os

import seaborn as sns


def plot_boxplots(classification_results, algos_names, title, subtitle, save_path):
    data = []
    for algo_name, values in zip(algos_names, classification_results):
        for v in values:
            data.append({"Algorithm": algo_name, "Score": v})

    df = pd.DataFrame(data)

    plt.figure(figsize=(5, 5))
    ax = sns.boxplot(
        x="Algorithm",
        y="Score",
        data=df,
        width=0.45,
        whis=[0, 100],
        palette="pastel",
        showfliers=False,
    )

    sns.stripplot(
        x="Algorithm",
        y="Score",
        data=df,
        color="black",
        size=2,
        alpha=0.3,
        jitter=True,
    )

    # ======================
    # Annotations
    # ======================
    for i, algo_name in enumerate(algos_names):
        values = np.array(classification_results[i])

        q0 = np.percentile(values, 0)
        q1 = np.percentile(values, 25)
        median = np.percentile(values, 50)
        q3 = np.percentile(values, 75)
        q4 = np.percentile(values, 100)

        x = i

        x_offset = 0.25

        # Median annotation
        ax.text(
            x + x_offset,
            median,
            f"{median:.2f}",
            va="center",
            ha="left",
            fontsize=8,
            color="black",
        )

        # Q1 annotation
        if np.abs(median - q1) >= 0.035:
            ax.text(
                x + x_offset,
                q1,
                f"{q1:.2f}",
                va="center",
                ha="left",
                fontsize=7,
                color="gray",
            )

        # Q3 annotation
        if np.abs(median - q3) >= 0.035:
            ax.text(
                x + x_offset,
                q3,
                f"{q3:.2f}",
                va="center",
                ha="left",
                fontsize=7,
                color="gray",
            )

        # Q0 annotation
        ax.text(
            x + x_offset,
            q0,
            f"{q0:.2f}",
            va="center",
            ha="left",
            fontsize=7,
            color="lightgray",
        )

        # Q4 annotation
        ax.text(
            x + x_offset,
            q4,
            f"{q4:.2f}",
            va="center",
            ha="left",
            fontsize=7,
            color="lightgray",
        )

    plt.ylim(0.0, 1.0)
    plt.ylabel("Classification score")
    plt.title(f"{title}\n{subtitle}")
    plt.grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def classify_and_plot_heightmaps(
    feature_dirs: list,
    algos_names: list,
    classfication_mode: str,
    custom_title: str,
    sub_title: str,
):
    if classfication_mode == "custom":
        classification_results = []
        for dir_path in feature_dirs:

            feature_vector_file_name = f"{Path(dir_path).name}_feature_vectors_new{"_".join(sub_title.lower().replace(":", "_").split())
            + "_".join(custom_title.lower().replace(":", "_").split())}.csv"
            if Path.exists(Path(feature_vector_file_name)):
                print("Loading precomputed feature vectors!")
                algo_feature_vectors_pd = pd.read_csv(feature_vector_file_name)
                algo_feature_vectors = algo_feature_vectors_pd.to_numpy()[:, 1:]
            else:
                print("Recomputing feature vectors!")
                algo_feature_vectors = elib.build_metric_vectors(
                    dir_path,
                    args.chunk_size,
                    args.division_depth,
                    "jpg",
                    1.0,
                    False,
                )
                np.nan_to_num(
                    algo_feature_vectors, copy=False
                )  # mostly handled in metric computers, so mere safegurad here

                features_pd = pd.DataFrame(algo_feature_vectors)
                features_pd.to_csv(feature_vector_file_name)

            algo_classification_results = classlib.classify_with_saved_models(
                algo_feature_vectors, "test_model_full", False
            )["mlp_prob"].to_numpy()

            # algo_classification_results = classlib.classify_with_ensemble(
            #     algo_feature_vectors,
            #     model_prefix="test_ensemble",
            #     use_data_ensemble=False,
            # )["ensemble_prob"].to_numpy()

            classification_results.append(algo_classification_results)

        plot_boxplots(
            classification_results,
            algos_names,
            custom_title,
            sub_title,
            "_".join(sub_title.lower().replace(":", "_").split())
            + "_".join(custom_title.lower().replace(":", "_").split())
            + "_custom_boxplot.png",
        )

    elif classfication_mode == "llm":
        classification_results = []
        for dir_path in [args.fft_dir, args.rmd_dir, args.sdf_dir, args.un_dir]:
            llm_class_file_name = f"{Path(dir_path).name}_llm_rate{"_".join(sub_title.lower().replace(":", "_").split())
            + "_".join(custom_title.lower().replace(":", "_").split())}.csv"
            if Path.exists(Path(llm_class_file_name)):
                print("Loading precomputed LLM rating!")
                algo_classification_results = pd.read_csv(
                    llm_class_file_name
                ).to_numpy()[:, 1:]
            else:
                print(f"Rerating {Path(dir_path).name} with LLM!")
                algo_classification_results = llmlib.classification_routine(
                    dir_path, f"class_res_{Path(dir_path).name}"
                )

                pd.DataFrame(algo_classification_results).to_csv(llm_class_file_name)

            classification_results.append(algo_classification_results.ravel())

        plot_boxplots(
            classification_results,
            algos_names,
            custom_title,
            sub_title,
            "_".join(sub_title.lower().replace(":", "_").split())
            + "_".join(custom_title.lower().replace(":", "_").split())
            + "_llm_boxplot.png",
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Aesthetics evaluation script")
    parser.add_argument(
        "--fft-dir", type=str, required=True, help="The path to FFT heightmap data."
    )
    parser.add_argument(
        "--rmd-dir", type=str, required=True, help="The path to RMD heightmap data."
    )
    parser.add_argument(
        "--sdf-dir", type=str, required=True, help="The path to SDF heightmap data."
    )
    parser.add_argument(
        "--un-dir", type=str, required=True, help="The path to UN heightmap data."
    )
    parser.add_argument(
        "--class-mode",
        choices=["llm", "custom"],
        required=True,
        help="What model to use for classification.",
    )
    parser.add_argument(
        "--custom-model-path",
        type=str,
        help="Where .joblibs of custom models are stored.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        help="Should such a need arise, the heightmap will be split into chunks of size NxN",
    )

    parser.add_argument(
        "--division-depth",
        type=int,
        help="How many divisions to make along heightmap side for composite aes metric.",
    )

    parser.add_argument(
        "--plot-title",
        type=str,
        required=True,
        help="The title to add.",
    )

    parser.add_argument(
        "--sub-title",
        type=str,
        required=True,
        help="The subtitle to add.",
    )

    args = parser.parse_args()

    classify_and_plot_heightmaps(
        [args.un_dir, args.fft_dir, args.rmd_dir, args.sdf_dir],
        ["UN", "FFT", "RMD", "SDF"],
        args.class_mode,
        args.plot_title,
        args.sub_title,
    )
