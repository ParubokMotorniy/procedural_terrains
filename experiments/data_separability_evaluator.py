import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, roc_curve, auc, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

import argparse
import pandas as pd
import os

import umap
import hdbscan


def _best_cluster_accuracy(y_true, y_pred):
    acc1 = accuracy_score(y_true, y_pred)
    acc2 = accuracy_score(y_true, 1 - y_pred)
    return max(acc1, acc2)


def evaluate_separability(
    X,
    y,
    classes_names=["boring", "interesting"],
    splits=[
        [0, 1],
        [2, 3],
        [4, 5],
    ],
    splits_names=[
        "geometric",
        "fractal",
        "aesthetical",
    ],
    legend_pos=[(0.27, 0.05), (0.05, 0.05), (0.25, 0.7)],
    axes_names=[
        (r"$m_\text{erosion}$", r"$m_\text{gradient}$"),
        (r"$D$", r"$\beta$"),
        (r"$m_\text{global}$", r"$m_\text{composite}$"),
    ],
    kmeans_k_list=[2],
    hdbscan_min_cluster_sizes=[20, 50, 100, 150],
    hdbscan_min_samples=[3, 5, 10, 20, 30],
):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    all_splits = splits + [list(range(X.shape[1]))]
    all_axes_names = axes_names + [("", "")]

    roc_data = {}

    for split, split_name, (ax1_name, ax2_name), plot_legend_pos in zip(
        all_splits, splits_names + ["FULL"], all_axes_names, legend_pos + [(None, None)]
    ):
        X_sub = X_scaled[:, split]

        print(f"\n--- Features: {split_name} ---")

        # ======================
        # KMEANS
        # ======================
        best_km_acc = 0
        best_k = None

        for k in kmeans_k_list:
            kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
            pred = kmeans.fit_predict(X_sub)

            acc = _best_cluster_accuracy(y, pred)

            if acc > best_km_acc:
                best_km_acc = acc
                best_k = k

        print(f"KMeans        | Best k={best_k} | Accuracy: {best_km_acc:.4f}")

        # ======================
        # HDBSCAN
        # ======================
        best_hdb_acc = 0
        best_hdb_out = 1.0
        best_params = None

        for mcs in hdbscan_min_cluster_sizes:
            for ms in hdbscan_min_samples:
                clusterer = hdbscan.HDBSCAN(min_cluster_size=mcs, min_samples=ms)
                pred = clusterer.fit_predict(X_sub)

                mask = pred != -1

                if np.sum(mask) == 0:
                    continue

                acc = _best_cluster_accuracy(y[mask], pred[mask])
                out_ratio = np.mean(~mask)

                score = acc * (1 - out_ratio)

                if score > best_hdb_acc:
                    best_hdb_acc = score
                    best_hdb_out = out_ratio
                    best_params = (mcs, ms)

        print(
            f"HDBSCAN       | Acc*: {best_hdb_acc:.4f} | Outliers: {best_hdb_out:.3f} | params={best_params}"
        )

        # ======================
        # SVM + ROC
        # ======================
        param_dist = {
            "C": np.logspace(-2, 2, 20),
            "gamma": ["scale", "auto"] + list(np.logspace(-3, 1, 10)),
            "kernel": ["rbf"],
        }

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=67586)

        tprs = []
        aucs = []
        svm_f1 = []
        mean_fpr = np.linspace(0, 1, 200)

        qda_aucs = []
        qda_f1 = []

        for train_idx, test_idx in skf.split(X_sub, y):
            X_train, X_test = X_sub[train_idx], X_sub[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            # SVM
            svm_search = RandomizedSearchCV(
                SVC(probability=True),
                param_dist,
                n_iter=10,
                scoring="roc_auc",
                cv=3,
                n_jobs=8,
                verbose=2,
                random_state=456,
            )

            svm_search.fit(X_train, y_train)
            model = svm_search.best_estimator_

            probs = model.predict_proba(X_test)[:, 1]

            fpr, tpr, _ = roc_curve(y_test, probs)
            roc_auc = auc(fpr, tpr)

            aucs.append(roc_auc)

            interp_tpr = np.interp(mean_fpr, fpr, tpr)

            interp_tpr[0] = 0.0
            tprs.append(interp_tpr)
            svm_f1.append(
                (
                    # boring f1
                    f1_score(y_test, (probs > 0.5).astype(int), pos_label=0),
                    # interesting f1
                    f1_score(y_test, (probs > 0.5).astype(int), pos_label=1),
                )
            )

            # QDA
            qda = QuadraticDiscriminantAnalysis(solver="eigen")
            qda.fit(X_train, y_train)

            qda_probs = qda.predict_proba(X_test)[:, 1]

            fpr, tpr, _ = roc_curve(y_test, qda_probs)
            roc_auc = auc(fpr, tpr)

            qda_aucs.append(roc_auc)
            qda_f1.append(
                (
                    # boring f1
                    f1_score(y_test, (qda_probs > 0.5).astype(int), pos_label=0),
                    # interesting f1
                    f1_score(y_test, (qda_probs > 0.5).astype(int), pos_label=1),
                )
            )

        mean_tpr = np.mean(tprs, axis=0)
        std_tpr = np.std(tprs, axis=0)
        mean_auc = np.mean(aucs)
        std_auc = np.std(aucs)
        mean_f1 = np.sum([f1_1 + f1_2 for f1_1, f1_2 in svm_f1]) / (2.0 * len(svm_f1))

        print(f"SVM           | CV ROC-AUC: {mean_auc:.4f} ± {std_auc:.4f}")

        mean_qda_auc = np.mean(qda_aucs)
        std_qda_auc = np.std(qda_aucs)
        mean_qda_f1 = np.sum([f1_1 + f1_2 for f1_1, f1_2 in qda_f1]) / (
            2.0 * len(qda_f1)
        )

        print(f"QDA           | CV ROC-AUC: {mean_qda_auc:.4f} ± {std_qda_auc:.4f}")

        if split_name != "FULL":
            plt.figure(figsize=(7, 6))
            for label in np.unique(y):
                idx = y == label
                plt.scatter(
                    X_sub[idx, 0],
                    X_sub[idx, 1],
                    label=f"Terrain class: {classes_names[label]}",
                    alpha=0.5,
                )

            plt.title(
                f"Terrain heightmap data separability\nMetric family: {split_name}"
            )
            plt.xlabel(ax1_name)
            text = (
                "Clusterization:\n"
                f"{'KMeans':8} | {'Accuracy':8} : {best_km_acc:.4f}\n"
                f"{'HDBSCAN':8} | {'Acc':8} : {best_hdb_acc:.4f} | Out: {best_hdb_out:.3f}\n"
                f"{'SVM':8} | {'ROC-AUC':8} : {mean_auc:.4f} ± {std_auc:.4f} | $\overline{{F1}}$ : {mean_f1:.2f}\n"
                f"{'QDA':8} | {'ROC-AUC':8} : {mean_qda_auc:.4f} ± {std_qda_auc:.4f} | $\overline{{F1}}$ : {mean_qda_f1:.2f}"
            )
            plt.annotate(
                text,
                plot_legend_pos,
                xycoords="axes fraction",
                bbox=dict(
                    boxstyle="round",
                    facecolor="white",
                    alpha=0.6,
                    edgecolor="gray",
                ),
                fontfamily="monospace",
            )
            plt.ylabel(ax2_name)
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(f"split_{split_name}_scatter.png", dpi=300)
            plt.show()
            plt.cla()
            plt.close()
        else:
            roc_data = {
                "mean_fpr": mean_fpr,
                "mean_tpr": mean_tpr,
                "std_tpr": std_tpr,
                "mean_auc": mean_auc,
                "std_auc": std_auc,
                "avg_f1": mean_f1,
            }

            # ======================
            # PCA
            # ======================
            print("\n--- PCA (2D) ---")

            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(X_scaled)

            plt.figure(figsize=(7, 6))

            for label in np.unique(y):
                idx = y == label
                plt.scatter(
                    X_pca[idx, 0],
                    X_pca[idx, 1],
                    label=f"Terrain class: {classes_names[label]}",
                    alpha=0.5,
                )

            plt.title(
                f"PCA Projection (explained var={np.sum(pca.explained_variance_ratio_):.2f})"
            )
            plt.xlabel("PC1")
            plt.ylabel("PC2")
            text = (
                "Clusterization:\n"
                f"{'KMeans':8} | {'Accuracy':8} : {best_km_acc:.4f}\n"
                f"{'HDBSCAN':8} | {'Acc':8} : {best_hdb_acc:.4f} | Out: {best_hdb_out:.3f}\n"
                f"{'SVM':8} | {'ROC-AUC':8} : {mean_auc:.4f} ± {std_auc:.4f} | $\overline{{F1}}$ : {mean_f1:.2f}\n"
                f"{'QDA':8} | {'ROC-AUC':8} : {mean_qda_auc:.4f} ± {std_qda_auc:.4f} | $\overline{{F1}}$ : {mean_qda_f1:.2f}"
            )
            plt.annotate(
                text,
                (0.05, 0.7),
                xycoords="axes fraction",
                bbox=dict(
                    boxstyle="round",
                    facecolor="white",
                    alpha=0.6,
                    edgecolor="gray",
                ),
                fontfamily="monospace",
            )
            plt.legend()
            plt.grid(True)

            plt.tight_layout()
            plt.savefig("pca_projection.png", dpi=300)
            plt.show()
            plt.cla()
            plt.close()

            # ======================
            # UMAP
            # ======================
            print("\n--- UMAP (2D) ---")

            reducer = umap.UMAP(n_components=2, random_state=76340734)
            X_umap = reducer.fit_transform(X_scaled)

            plt.figure(figsize=(7, 6))

            for label in np.unique(y):
                idx = y == label
                plt.scatter(
                    X_umap[idx, 0],
                    X_umap[idx, 1],
                    label=f"Terrain class: {classes_names[label]}",
                    alpha=0.5,
                )

            plt.title("UMAP Projection")
            plt.xlabel("UMAP-1")
            plt.ylabel("UMAP-2")
            text = (
                "Clusterization:\n"
                f"{'KMeans':8} | {'Accuracy':8} : {best_km_acc:.4f}\n"
                f"{'HDBSCAN':8} | {'Acc':8} : {best_hdb_acc:.4f} | Out: {best_hdb_out:.3f}\n"
                f"{'SVM':8} | {'ROC-AUC':8} : {mean_auc:.4f} ± {std_auc:.4f} | $\overline{{F1}}$ : {mean_f1:.2f}\n"
                f"{'QDA':8} | {'ROC-AUC':8} : {mean_qda_auc:.4f} ± {std_qda_auc:.4f} | $\overline{{F1}}$ : {mean_qda_f1:.2f}"
            )
            plt.annotate(
                text,
                (0.05, 0.05),
                xycoords="axes fraction",
                bbox=dict(
                    boxstyle="round",
                    facecolor="white",
                    alpha=0.6,
                    edgecolor="gray",
                ),
                fontfamily="monospace",
            )
            plt.legend()
            plt.grid(True)

            plt.tight_layout()
            plt.savefig("umap_projection.png", dpi=300)
            plt.show()
            plt.cla()
            plt.close()

    # ======================
    # ROC PLOT
    # ======================
    plt.figure(figsize=(7, 6))

    plt.plot(
        roc_data["mean_fpr"],
        roc_data["mean_tpr"],
        label=f"SVM (AUC = {roc_data['mean_auc']:.3f} ± {roc_data['std_auc']:.3f}, $\overline{{F1}}$ = {roc_data["avg_f1"]:.2f})",
        linewidth=2,
    )

    plt.fill_between(
        roc_data["mean_fpr"],
        np.maximum(roc_data["mean_tpr"] - roc_data["std_tpr"], 0),
        np.minimum(roc_data["mean_tpr"] + roc_data["std_tpr"], 1),
        alpha=0.2,
    )

    plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1)

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("CV ROC Curve (All metrics)")
    plt.legend(loc="lower right")
    plt.grid(True)

    plt.tight_layout()
    plt.savefig("roc_curve.png", dpi=300)
    plt.show()
    plt.cla()
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Evaluate metrics for the heightmaps.")

    parser.add_argument(
        "--vectors-interesting",
        type=str,
        required=True,
        help="The path to csv with metric vectors of 'interesting' terrains",
    )

    parser.add_argument(
        "--vectors-boring",
        type=str,
        required=True,
        help="The path to csv with metric vectors of 'boring' terrains",
    )

    args = parser.parse_args()

    interesting_vectors_pd = pd.read_csv(args.vectors_interesting)
    interesting_vectors = interesting_vectors_pd.to_numpy()[:, 1:7]
    interesting_vectors = interesting_vectors[interesting_vectors[:, 2] != 2.0]

    boring_vectors_pd = pd.read_csv(args.vectors_boring)
    boring_vectors = boring_vectors_pd.to_numpy()[:, 1:7]
    # boring_vectors = boring_vectors[boring_vectors[:, 2] != 2.0]

    all_features = np.vstack([boring_vectors, interesting_vectors])
    all_labels = np.array([0] * len(boring_vectors) + [1] * len(interesting_vectors))

    evaluate_separability(all_features, all_labels)


if __name__ == "__main__":
    main()
