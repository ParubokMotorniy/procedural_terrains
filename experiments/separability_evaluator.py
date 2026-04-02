import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, roc_curve, auc
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

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
    splits=[[0, 1], [2, 3], [4, 5]],
    kmeans_k_list=[2],
    hdbscan_min_cluster_sizes=[10, 20, 50, 100, 150],
    hdbscan_min_samples=[5, 10, 20, 30],
):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    all_splits = splits + [list(range(X.shape[1]))]

    roc_data = {}

    for split in all_splits:
        X_sub = X_scaled[:, split]
        split_name = f"{split}" if len(split) < X.shape[1] else "FULL"

        print(f"\n--- Features: {split_name} ---")

        if split_name != "FULL":
            plt.figure(figsize=(7, 6))
            for label in np.unique(y):
                idx = y == label
                plt.scatter(
                    X_sub[idx, 0], X_sub[idx, 1], label=f"Class {label}", alpha=0.7
                )

            plt.title(f"Split {split_name}")
            plt.xlabel("d1")
            plt.ylabel("d2")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(f"split_{split_name}_scatter.png", dpi=300)
            plt.show()
            plt.cla()
            plt.close()

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
        # HDBSCAN (SEARCH)
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
        mean_fpr = np.linspace(0, 1, 200)

        for train_idx, test_idx in skf.split(X_sub, y):
            X_train, X_test = X_sub[train_idx], X_sub[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            svm_search = RandomizedSearchCV(
                SVC(probability=True),
                param_dist,
                n_iter=10,
                scoring="roc_auc",
                cv=3,
                n_jobs=8,
                verbose=0,
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

        mean_tpr = np.mean(tprs, axis=0)
        std_tpr = np.std(tprs, axis=0)
        mean_auc = np.mean(aucs)
        std_auc = np.std(aucs)

        print(f"SVM           | ROC-AUC: {mean_auc:.4f} ± {std_auc:.4f}")

        if split_name == "FULL":
            roc_data = {
                "mean_fpr": mean_fpr,
                "mean_tpr": mean_tpr,
                "std_tpr": std_tpr,
                "mean_auc": mean_auc,
                "std_auc": std_auc,
            }

    # ======================
    # ROC PLOT
    # ======================
    if roc_data:
        plt.figure(figsize=(7, 6))

        plt.plot(
            roc_data["mean_fpr"],
            roc_data["mean_tpr"],
            label=f"SVM (AUC = {roc_data['mean_auc']:.3f} ± {roc_data['std_auc']:.3f})",
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
        plt.title("ROC Curve (Full Features)")
        plt.legend(loc="lower right")
        plt.grid(True)

        plt.tight_layout()
        plt.savefig("roc_curve.png", dpi=300)
        plt.show()

    # ======================
    # PCA
    # ======================
    print("\n--- PCA (2D) ---")

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    plt.figure(figsize=(7, 6))

    for label in np.unique(y):
        idx = y == label
        plt.scatter(X_pca[idx, 0], X_pca[idx, 1], label=f"Class {label}", alpha=0.7)

    plt.title(
        f"PCA Projection (explained var={np.sum(pca.explained_variance_ratio_):.2f})"
    )
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig("pca_projection.png", dpi=300)
    plt.show()

    # ======================
    # UMAP
    # ======================
    print("\n--- UMAP (2D) ---")

    reducer = umap.UMAP(n_components=2, random_state=76340734)
    X_umap = reducer.fit_transform(X_scaled)

    plt.figure(figsize=(7, 6))

    for label in np.unique(y):
        idx = y == label
        plt.scatter(X_umap[idx, 0], X_umap[idx, 1], label=f"Class {label}", alpha=0.7)

    plt.title("UMAP Projection")
    plt.xlabel("UMAP-1")
    plt.ylabel("UMAP-2")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig("umap_projection.png", dpi=300)
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Evaluate metrics for the heightmaps.")

    parser.add_argument(
        "--vectors-interesting",
        type=str,
        required=True,
        help="The path to features of 'interesting' terrains",
    )

    parser.add_argument(
        "--vectors-boring",
        type=str,
        required=True,
        help="The path to features of 'boring' terrains",
    )

    args = parser.parse_args()

    interesting_vectors_pd = pd.read_csv(args.vectors_interesting)
    interesting_vectors = interesting_vectors_pd.to_numpy()[:, 1:]
    interesting_vectors = np.nan_to_num(interesting_vectors, nan=0.0)

    boring_vectors_pd = pd.read_csv(args.vectors_boring)
    boring_vectors = boring_vectors_pd.to_numpy()[:, 1:]
    boring_vectors = np.nan_to_num(boring_vectors, nan=0.0)

    all_features = np.vstack([boring_vectors, interesting_vectors])
    all_labels = np.array([0] * len(boring_vectors) + [1] * len(interesting_vectors))

    evaluate_separability(all_features, all_labels)


if __name__ == "__main__":
    main()
