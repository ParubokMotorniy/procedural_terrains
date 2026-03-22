import evaluationlib as elib
import numpy as np
from PIL import Image
import argparse
import os
import tqdm
import pandas as pd

import numpy as np
import matplotlib.pyplot as plt

from sklearn.svm import SVC, LinearSVC
from sklearn.cluster import KMeans
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance

def evaluate_separability(
    class0: np.ndarray,
    class1: np.ndarray,
    feature_weights: np.ndarray | None = None,
    n_splits: int = 5
):
    # ============================================================
    # Data prep
    # ============================================================
    X = np.vstack([class0, class1])
    y = np.array([0] * len(class0) + [1] * len(class1))

    # Apply manual feature weights if provided
    if feature_weights is not None:
        X = X * feature_weights  # element-wise scaling

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    print(f"Dataset: {X.shape}")

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    svm_scores = []
    mlp_scores = []
    kmeans_scores = []

    # ============================================================
    # Cross-validation loop
    # ============================================================
    for train_idx, val_idx in tqdm.tqdm(skf.split(X, y)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # ---------------- SVM ----------------
        svm = SVC(kernel='rbf', gamma='scale')
        svm.fit(X_train, y_train)
        svm_scores.append(accuracy_score(y_val, svm.predict(X_val)))

        # ---------------- MLP ----------------
        mlp = MLPClassifier(
            hidden_layer_sizes=(32, 16),
            max_iter=500,
            early_stopping=True,
            n_iter_no_change=10,
            random_state=42,
            alpha=0.05
        )
        mlp.fit(X_train, y_train)
        mlp_scores.append(accuracy_score(y_val, mlp.predict(X_val)))

        # ---------------- KMeans ----------------
        kmeans = KMeans(n_clusters=2, n_init=10, random_state=42)
        labels = kmeans.fit_predict(X_val)

        acc1 = accuracy_score(y_val, labels)
        acc2 = accuracy_score(y_val, 1 - labels)
        kmeans_scores.append(max(acc1, acc2))

    print("\n=== Cross-Validation Results ===")
    print(f"SVM Accuracy:   {np.mean(svm_scores):.3f} ± {np.std(svm_scores):.3f}")
    print(f"MLP Accuracy:   {np.mean(mlp_scores):.3f} ± {np.std(mlp_scores):.3f}")
    print(f"KMeans Accuracy:{np.mean(kmeans_scores):.3f} ± {np.std(kmeans_scores):.3f}")

    # ============================================================
    # Feature Importance (on full dataset)
    # ============================================================
    # print("\n=== Feature Importance ===")

    # # --- Linear SVM for interpretability ---
    # lin_svm = LinearSVC()
    # lin_svm.fit(X, y)

    # importance_linear = np.abs(lin_svm.coef_[0])

    # print("\nLinear SVM importance:")
    # print(importance_linear)

    # # --- Permutation importance ---
    # perm = permutation_importance(
    #     svm, X, y, n_repeats=10, random_state=42
    # )

    # print("\nPermutation importance:")
    # print(perm.importances_mean)

    # ============================================================
    # Visualization
    # ============================================================
    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X)

    def plot(title, labels):
        plt.figure()
        plt.title(title)
        plt.scatter(X_2d[:, 0], X_2d[:, 1], c=labels, cmap='coolwarm', alpha=0.7)
        plt.colorbar()
        plt.show()

    plot("Ground Truth", y)

    # SVM decision
    svm.fit(X, y)
    plot("SVM", svm.predict(X))

    # KMeans
    kmeans = KMeans(n_clusters=2, n_init=10, random_state=42)
    plot("KMeans", kmeans.fit_predict(X))

    return {
        "svm_mean": np.mean(svm_scores),
        "mlp_mean": np.mean(mlp_scores),
        "kmeans_mean": np.mean(kmeans_scores),
        # "feature_importance_linear": importance_linear,
        # "feature_importance_permutation": perm.importances_mean
    }


def build_metric_vectors(directory: str, chunk_size: int, division_depth: int):
    if not os.path.isdir(directory):
        raise ValueError(f"{directory} is not a valid directory")
    
    files = os.listdir(directory)[:10]

    metric_vectors = []

    for filename in tqdm.tqdm(files):
        path = os.path.join(directory, filename)
        heightmap = elib.read_jpg_grayscale(path)
        metric_vector = elib.get_metric_vector(heightmap, chunk_size, division_depth, True)

        metric_vectors.append(metric_vector)

        del heightmap

    return np.array(metric_vectors)


def main():
    parser = argparse.ArgumentParser(description="Evaluate metrics for the heightmaps.")

    parser.add_argument(
        "--directory-interesting",
        type=str,
        help="Directory containing interesting heightmaps",
    )

    parser.add_argument(
        "--directory-boring",
        type=str,
        help="Directory containing boring heightmaps",
    )

    parser.add_argument(
        "--chunk-size",
        type=int,
        required=True,
        help="Should such a need arise, the heightmap will be split into chunks of size NxN",
    )

    parser.add_argument(
        "--division-depth",
        type=int,
        required=True,
        help="The depth of the order-analyzing tree",
    )

    args = parser.parse_args()

    directory_interesting = args.directory_interesting
    directory_boring = args.directory_boring

    interesting_vectors = build_metric_vectors(directory_interesting, args.chunk_size, args.division_depth)
    interesting_vectors_pd = pd.DataFrame(interesting_vectors)
    interesting_vectors_pd.to_csv(os.path.join(directory_interesting, "metric_vectors.csv"))
    
    boring_vectors = build_metric_vectors(directory_boring, args.chunk_size, args.division_depth)
    boring_vectors_pd = pd.DataFrame(boring_vectors)
    boring_vectors_pd.to_csv(os.path.join(directory_boring, "metric_vectors.csv"))

    # evaluate_separability(interesting_vectors, boring_vectors)


if __name__ == "__main__":
    main()
