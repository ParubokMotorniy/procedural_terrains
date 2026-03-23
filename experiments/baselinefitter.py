import evaluationlib as elib
import numpy as np
from PIL import Image
import argparse
import os
import tqdm
import pandas as pd

import numpy as np
import matplotlib.pyplot as plt

import joblib
from sklearn.metrics import roc_curve, auc

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
    n_splits: int = 5,
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
        svm = SVC(kernel="rbf", gamma="scale")
        svm.fit(X_train, y_train)
        svm_scores.append(accuracy_score(y_val, svm.predict(X_val)))

        # ---------------- MLP ----------------
        mlp = MLPClassifier(
            hidden_layer_sizes=(32, 16),
            max_iter=500,
            early_stopping=True,
            n_iter_no_change=10,
            random_state=42,
            alpha=0.05,
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
        plt.scatter(X_2d[:, 0], X_2d[:, 1], c=labels, cmap="coolwarm", alpha=0.7)
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


def train_and_save_models(
    class0, class1, feature_weights=None, model_prefix="model", n_splits=5
):
    X = np.vstack([class0, class1])
    y = np.array([0] * len(class0) + [1] * len(class1))

    if feature_weights is not None:
        X = X * feature_weights

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    joblib.dump(scaler, f"{model_prefix}_scaler.joblib")

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    all_y, all_svm_proba, all_mlp_proba = [], [], []

    for train_idx, val_idx in skf.split(X, y):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        svm = SVC(kernel="rbf", probability=True)
        svm.fit(X_train, y_train)
        svm_proba = svm.predict_proba(X_val)[:, 1]

        mlp = MLPClassifier(
            hidden_layer_sizes=(32, 16), max_iter=500, early_stopping=True, alpha=0.005
        )
        mlp.fit(X_train, y_train)
        mlp_proba = mlp.predict_proba(X_val)[:, 1]

        all_y.extend(y_val)
        all_svm_proba.extend(svm_proba)
        all_mlp_proba.extend(mlp_proba)

    # --- Train final models ---
    svm_final = SVC(kernel="rbf", probability=True)
    svm_final.fit(X, y)

    mlp_final = MLPClassifier(
        hidden_layer_sizes=(16, 8), max_iter=500, early_stopping=True
    )
    mlp_final.fit(X, y)

    joblib.dump(svm_final, f"{model_prefix}_svm.joblib")
    joblib.dump(mlp_final, f"{model_prefix}_mlp.joblib")

    # --- ROC ---
    fpr_svm, tpr_svm, _ = roc_curve(all_y, all_svm_proba)
    fpr_mlp, tpr_mlp, _ = roc_curve(all_y, all_mlp_proba)

    plt.figure()
    plt.plot(fpr_svm, tpr_svm, label="SVM")
    plt.plot(fpr_mlp, tpr_mlp, label="MLP")
    plt.title("ROC of individual classifiers")
    plt.legend()
    plt.savefig(f"{model_prefix}_roc.png")
    plt.close()


def train_subset_ensemble(
    class0, class1, subsets, subset_weights, family_weights, model_prefix="ensemble"
):
    X = np.vstack([class0, class1])
    y = np.array([0] * len(class0) + [1] * len(class1))

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    joblib.dump(scaler, f"{model_prefix}_scaler.joblib")

    svm_models = []
    mlp_models = []

    for i, subset in enumerate(subsets):
        X_sub = X[:, subset]

        svm = SVC(kernel="rbf", probability=True)
        svm.fit(X_sub, y)

        mlp = MLPClassifier(
            hidden_layer_sizes=(16, 8), max_iter=500, early_stopping=True, alpha=0.005
        )
        mlp.fit(X_sub, y)

        joblib.dump(svm, f"{model_prefix}_svm_{i}.joblib")
        joblib.dump(mlp, f"{model_prefix}_mlp_{i}.joblib")

        svm_models.append(svm)
        mlp_models.append(mlp)

    def combined_proba(X_input):
        svm_probs = np.zeros(len(X_input))
        mlp_probs = np.zeros(len(X_input))

        for i, subset in enumerate(subsets):
            w = subset_weights[i]

            svm_probs += w * svm_models[i].predict_proba(X_input[:, subset])[:, 1]
            mlp_probs += w * mlp_models[i].predict_proba(X_input[:, subset])[:, 1]

        svm_probs /= sum(subset_weights)
        mlp_probs /= sum(subset_weights)

        final = (
            family_weights["mlp"] * mlp_probs + family_weights["svm"] * svm_probs
        ) / (family_weights["mlp"] + family_weights["svm"])

        return final

    # ROC
    probs = combined_proba(X)
    fpr, tpr, _ = roc_curve(y, probs)

    plt.figure()
    plt.plot(fpr, tpr, label="Ensemble")
    plt.legend()
    plt.title("ROC of classifier ensemble")
    plt.savefig(f"{model_prefix}_roc.png")
    plt.close()


def classify_with_saved_models(X, model_prefix="model"):
    scaler = joblib.load(f"{model_prefix}_scaler.joblib")
    svm = joblib.load(f"{model_prefix}_svm.joblib")
    mlp = joblib.load(f"{model_prefix}_mlp.joblib")

    X = scaler.transform(X)

    svm_p = svm.predict_proba(X)[:, 1]
    mlp_p = mlp.predict_proba(X)[:, 1]

    df = pd.DataFrame({"svm_prob": svm_p, "mlp_prob": mlp_p})

    df.to_csv(f"{model_prefix}_predictions.csv", index=False)
    print(df)


def classify_with_ensemble(
    X, subsets, subset_weights, family_weights, model_prefix="ensemble"
):
    scaler = joblib.load(f"{model_prefix}_scaler.joblib")
    X = scaler.transform(X)

    svm_models = [
        joblib.load(f"{model_prefix}_svm_{i}.joblib") for i in range(len(subsets))
    ]
    mlp_models = [
        joblib.load(f"{model_prefix}_mlp_{i}.joblib") for i in range(len(subsets))
    ]

    svm_probs = np.zeros(len(X))
    mlp_probs = np.zeros(len(X))

    for i, subset in enumerate(subsets):
        w = subset_weights[i]

        svm_probs += w * svm_models[i].predict_proba(X[:, subset])[:, 1]
        mlp_probs += w * mlp_models[i].predict_proba(X[:, subset])[:, 1]

    svm_probs /= sum(subset_weights)
    mlp_probs /= sum(subset_weights)

    final = (family_weights["mlp"] * mlp_probs + family_weights["svm"] * svm_probs) / (
        family_weights["mlp"] + family_weights["svm"]
    )

    df = pd.DataFrame({"ensemble_prob": final})
    df.to_csv(f"{model_prefix}_predictions.csv", index=False)

    print(df)


def build_metric_vectors(directory: str, chunk_size: int, division_depth: int):
    if not os.path.isdir(directory):
        raise ValueError(f"{directory} is not a valid directory")

    files = [f for f in os.listdir(directory) if f.lower().strip().endswith('.jpg')]

    metric_vectors = []

    for filename in tqdm.tqdm(files):
        # if not filename.strip().endswith('.jpg'):
            # print(f"Skipping: {filename}")
            # continue
        path = os.path.join(directory, filename)
        heightmap = elib.read_jpg_grayscale(path)
        metric_vector = elib.get_metric_vector(
            heightmap, chunk_size, division_depth, True
        )

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

    parser.add_argument("--use-saved-vectors", action="store_true", default=False)

    args = parser.parse_args()

    directory_interesting = args.directory_interesting
    directory_boring = args.directory_boring

    if not args.use_saved_vectors:
        interesting_vectors = build_metric_vectors(
            directory_interesting, args.chunk_size, args.division_depth
        )
        interesting_vectors_pd = pd.DataFrame(interesting_vectors)
        interesting_vectors_pd.to_csv(
            os.path.join(directory_interesting, "interesting_metric_vectors.csv")
        )

        boring_vectors = build_metric_vectors(
            directory_boring, args.chunk_size, args.division_depth
        )
        boring_vectors_pd = pd.DataFrame(boring_vectors)
        boring_vectors_pd.to_csv(
            os.path.join(directory_boring, "boring_metric_vectors.csv")
        )
    else:
        interesting_vectors_pd = pd.read_csv(
            os.path.join(directory_interesting, "interesting_metric_vectors.csv")
        )
        interesting_vectors = interesting_vectors_pd.to_numpy()

        boring_vectors_pd = pd.read_csv(
            os.path.join(directory_boring, "boring_metric_vectors.csv")
        )
        boring_vectors = boring_vectors_pd.to_numpy()

    family_weights = {"mlp": 0.65, "svm": 0.35}
    feature_splits = [[0, 1], [2, 3], [4, 5]]
    splits_weights = [0.2, 0.4, 0.4]

    train_and_save_models(boring_vectors, interesting_vectors, None, "test_train", 6)
    train_subset_ensemble(
        boring_vectors,
        interesting_vectors,
        feature_splits,
        splits_weights,
        family_weights,
        "test_ensemble",
    )

    # evaluate_separability(interesting_vectors, boring_vectors)


if __name__ == "__main__":
    main()
