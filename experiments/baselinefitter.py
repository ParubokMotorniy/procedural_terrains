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
from sklearn.model_selection import RandomizedSearchCV

family_weights_glob = {"mlp": 0.6, "svm": 0.4}
subsets_glob = [[0, 1], [2, 3], [4, 5]]
subset_weights_glob = [0.15, 0.35, 0.5]

def train_and_save_models_auto(
    class0, class1, feature_weights=None, model_prefix="model", n_iter=100
):
    X = np.vstack([class0, class1])
    y = np.array([0] * len(class0) + [1] * len(class1))
    
    print(f"Total vectors: {len(X)}")

    if feature_weights is not None:
        X = X * feature_weights

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    joblib.dump(scaler, f"{model_prefix}_scaler.joblib")

    svm_param_dist = {
        "C": np.logspace(-3, 3, 40),
        "gamma": ["scale", "auto"] + list(np.logspace(-3, 1, 10)),
        "kernel": ["rbf"],
    }

    svm_search = RandomizedSearchCV(
        SVC(probability=True),
        svm_param_dist,
        n_iter=n_iter,
        scoring="roc_auc",
        cv=5,
        verbose=2,
        n_jobs=8,
    )

    svm_search.fit(X, y)
    best_svm = svm_search.best_estimator_

    print("Best SVM params:", svm_search.best_params_)

    mlp_param_dist = {
        "hidden_layer_sizes": [(16, 8), (16, 16), (32, 8), (32, 16), (64, 8), (64, 16)],
        "alpha": np.logspace(-5, -2, 10),
        "learning_rate_init": np.logspace(-4, -2, 10),
        "beta_1": np.linspace(0.5, 1.0, 20),
        "beta_2": np.linspace(0.5, 1.0, 20),
    }

    mlp_search = RandomizedSearchCV(
        MLPClassifier(max_iter=500, early_stopping=True),
        mlp_param_dist,
        n_iter=n_iter,
        scoring="roc_auc",
        cv=5,
        verbose=2,
        n_jobs=8,
    )

    mlp_search.fit(X, y)
    best_mlp = mlp_search.best_estimator_

    print("Best MLP params:", mlp_search.best_params_)

    joblib.dump(best_svm, f"{model_prefix}_svm.joblib")
    joblib.dump(best_mlp, f"{model_prefix}_mlp.joblib")

    svm_probs = best_svm.predict_proba(X)[:, 1]
    mlp_probs = best_mlp.predict_proba(X)[:, 1]

    fpr_svm, tpr_svm, _ = roc_curve(y, svm_probs)
    fpr_mlp, tpr_mlp, _ = roc_curve(y, mlp_probs)

    plt.figure()
    plt.plot(fpr_svm, tpr_svm, label="Best SVM")
    plt.plot(fpr_mlp, tpr_mlp, label="Best MLP")
    plt.legend()
    plt.title("Best Model ROC")
    plt.savefig(f"{model_prefix}_best_roc.png")
    plt.close()


def train_subset_ensemble_auto(
    class0,
    class1,
    subsets=subsets_glob,
    subset_weights=subset_weights_glob,
    family_weights=family_weights_glob,
    model_prefix="ensemble",
    n_iter=100,
):
    X = np.vstack([class0, class1])
    y = np.array([0] * len(class0) + [1] * len(class1))

    print(f"Total vectors: {len(X)}")

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    joblib.dump(scaler, f"{model_prefix}_scaler.joblib")

    svm_models = []
    mlp_models = []

    svm_param_dist = {
        "C": np.logspace(-2, 2, 20),
        "gamma": ["scale", "auto"] + list(np.logspace(-3, 1, 10)),
        "kernel": ["rbf"],
    }

    mlp_param_dist = {
        "hidden_layer_sizes": [(16,), (32,), (32, 16), (64, 32)],
        "alpha": np.logspace(-5, -2, 10),
        "learning_rate_init": np.logspace(-4, -2, 10),
        "beta_1": np.linspace(0.5, 1.0, 20),
        "beta_2": np.linspace(0.5, 1.0, 20),
    }

    for i, subset in enumerate(subsets):
        print(f"\n=== Training subset {i} ({subset}) ===")

        X_sub = X[:, subset]

        # -------- SVM search --------
        svm_search = RandomizedSearchCV(
            SVC(probability=True),
            svm_param_dist,
            n_iter=n_iter,
            scoring="roc_auc",
            cv=5,
            n_jobs=8,
            verbose=1,
        )

        svm_search.fit(X_sub, y)
        best_svm = svm_search.best_estimator_

        print(f"Subset {i} best SVM:", svm_search.best_params_)

        # -------- MLP search --------
        mlp_search = RandomizedSearchCV(
            MLPClassifier(max_iter=500, early_stopping=True),
            mlp_param_dist,
            n_iter=n_iter,
            scoring="roc_auc",
            cv=5,
            n_jobs=8,
            verbose=1,
        )

        mlp_search.fit(X_sub, y)
        best_mlp = mlp_search.best_estimator_

        print(f"Subset {i} best MLP:", mlp_search.best_params_)

        # Save
        joblib.dump(best_svm, f"{model_prefix}_svm_{i}.joblib")
        joblib.dump(best_mlp, f"{model_prefix}_mlp_{i}.joblib")

        svm_models.append(best_svm)
        mlp_models.append(best_mlp)
        
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

    probs = combined_proba(X)

    fpr, tpr, _ = roc_curve(y, probs)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, label=f"Ensemble (AUC={roc_auc:.3f})")
    plt.legend()
    plt.title("Optimized Ensemble ROC")
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
    X,
    subsets=subsets_glob,
    subset_weights=subset_weights_glob,
    family_weights=family_weights_glob,
    model_prefix="ensemble",
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


def build_metric_vectors(
    directory: str, chunk_size: int, division_depth: int, fmt: str
):
    if not os.path.isdir(directory):
        raise ValueError(f"{directory} is not a valid directory")

    extensions = {
        "exr": ([".exr"], elib.read_exr_grayscale, 1.0),
        "jpg": ([".jpg", ".jpeg"], elib.read_jpg_grayscale, 255.0),
        "jpeg": ([".jpg", ".jpeg"], elib.read_jpg_grayscale, 255.0),
        # "tif": ([".tif", ".tiff"], elib.read_tiff_grayscale),
        # "tiff": ([".tif", ".tiff"], elib.read_tiff_grayscale),
    }

    if fmt not in extensions:
        raise ValueError("Unsupported format")

    possible_extensions, file_reader, population_max = extensions[fmt]

    files = sorted(
        f
        for f in os.listdir(directory)
        if any(f.lower().strip().endswith(ext) for ext in possible_extensions)
    )

    print(f"Total heightmaps to evaluate: {len(files)}")

    metric_vectors = []

    for filename in tqdm.tqdm(files):
        path = os.path.join(directory, filename)
        heightmap = file_reader(path)
        print(f"Processing heightmap: {filename}")
        metric_vector = elib.get_metric_vector(
            heightmap, chunk_size, division_depth, population_max, False
        )

        metric_vectors.append(metric_vector)

        del heightmap

    return np.array(metric_vectors)


def main():
    parser = argparse.ArgumentParser(description="Evaluate metrics for the heightmaps.")

    parser.add_argument(
        "--mode",
        type=str,
        help="<train|classify>",
        required=True
    )

    parser.add_argument(
        "--format",
        type=str,
        default="exr",
        help="Heightmap format (exr, jpg, jpeg, tif, tiff)",
    )

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
        "--directory-classify",
        type=str,
        help="Directory containing heightmaps to classify",
    )

    parser.add_argument(
        "--chunk-size",
        type=int,
        help="Should such a need arise, the heightmap will be split into chunks of size NxN",
    )

    parser.add_argument(
        "--division-depth",
        type=int,
        help="The depth of the order-analyzing tree",
    )

    parser.add_argument(
        "--use-saved",
        action="store_true",
        default=False,
        help="If use previously stored vectors for training.",
    )

    args = parser.parse_args()

    fmt = args.format.lower().strip()

    directory_interesting = args.directory_interesting
    directory_boring = args.directory_boring

    if args.mode == "train":
        if not args.use_saved:
            print(f"Building vectors anew!")
            interesting_vectors = build_metric_vectors(
                directory_interesting, args.chunk_size, args.division_depth, fmt
            )
            interesting_vectors_pd = pd.DataFrame(interesting_vectors)
            interesting_vectors_pd.to_csv(
                os.path.join(
                    directory_interesting, "interesting_metric_vectors_new.csv"
                )
            )

            boring_vectors = build_metric_vectors(
                directory_boring, args.chunk_size, args.division_depth, fmt
            )
            boring_vectors_pd = pd.DataFrame(boring_vectors)
            boring_vectors_pd.to_csv(
                os.path.join(directory_boring, "boring_metric_vectors_new.csv")
            )
        else:
            print(f"Loading the stored vectors!")
            interesting_vectors_pd = pd.read_csv(
                os.path.join(directory_interesting, "interesting_metric_vectors.csv")
            )
            interesting_vectors = interesting_vectors_pd.to_numpy()[:,1:]

            boring_vectors_pd = pd.read_csv(
                os.path.join(directory_boring, "boring_metric_vectors.csv")
            )
            boring_vectors = boring_vectors_pd.to_numpy()[:,1:]

        train_and_save_models_auto(
            boring_vectors, interesting_vectors, None, "test_model", 20
        )
        train_subset_ensemble_auto(
            boring_vectors, interesting_vectors, model_prefix="test_ensemble"
        )
    elif args.mode == "classify":
        vectors_to_classify = build_metric_vectors(
            args.directory_classify, args.chunk_size, args.division_depth, fmt
        )
        classify_with_saved_models(vectors_to_classify, "test_model")
        classify_with_ensemble(vectors_to_classify, model_prefix="test_ensemble")
    else:
        raise ValueError("Wrong script mode")

    # evaluate_separability(interesting_vectors, boring_vectors)


if __name__ == "__main__":
    main()
