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


def train_data_ensemble_models(
    X, y, train_func, class_separation_idx, n_models=7, fraction=0.7, random_state=42
):
    rng = np.random.RandomState(random_state)
    models = []

    subset_size_1 = class_separation_idx
    n_samples_1 = int(subset_size_1 * fraction)

    subset_size_2 = len(y) - class_separation_idx
    n_samples_2 = int(subset_size_2 * fraction)

    # print(len(X), subset_size_1, n_samples_1)
    # print(len(X), subset_size_2, n_samples_2)

    for i in range(n_models):
        idx1 = rng.choice(subset_size_1, n_samples_1, replace=False)
        X_sub_1, y_sub_1 = X[idx1], y[idx1]

        idx2 = (
            rng.choice(subset_size_2, n_samples_2, replace=False) + class_separation_idx
        )
        X_sub_2, y_sub_2 = X[idx2], y[idx2]

        model = train_func(
            np.vstack([X_sub_1, X_sub_2]),
            np.vstack([y_sub_1.reshape(-1, 1), y_sub_2.reshape(-1, 1)]).ravel(),
        )
        models.append(model)

    return models


def train_and_save_models_auto(
    class0,
    class1,
    feature_weights=None,
    model_prefix="model",
    n_iter=100,
    use_data_ensemble=False,
    ensemble_fraction=0.7,
    n_ensemble_models=7,
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

    mlp_param_dist = {
        "hidden_layer_sizes": [(16, 8), (16, 16), (32, 8), (32, 16), (64, 8), (64, 16)],
        "alpha": np.logspace(-5, -2, 10),
        "learning_rate_init": np.logspace(-4, -2, 10),
        "beta_1": np.linspace(0.5, 0.999, 20),
        "beta_2": np.linspace(0.5, 0.999, 20),
    }

    def train_svm(X, y):
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
        return svm_search.best_estimator_

    def train_mlp(X, y):
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
        return mlp_search.best_estimator_

    if use_data_ensemble:
        svm_models = train_data_ensemble_models(
            X,
            y,
            train_svm,
            len(class0),
            n_models=n_ensemble_models,
            fraction=ensemble_fraction,
        )
        mlp_models = train_data_ensemble_models(
            X,
            y,
            train_mlp,
            len(class0),
            n_models=n_ensemble_models,
            fraction=ensemble_fraction,
        )

        joblib.dump(svm_models, f"{model_prefix}_svm_ensemble.joblib")
        joblib.dump(mlp_models, f"{model_prefix}_mlp_ensemble.joblib")

        def predict_proba_ensemble(models, X):
            probs = np.zeros(len(X))
            for m in models:
                probs += m.predict_proba(X)[:, 1]
            print(len(models), len(X), probs.shape)
            return probs / len(models)

        svm_probs = predict_proba_ensemble(svm_models, X)
        mlp_probs = predict_proba_ensemble(mlp_models, X)

    else:
        best_svm = train_svm(X, y)
        best_mlp = train_mlp(X, y)

        joblib.dump(best_svm, f"{model_prefix}_svm.joblib")
        joblib.dump(best_mlp, f"{model_prefix}_mlp.joblib")

        svm_probs = best_svm.predict_proba(X)[:, 1]
        mlp_probs = best_mlp.predict_proba(X)[:, 1]

    fpr_svm, tpr_svm, _ = roc_curve(y, svm_probs)
    fpr_mlp, tpr_mlp, _ = roc_curve(y, mlp_probs)

    plt.figure()
    plt.plot(fpr_svm, tpr_svm, label=f"Best SVM (AUC={auc(fpr_svm, tpr_svm):.3f})")
    plt.plot(fpr_mlp, tpr_mlp, label=f"Best MLP (AUC={auc(fpr_mlp, tpr_mlp):.3f})")
    plt.legend()
    plt.title("Best Model ROC")
    plt.savefig(f"{model_prefix}_best_roc_{'ens' if use_data_ensemble else 'sin'}.png")
    plt.close()


def train_subset_ensemble_auto(
    class0,
    class1,
    subsets=subsets_glob,
    subset_weights=subset_weights_glob,
    family_weights=family_weights_glob,
    model_prefix="ensemble",
    n_iter=100,
    use_data_ensemble=False,
    ensemble_fraction=0.7,
    n_ensemble_models=7,
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
        "beta_1": np.linspace(0.5, 0.999, 20),
        "beta_2": np.linspace(0.5, 0.999, 20),
    }

    def train_svm(X, y):
        return (
            RandomizedSearchCV(
                SVC(probability=True),
                svm_param_dist,
                n_iter=n_iter,
                scoring="roc_auc",
                cv=5,
                n_jobs=8,
                verbose=2,
            )
            .fit(X, y)
            .best_estimator_
        )

    def train_mlp(X, y):
        return (
            RandomizedSearchCV(
                MLPClassifier(max_iter=500, early_stopping=True),
                mlp_param_dist,
                n_iter=n_iter,
                scoring="roc_auc",
                cv=5,
                n_jobs=8,
                verbose=2,
            )
            .fit(X, y)
            .best_estimator_
        )

    for i, subset in enumerate(subsets):
        print(f"\n=== Training subset {i} ({subset}) ===")

        X_sub = X[:, subset]

        if use_data_ensemble:
            svm_models_sub = train_data_ensemble_models(
                X_sub,
                y,
                train_svm,
                len(class0),
                n_models=n_ensemble_models,
                fraction=ensemble_fraction,
            )
            mlp_models_sub = train_data_ensemble_models(
                X_sub,
                y,
                train_mlp,
                len(class0),
                n_models=n_ensemble_models,
                fraction=ensemble_fraction,
            )

            joblib.dump(svm_models_sub, f"{model_prefix}_svm_{i}_ensemble.joblib")
            joblib.dump(mlp_models_sub, f"{model_prefix}_mlp_{i}_ensemble.joblib")

            svm_models.append(svm_models_sub)
            mlp_models.append(mlp_models_sub)

        else:
            best_svm = train_svm(X_sub, y)
            best_mlp = train_mlp(X_sub, y)

            joblib.dump(best_svm, f"{model_prefix}_svm_{i}.joblib")
            joblib.dump(best_mlp, f"{model_prefix}_mlp_{i}.joblib")

            svm_models.append(best_svm)
            mlp_models.append(best_mlp)

    def combined_proba(X_input):
        def predict_family(models, X_sub):
            if isinstance(models, list):
                # data ensemble
                p = np.zeros(len(X_sub))
                for m in models:
                    p += m.predict_proba(X_sub)[:, 1]
                return p / len(models)
            else:
                return models.predict_proba(X_sub)[:, 1]

        svm_probs = np.zeros(len(X_input))
        mlp_probs = np.zeros(len(X_input))

        for i, subset in enumerate(subsets):
            w = subset_weights[i]

            svm_probs += w * predict_family(svm_models[i], X_input[:, subset])
            mlp_probs += w * predict_family(mlp_models[i], X_input[:, subset])

        svm_probs /= sum(subset_weights)
        mlp_probs /= sum(subset_weights)

        final = (
            family_weights["mlp"] * mlp_probs + family_weights["svm"] * svm_probs
        ) / (family_weights["mlp"] + family_weights["svm"])

        return final

    fpr, tpr, _ = roc_curve(y, combined_proba(X))
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, label=f"MLP+SVM Ensemble (AUC={roc_auc:.3f})")
    plt.legend()
    plt.savefig(f"{model_prefix}_cv_roc_{'ens' if use_data_ensemble else 'sin'}.png")
    plt.close()


def classify_with_saved_models(X, model_prefix="model", use_data_ensemble=False):
    if use_data_ensemble:
        svm_models = joblib.load(f"{model_prefix}_svm_ensemble.joblib")
        mlp_models = joblib.load(f"{model_prefix}_mlp_ensemble.joblib")

        def predict(models):
            p = np.zeros(len(X))
            for m in models:
                p += m.predict_proba(X)[:, 1]
            return p / len(models)

        svm_p = predict(svm_models)
        mlp_p = predict(mlp_models)

    else:
        svm = joblib.load(f"{model_prefix}_svm.joblib")
        mlp = joblib.load(f"{model_prefix}_mlp.joblib")

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
    use_data_ensemble=False,
):
    scaler = joblib.load(f"{model_prefix}_scaler.joblib")
    X = scaler.transform(X)

    svm_models = []
    mlp_models = []

    for i in range(len(subsets)):
        if use_data_ensemble:
            svm_models.append(joblib.load(f"{model_prefix}_svm_{i}_ensemble.joblib"))
            mlp_models.append(joblib.load(f"{model_prefix}_mlp_{i}_ensemble.joblib"))
        else:
            svm_models.append(joblib.load(f"{model_prefix}_svm_{i}.joblib"))
            mlp_models.append(joblib.load(f"{model_prefix}_mlp_{i}.joblib"))

    def predict_family(models, X_sub):
        if isinstance(models, list):
            probs = np.zeros(len(X_sub))
            for m in models:
                probs += m.predict_proba(X_sub)[:, 1]
            return probs / len(models)
        else:
            return models.predict_proba(X_sub)[:, 1]

    svm_probs = np.zeros(len(X))
    mlp_probs = np.zeros(len(X))

    for i, subset in enumerate(subsets):
        w = subset_weights[i]

        X_sub = X[:, subset]

        svm_probs += w * predict_family(svm_models[i], X_sub)
        mlp_probs += w * predict_family(mlp_models[i], X_sub)

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

    parser.add_argument("--mode", type=str, help="<train|classify>", required=True)

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

    parser.add_argument(
        "--train-ensemble",
        action="store_true",
        default=False,
        help="If train ensembles of models on different subsets of data.",
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
                os.path.join(
                    directory_interesting,
                    "interesting_metric_vectors_thrsh_05_entrp.csv",
                )
            )
            interesting_vectors = interesting_vectors_pd.to_numpy()[:, 1:]

            boring_vectors_pd = pd.read_csv(
                os.path.join(
                    directory_boring, "boring_metric_vectors_thrsh_05_entrp.csv"
                )
            )
            boring_vectors = boring_vectors_pd.to_numpy()[:, 1:]

        train_and_save_models_auto(
            boring_vectors,
            interesting_vectors,
            None,
            "test_model",
            20,
            use_data_ensemble=args.train_ensemble,
        )
        train_subset_ensemble_auto(
            boring_vectors,
            interesting_vectors,
            use_data_ensemble=args.train_ensemble,
            model_prefix="test_ensemble",
        )
    elif args.mode == "classify":
        vectors_to_classify = build_metric_vectors(
            args.directory_classify, args.chunk_size, args.division_depth, fmt
        )
        classify_with_saved_models(
            vectors_to_classify,
            model_prefix="test_model",
            use_data_ensemble=args.train_ensemble,
        )
        classify_with_ensemble(
            vectors_to_classify,
            model_prefix="test_ensemble",
            use_data_ensemble=args.train_ensemble,
        )
    else:
        raise ValueError("Wrong script mode")

    # evaluate_separability(interesting_vectors, boring_vectors)


if __name__ == "__main__":
    main()
