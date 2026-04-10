import evaluationlib as elib
import numpy as np
from PIL import Image
import argparse
import os
import tqdm
import pandas as pd

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import joblib
from sklearn.metrics import roc_curve, auc

from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

family_weights_glob = {"mlp": 0.45, "svm": 0.55}
subsets_glob = [[0, 1], [2, 3], [4, 5]]
subset_weights_glob = [0.35, 0.35, 0.3]

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


def cross_validated_roc(model, X, y, n_splits=5):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=3673655)
    scaler = StandardScaler()

    mean_fpr = np.linspace(0, 1, 200)
    tprs = []
    aucs = []

    for train_idx, test_idx in skf.split(X, y):
        X_train = scaler.fit_transform(X[train_idx])
        X_test = scaler.transform(X[test_idx])
        y_train, y_test = y[train_idx], y[test_idx]

        model_clone = model.__class__(**model.get_params())
        model_clone.fit(X_train, y_train)

        probs = model_clone.predict_proba(X_test)[:, 1]

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

    return mean_fpr, mean_tpr, std_tpr, mean_auc, std_auc


def train_data_ensemble_models(
    X,
    y,
    train_func,
    class_separation_idx,
    n_models=5,
    fraction=0.75,
    random_state=57835,
):
    rng = np.random.RandomState(random_state)
    models = []

    subset_size_1 = class_separation_idx
    n_samples_1 = int(subset_size_1 * fraction)

    subset_size_2 = len(y) - class_separation_idx
    n_samples_2 = int(subset_size_2 * fraction)

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
    ensemble_fraction=0.75,
    n_ensemble_models=5,
    components_to_try=[2, 3, 4],
):
    X = np.vstack([class0, class1])
    y = np.array([0] * len(class0) + [1] * len(class1))

    print(f"Total vectors: {len(X)}")

    if feature_weights is not None:
        X = X * feature_weights

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    joblib.dump(scaler, f"{model_prefix}_scaler.joblib")

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

    def do_fit(
        actual_x: np.ndarray, y: np.ndarray, differentiation_tag: str, subtitle: str
    ):
        if use_data_ensemble:
            svm_models = train_data_ensemble_models(
                actual_x,
                y,
                train_svm,
                len(class0),
                n_models=n_ensemble_models,
                fraction=ensemble_fraction,
            )
            mlp_models = train_data_ensemble_models(
                actual_x,
                y,
                train_mlp,
                len(class0),
                n_models=n_ensemble_models,
                fraction=ensemble_fraction,
            )

            joblib.dump(
                svm_models,
                f"{model_prefix}_{differentiation_tag}_svm_ensemble.joblib",
            )
            joblib.dump(
                mlp_models,
                f"{model_prefix}_{differentiation_tag}_mlp_ensemble.joblib",
            )

        else:
            best_svm = train_svm(actual_x, y)
            best_mlp = train_mlp(actual_x, y)

            joblib.dump(best_svm, f"{model_prefix}_{differentiation_tag}_svm.joblib")
            joblib.dump(best_mlp, f"{model_prefix}_{differentiation_tag}_mlp.joblib")

        if not use_data_ensemble:
            fpr_svm, tpr_svm, std_svm, auc_svm, std_auc_svm = cross_validated_roc(
                best_svm, actual_x, y
            )
            fpr_mlp, tpr_mlp, std_mlp, auc_mlp, std_auc_mlp = cross_validated_roc(
                best_mlp, actual_x, y
            )

            plt.figure()

            plt.plot(
                fpr_svm, tpr_svm, label=f"SVM (AUC={auc_svm:.3f}±{std_auc_svm:.3f})"
            )
            plt.fill_between(
                fpr_svm,
                np.maximum(tpr_svm - std_svm, 0),
                np.minimum(tpr_svm + std_svm, 1),
                alpha=0.2,
            )

            plt.plot(
                fpr_mlp, tpr_mlp, label=f"MLP (AUC={auc_mlp:.3f}±{std_auc_mlp:.3f})"
            )
            plt.fill_between(
                fpr_mlp,
                np.maximum(tpr_mlp - std_mlp, 0),
                np.minimum(tpr_mlp + std_mlp, 1),
                alpha=0.2,
            )

            plt.plot([0, 1], [0, 1], "--", color="gray")

            plt.legend()
            plt.grid(True)
            plt.title(f"Cross-Validated ROC for individual classifiers{subtitle}")
            plt.savefig(f"{model_prefix}_{differentiation_tag}_cv_roc.png")
            plt.cla()
            plt.close()
        else:
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=763487356)

            mean_fpr = np.linspace(0, 1, 200)
            tprs = {"svm": [], "mlp": []}
            aucs = {"svm": [], "mlp": []}

            for train_idx, test_idx in skf.split(actual_x, y):
                X_train, X_test = actual_x[train_idx], actual_x[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]

                def predict_proba_ensemble(models, X):
                    probs = np.zeros(len(X))
                    for m in models:
                        probs += m.predict_proba(X)[:, 1]
                    print(len(models), len(X), probs.shape)
                    return probs / len(models)

                for model_key, models in [("svm", svm_models), ("mlp", mlp_models)]:

                    probs = predict_proba_ensemble(models, X_test)
                    # mlp_probs = predict_proba_ensemble(mlp_models, X_test)

                    fpr, tpr, _ = roc_curve(y_test, probs)
                    roc_auc = auc(fpr, tpr)

                    aucs[model_key].append(roc_auc)

                    interp_tpr = np.interp(mean_fpr, fpr, tpr)
                    interp_tpr[0] = 0.0
                    tprs[model_key].append(interp_tpr)

            plt.figure()
            for model_key in ["svm", "mlp"]:
                mean_tpr = np.mean(tprs[model_key], axis=0)
                std_tpr = np.std(tprs[model_key], axis=0)

                plt.plot(
                    mean_fpr,
                    mean_tpr,
                    label=f"Ensemble of `{model_key.upper()}`s (AUC={np.mean(aucs[model_key]):.3f})",
                )
                plt.fill_between(
                    mean_fpr,
                    np.maximum(mean_tpr - std_tpr, 0),
                    np.minimum(mean_tpr + std_tpr, 1),
                    alpha=0.2,
                )

            plt.plot([0, 1], [0, 1], "--", color="gray")
            plt.title(
                f"Fold-wise ROC for {n_ensemble_models} ensembled classifiers\n({ensemble_fraction*100}% data seen){subtitle}"
            )
            plt.legend()
            plt.grid(True)
            plt.savefig(f"{model_prefix}_{differentiation_tag}_cv_roc_ens.png")
            plt.cla()
            plt.close()

    for n_components in components_to_try:
        transformers = [
            (PCA(n_components), "PCA"),
            # (LinearDiscriminantAnalysis("eigen", n_components=n_components), "LDA"),
        ]
        for transformer, name_transformer in transformers:
            actual_x = transformer.fit_transform(X, y)
            do_fit(
                actual_x,
                y,
                f"{name_transformer}_{n_components}",
                f"\nNum. PCA components : {n_components}; explained var. : {np.sum(transformer.explained_variance_ratio_)}",
            )
    do_fit(X, y, "full", "")


def train_subset_ensemble_auto(
    class0,
    class1,
    subsets=subsets_glob,
    subset_weights=subset_weights_glob,
    family_weights=family_weights_glob,
    model_prefix="ensemble",
    n_iter=100,
    use_data_ensemble=False,
    ensemble_fraction=0.75,
    n_ensemble_models=5,
):
    X = np.vstack([class0, class1])
    y = np.array([0] * len(class0) + [1] * len(class1))

    print(f"Total vectors: {len(X)}")

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    joblib.dump(scaler, f"{model_prefix}_scaler.joblib")

    svm_models = []
    mlp_models = []

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

    def re_fit_model(X, y, model_to_refit):
        model_clone = model_to_refit.__class__(**model_to_refit.get_params())
        model_clone.fit(X, y)
        return model_clone

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

    def predict_proba_ensemble(models, X):
        probs = np.zeros(len(X))
        for m in models:
            probs += m.predict_proba(X)[:, 1]
        print(len(models), len(X), probs.shape)
        return probs / len(models)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=46384)
    mean_fpr = np.linspace(0, 1, 200)

    tprs = []
    aucs = []

    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        svm_models_fold = []
        mlp_models_fold = []

        if not use_data_ensemble:
            for i, subset in enumerate(subsets):
                X_sub_train = X_train[:, subset]

                svm_models_fold.append(
                    re_fit_model(X_sub_train, y_train, svm_models[i])
                )
                mlp_models_fold.append(
                    re_fit_model(X_sub_train, y_train, mlp_models[i])
                )
        else:
            svm_models_fold = svm_models
            mlp_models_fold = mlp_models

        def combined_proba_fold(X_input):
            svm_probs = np.zeros(len(X_input))
            mlp_probs = np.zeros(len(X_input))

            for i, subset in enumerate(subsets):
                w = subset_weights[i]

                if not use_data_ensemble:
                    svm_probs += (
                        w * svm_models_fold[i].predict_proba(X_input[:, subset])[:, 1]
                    )
                    mlp_probs += (
                        w * mlp_models_fold[i].predict_proba(X_input[:, subset])[:, 1]
                    )
                else:
                    svm_probs += w * predict_proba_ensemble(
                        svm_models_fold[i], X_input[:, subset]
                    )
                    mlp_probs += w * predict_proba_ensemble(
                        mlp_models_fold[i], X_input[:, subset]
                    )

            svm_probs /= sum(subset_weights)
            mlp_probs /= sum(subset_weights)

            return (
                family_weights["mlp"] * mlp_probs + family_weights["svm"] * svm_probs
            ) / (family_weights["mlp"] + family_weights["svm"])

        probs = combined_proba_fold(X_test)

        fpr, tpr, _ = roc_curve(y_test, probs)
        roc_auc = auc(fpr, tpr)

        aucs.append(roc_auc)

        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0
        tprs.append(interp_tpr)

    mean_tpr = np.mean(tprs, axis=0)
    std_tpr = np.std(tprs, axis=0)

    plt.figure()
    plt.plot(
        mean_fpr, mean_tpr, label=f"Metric family ensemble (AUC={np.mean(aucs):.3f})"
    )
    plt.fill_between(
        mean_fpr,
        np.maximum(mean_tpr - std_tpr, 0),
        np.minimum(mean_tpr + std_tpr, 1),
        alpha=0.2,
    )
    plt.plot([0, 1], [0, 1], "--", color="gray")
    if use_data_ensemble:
        plt.title(
            f"Fold-wise ROC for ensembled per-metric-family classifiers\n({ensemble_fraction*100}% data seen;{n_ensemble_models} models in ensemble)"
        )
    else:
        plt.title(f"Cross-Validated ROC for ensembled per-metric-family classifiers")
    plt.legend()
    plt.grid(True)
    plt.savefig(
        f"{model_prefix}_cv_roc_metric_split_{'ens' if use_data_ensemble else ''}.png"
    )
    plt.cla()
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
    directory: str,
    chunk_size: int,
    division_depth: int,
    fmt: str,
    data_fraction: float = 1.0,
    if_quad_data: bool = False,
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

    files = np.array(
        sorted(
            f
            for f in os.listdir(directory)
            if any(f.lower().strip().endswith(ext) for ext in possible_extensions)
        )
    )
    files = files[np.random.randint(0, len(files), int(data_fraction * len(files)))]

    print(f"Total heightmaps to evaluate: {len(files)}")

    metric_vectors = []

    if if_quad_data:
        for filename in tqdm.tqdm(files):
            path = os.path.join(directory, filename)
            heightmap = file_reader(path)
            half_side_len = int(len(heightmap) / 2)

            print(f"\nProcessing heightmap: {filename}")

            for sx in range(2):
                for sy in range(2):
                    subterrain = heightmap[
                        sx * half_side_len : (sx + 1) * half_side_len,
                        sy * half_side_len : (sy + 1) * half_side_len,
                    ]

                    height_max = subterrain.max()
                    if (
                        np.isnan(height_max)
                        or np.isnan(subterrain.min())
                        or np.isclose(height_max, 0.0)
                    ):
                        print("Skipping tile! Invalid values")
                        continue

                    if np.mean(subterrain > (255.0 * 0.05)) <= 0.6:
                        print("Skipping tile! Too much water")
                        continue

                    print(f"Terrain split: {sx * 2 + sy}")
                    metric_vector = elib.get_metric_vector(
                        subterrain, chunk_size, division_depth, subterrain.max(), True
                    )

                    metric_vectors.append(metric_vector)

            del heightmap
        else:
            for filename in tqdm.tqdm(files):
                path = os.path.join(directory, filename)
                heightmap = file_reader(path)

                print(f"\nProcessing heightmap: {filename}")

                metric_vector = elib.get_metric_vector(
                    heightmap, chunk_size, division_depth, population_max, True
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
        "--train-data-fraction",
        type=float,
        help="What fraction of the actual visual data to vectorize.",
    )

    parser.add_argument(
        "--use-saved",
        action="store_true",
        default=False,
        help="If use previously stored vectors for training.",
    )

    parser.add_argument(
        "--quad-data",
        action="store_true",
        default=False,
        help="If cut up heightmaps in four subheightmap to increase the dataste size.",
    )

    parser.add_argument(
        "--train-ensemble",
        action="store_true",
        default=False,
        help="If train ensembles of models on different subsets of data.",
    )

    parser.add_argument(
        "--only-embed",
        action="store_true",
        default=False,
        help="If only compute vector embeddings of heightmaps.",
    )

    args = parser.parse_args()

    fmt = args.format.lower().strip()

    directory_interesting = args.directory_interesting
    directory_boring = args.directory_boring

    if args.mode == "train":
        if not args.use_saved:
            print(f"Building vectors anew!")
            interesting_vectors = build_metric_vectors(
                directory_interesting,
                args.chunk_size,
                args.division_depth,
                fmt,
                args.train_data_fraction if args.train_data_fraction else 1.0,
                args.quad_data,
            )
            interesting_vectors_pd = pd.DataFrame(interesting_vectors)
            interesting_vectors_pd.to_csv(
                os.path.join(
                    directory_interesting, "interesting_metric_vectors_new.csv"
                )
            )

            boring_vectors = build_metric_vectors(
                directory_boring,
                args.chunk_size,
                args.division_depth,
                fmt,
                args.train_data_fraction if args.train_data_fraction else 1.0,
                args.quad_data,
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
                    "interesting_metric_vectors_new.csv",
                )
            )
            interesting_vectors = interesting_vectors_pd.to_numpy()[:, 1:]
            interesting_vectors = np.nan_to_num(interesting_vectors)

            boring_vectors_pd = pd.read_csv(
                os.path.join(directory_boring, "boring_metric_vectors_new.csv")
            )
            boring_vectors = boring_vectors_pd.to_numpy()[:, 1:]
            boring_vectors = np.nan_to_num(boring_vectors)

        if not args.only_embed:
            train_and_save_models_auto(
                boring_vectors,
                interesting_vectors,
                None,
                "test_model",
                25,
                use_data_ensemble=args.train_ensemble,
            )
            train_subset_ensemble_auto(
                boring_vectors,
                interesting_vectors,
                use_data_ensemble=args.train_ensemble,
                model_prefix="test_ensemble",
                n_iter=25,
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


if __name__ == "__main__":
    main()
