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
from sklearn.metrics import roc_curve, auc, classification_report

from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV

import classificationlib as classlib

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
    reports = []

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

        y_pred = (probs >= 0.5).astype(int)
        rep = classification_report(y_test, y_pred, output_dict=True)
        reports.append(rep)

    mean_tpr = np.mean(tprs, axis=0)
    std_tpr = np.std(tprs, axis=0)
    mean_auc = np.mean(aucs)
    std_auc = np.std(aucs)

    avg_report = {}

    for label in ["0", "1"]:
        avg_report[label] = {}
        for metric in reports[0][label].keys():
            avg_report[label][metric] = np.mean([rep[label][metric] for rep in reports])

    report_df = pd.DataFrame(avg_report)

    return mean_fpr, mean_tpr, std_tpr, mean_auc, std_auc, report_df


def train_and_save_models_auto(
    class0,
    class1,
    feature_weights=None,
    model_prefix="model",
    n_iter=100,
    components_to_try=[2, 3, 4],
    state_mlp=4565387,
    state_svm=8574554,
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
            SVC(probability=True, random_state=state_svm),
            svm_param_dist,
            n_iter=n_iter,
            scoring="roc_auc",
            cv=5,
            verbose=2,
            n_jobs=8,
            random_state=state_svm,
        )
        svm_search.fit(X, y)
        return svm_search.best_estimator_

    def train_mlp(X, y):
        mlp_search = RandomizedSearchCV(
            MLPClassifier(max_iter=500, early_stopping=True, random_state=state_mlp),
            mlp_param_dist,
            n_iter=n_iter,
            scoring="roc_auc",
            cv=5,
            verbose=2,
            n_jobs=8,
            random_state=state_mlp,
        )
        mlp_search.fit(X, y)
        return mlp_search.best_estimator_

    def do_fit(
        actual_x: np.ndarray, y: np.ndarray, differentiation_tag: str, subtitle: str
    ):
        best_svm = train_svm(actual_x, y)
        best_mlp = train_mlp(actual_x, y)

        joblib.dump(best_svm, f"{model_prefix}_{differentiation_tag}_svm.joblib")
        joblib.dump(best_mlp, f"{model_prefix}_{differentiation_tag}_mlp.joblib")

        fpr_svm, tpr_svm, std_svm, auc_svm, std_auc_svm, svm_f1 = cross_validated_roc(
            best_svm, actual_x, y
        )
        fpr_mlp, tpr_mlp, std_mlp, auc_mlp, std_auc_mlp, mlp_f1 = cross_validated_roc(
            best_mlp, actual_x, y
        )

        plt.figure()

        plt.plot(
            fpr_svm,
            tpr_svm,
            label=f"SVM (AUC={auc_svm:.3f}±{std_auc_svm:.3f}, $\overline{{F1}}$ = {0.5*(svm_f1['1']["f1-score"] + svm_f1['0']["f1-score"]):.2f})",
        )
        plt.fill_between(
            fpr_svm,
            np.maximum(tpr_svm - std_svm, 0),
            np.minimum(tpr_svm + std_svm, 1),
            alpha=0.2,
        )

        plt.plot(
            fpr_mlp,
            tpr_mlp,
            label=f"MLP (AUC={auc_mlp:.3f}±{std_auc_mlp:.3f}, $\overline{{F1}}$ = {0.5*(mlp_f1['1']["f1-score"] + mlp_f1['0']["f1-score"]):.2f})",
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

        svm_f1.to_csv(f"{model_prefix}_{differentiation_tag}_cv_f1_svm.csv")
        mlp_f1.to_csv(f"{model_prefix}_{differentiation_tag}_cv_f1_mlp.csv")

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
                f"\nNum. PCA components : {n_components}; explained var. : {np.sum(transformer.explained_variance_ratio_):.2f}",
            )
    do_fit(X, y, "full", "")


def train_subset_ensemble_auto(
    class0,
    class1,
    subsets=classlib.subsets_glob,
    subset_weights=classlib.subset_weights_glob,
    family_weights=classlib.family_weights_glob,
    model_prefix="ensemble",
    n_iter=100,
    state_mlp=4565387,
    state_svm=8574554,
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
                random_state=state_svm,
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
                random_state=state_mlp,
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

        best_svm = train_svm(X_sub, y)
        best_mlp = train_mlp(X_sub, y)

        joblib.dump(best_svm, f"{model_prefix}_svm_{i}.joblib")
        joblib.dump(best_mlp, f"{model_prefix}_mlp_{i}.joblib")

        svm_models.append(best_svm)
        mlp_models.append(best_mlp)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=46384)
    mean_fpr = np.linspace(0, 1, 200)

    tprs = []
    aucs = []
    reports = []

    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        svm_models_fold = []
        mlp_models_fold = []

        for i, subset in enumerate(subsets):
            X_sub_train = X_train[:, subset]

            svm_models_fold.append(re_fit_model(X_sub_train, y_train, svm_models[i]))
            mlp_models_fold.append(re_fit_model(X_sub_train, y_train, mlp_models[i]))

        def combined_proba_fold(X_input):
            svm_probs = np.zeros(len(X_input))
            mlp_probs = np.zeros(len(X_input))

            for i, subset in enumerate(subsets):
                w = subset_weights[i]

                svm_probs += (
                    w * svm_models_fold[i].predict_proba(X_input[:, subset])[:, 1]
                )
                mlp_probs += (
                    w * mlp_models_fold[i].predict_proba(X_input[:, subset])[:, 1]
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

        y_pred = (probs >= 0.5).astype(int)
        rep = classification_report(y_test, y_pred, output_dict=True)
        reports.append(rep)

    mean_tpr = np.mean(tprs, axis=0)
    std_tpr = np.std(tprs, axis=0)

    avg_report = {}

    for label in ["0", "1"]:
        avg_report[label] = {}
        for metric in reports[0][label].keys():
            avg_report[label][metric] = np.mean([rep[label][metric] for rep in reports])

    report_df = pd.DataFrame(avg_report)
    report_df.to_csv(f"{model_prefix}_cv_f1_metric_split_.csv")

    plt.figure()
    plt.plot(
        mean_fpr,
        mean_tpr,
        label=f"Metric family ensemble (AUC={np.mean(aucs):.3f}, $\overline{{F1}}$ = {0.5*(report_df['1']["f1-score"] + report_df['0']["f1-score"]):.2f})",
    )
    plt.fill_between(
        mean_fpr,
        np.maximum(mean_tpr - std_tpr, 0),
        np.minimum(mean_tpr + std_tpr, 1),
        alpha=0.2,
    )
    plt.plot([0, 1], [0, 1], "--", color="gray")
    plt.title(f"Cross-Validated ROC for ensembled per-metric-family classifiers")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{model_prefix}_cv_roc_metric_split_.png")
    plt.cla()
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Evaluate metrics for the heightmaps and train classification models.")

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
        help="How many divisions to make along heightmap side for composite aes metric.",
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
            interesting_vectors = elib.build_metric_vectors(
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

            boring_vectors = elib.build_metric_vectors(
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
                    "interesting_metric_vectors_4x4.csv",
                )
            )
            interesting_vectors = interesting_vectors_pd.to_numpy()[:, 1:]
            interesting_vectors = interesting_vectors[interesting_vectors[:, 2] != 2.0]

            boring_vectors_pd = pd.read_csv(
                os.path.join(directory_boring, "boring_metric_vectors_4x4.csv")
            )
            boring_vectors = boring_vectors_pd.to_numpy()[:, 1:]
            # boring_vectors = boring_vectors[boring_vectors[:, 2] != 2.0]

        if not args.only_embed:
            train_and_save_models_auto(
                boring_vectors,
                interesting_vectors,
                None,
                "test_model",
                35,
                state_mlp=295,
                state_svm=447,
            )
            train_subset_ensemble_auto(
                boring_vectors,
                interesting_vectors,
                model_prefix="test_ensemble",
                n_iter=25,
                state_mlp=447,
                state_svm=447,
            )
    elif args.mode == "classify":
        vectors_to_classify = elib.build_metric_vectors(
            args.directory_classify, args.chunk_size, args.division_depth, fmt
        )
        classlib.classify_with_saved_models(
            vectors_to_classify,
            model_prefix="test_model",
        )
        classlib.classify_with_ensemble(
            vectors_to_classify,
            model_prefix="test_ensemble",
        )
    else:
        raise ValueError("Wrong script mode")


if __name__ == "__main__":
    main()
