import joblib
import numpy as np
import pandas as pd

family_weights_glob = {"mlp": 0.48, "svm": 0.51}
subsets_glob = [[0, 1], [2, 3], [4, 5]]
subset_weights_glob = [0.36, 0.34, 0.28]


def classify_with_ensemble(
    X,
    subsets=subsets_glob,
    subset_weights=subset_weights_glob,
    family_weights=family_weights_glob,
    model_prefix="ensemble",
    use_data_ensemble=False,
) -> pd.DataFrame:
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
        if use_data_ensemble:
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

    return df


def classify_with_saved_models(
    X, model_prefix="model", use_data_ensemble=False
) -> pd.DataFrame:
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
    return df
