"""Inferencia compartida para evaluación y API."""
import numpy as np
import pandas as pd

from preprocessing import TARGET, apply_scaling, build_features, get_scale_columns


def build_inference_matrix(df, artifact, segment_artifact=None):
    """Construye matriz de features; segment_artifact para modelos duales."""
    source = segment_artifact or artifact
    feature_config = source.get("feature_config", artifact.get("feature_config", {}))
    X = build_features(df, feature_config)
    for col in source.get("dummy_columns", artifact.get("dummy_columns", [])):
        if col not in X.columns:
            X[col] = 0.0

    selected = source["feature_names"] if segment_artifact else artifact["selected_features"]
    for col in selected:
        if col not in X.columns:
            X[col] = 0.0

    X = X[selected]
    scale_cols = source.get("scale_cols") or source.get("scale_columns") or get_scale_columns(X)

    scaler_mean = source.get("scaler_mean") or artifact.get("scaler_mean")
    scaler_scale = source.get("scaler_scale") or artifact.get("scaler_scale")
    if scaler_mean and scaler_scale:
        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler()
        scaler.mean_ = np.array(scaler_mean)
        scaler.scale_ = np.array(scaler_scale)
        scaler.n_features_in_ = len(scale_cols)
        X, _ = apply_scaling(X, scale_cols, scaler=scaler, fit=False)
    return X


def _predict_log(model, X, use_log=True):
    pred = model.predict(X)
    if use_log:
        return np.expm1(pred)
    return pred


def predict_ensemble(df, artifact):
    """Combina predicciones de varios miembros según estrategia aprendida en CV."""
    members = artifact["members"]
    names = [m["name"] for m in members]
    stack = np.column_stack([predict_cnt(df, m["artifact"]) for m in members])
    strategy = artifact.get("strategy", "weighted_average")

    if strategy == "weighted_average":
        weights = artifact.get("weights", {})
        w = np.array([weights.get(n, 1.0 / len(names)) for n in names], dtype=float)
        w /= w.sum()
        preds = stack @ w
    elif strategy == "median":
        preds = np.median(stack, axis=1)
    elif strategy == "segment_routing":
        routing = artifact.get("segment_routing", {})
        preds = np.zeros(len(df), dtype=float)
        for i in range(len(df)):
            row = df.iloc[i]
            if row["workingday"] == 0:
                member = routing.get("workingday_0", names[0])
            elif row["yr"] == 1:
                member = routing.get("yr_1", routing.get("workingday_1", names[0]))
            else:
                member = routing.get("workingday_1", names[0])
            idx = names.index(member) if member in names else 0
            preds[i] = stack[i, idx]
    else:
        preds = np.mean(stack, axis=1)

    return np.maximum(preds, 0)


def predict_cnt(df, artifact):
    model_type = artifact.get("model_type", "")

    if model_type == "ensemble":
        return predict_ensemble(df, artifact)

    if model_type == "dual":
        preds = np.zeros(len(df), dtype=float)
        segments = artifact["segments"]
        workingday = df["workingday"].values

        for wd, key in [(1, "workingday"), (0, "weekend")]:
            mask = workingday == wd
            if not mask.any() or key not in segments:
                continue
            seg = segments[key]
            X = build_inference_matrix(df[mask], artifact, segment_artifact=seg)
            preds[mask] = np.maximum(_predict_log(seg["model"], X, artifact.get("use_log_transform", True)), 0)
        return preds

    if model_type == "two_stage":
        X = build_inference_matrix(df, artifact)
        is_peak = artifact["peak_classifier"].predict(X)
        reg_normal = artifact["regressor_normal"]
        reg_peak = artifact["regressor_peak"]
        use_log = artifact.get("use_log_transform", True)

        pred_normal = _predict_log(reg_normal, X, use_log)
        pred_peak = _predict_log(reg_peak, X, use_log)
        preds = np.where(is_peak == 1, pred_peak, pred_normal)
        return np.maximum(preds, 0)

    X = build_inference_matrix(df, artifact)
    model = artifact["model"]
    preds = _predict_log(model, X, artifact.get("use_log_transform", True))
    preds = np.maximum(preds, 0)

    factor = artifact.get("weekend_calibration_factor", 1.0)
    if factor != 1.0 and "workingday" in df.columns:
        weekend_mask = df["workingday"].values == 0
        preds[weekend_mask] *= factor

    return preds


def get_primary_artifact(artifact):
    """Artifact base para features/API cuando el modelo activo es un ensemble."""
    if artifact.get("model_type") != "ensemble":
        return artifact
    primary_name = artifact.get("primary_member")
    for member in artifact.get("members", []):
        if member["name"] == primary_name:
            return member["artifact"]
    if artifact.get("members"):
        return artifact["members"][0]["artifact"]
    return artifact


def build_row_from_payload(payload, artifact):
    row = {}
    for feature in artifact.get("base_features", []):
        if feature in ("season", "weathersit"):
            row[feature] = int(payload.get(feature, 1))
        else:
            row[feature] = float(payload.get(feature, 0))
    return pd.DataFrame([row])
