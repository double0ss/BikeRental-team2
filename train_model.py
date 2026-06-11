"""
Entrenamiento y selección del mejor modelo para predecir cnt.
Compara varios algoritmos con validación cruzada temporal.
"""
import json
import os
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.feature_selection import RFECV
from sklearn.linear_model import ElasticNetCV, HuberRegressor, LassoCV, LinearRegression
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

from preprocessing import (
    TARGET,
    apply_scaling,
    build_features,
    extract_train_stats,
    get_dummy_columns,
    get_scale_columns,
    load_and_clean,
    winsorize_target,
)

warnings.filterwarnings("ignore")

DATA_PATH = "data/BikeRentalDaily_train.csv"
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

FEATURE_CONFIGS = [
    {"name": "full", "cyclical": True, "interaction": True, "leaflets": True},
    {"name": "no_leaflets", "cyclical": True, "interaction": True, "leaflets": False},
    {"name": "no_cyclical", "cyclical": False, "interaction": True, "leaflets": True},
    {"name": "minimal", "cyclical": False, "interaction": False, "leaflets": False},
]


def adjusted_r2(r2, n, p):
    if n <= p + 1:
        return r2
    return 1 - (1 - r2) * (n - 1) / (n - p - 1)


def compute_metrics(y_true, y_pred, n_features=1):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.maximum(np.asarray(y_pred, dtype=float), 0)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return {
        "mae": float(mae),
        "rmse": float(rmse),
        "r2": float(r2),
        "r2_adj": float(adjusted_r2(r2, len(y_true), n_features)),
    }


def prepare_dataset(df, feature_config, fit_scaler=None, columns=None):
    X_raw = build_features(df, feature_config)
    if columns is not None:
        for col in columns:
            if col not in X_raw.columns:
                X_raw[col] = 0.0
        X_raw = X_raw[columns]

    y_raw = df[TARGET].astype(float).values
    y_log = np.log1p(y_raw)
    scale_cols = get_scale_columns(X_raw)

    fit = fit_scaler is None
    X, scaler = apply_scaling(X_raw, scale_cols, scaler=fit_scaler, fit=fit)
    return X, X_raw, y_log, y_raw, scaler, scale_cols


def predict_model(model, X, use_log=True):
    pred_log = model.predict(X)
    if use_log:
        return np.expm1(pred_log)
    return pred_log


def cv_evaluate(model, X_raw, y_log, y_raw, scale_cols=None, n_splits=4):
    from sklearn.base import clone

    if scale_cols is None:
        scale_cols = get_scale_columns(X_raw)

    tscv = TimeSeriesSplit(n_splits=n_splits)
    fold_metrics = []

    for train_idx, val_idx in tscv.split(X_raw):
        X_train_raw = X_raw.iloc[train_idx]
        X_val_raw = X_raw.iloc[val_idx]
        y_train_log = y_log[train_idx]
        y_val_raw = y_raw[val_idx]

        X_train, fold_scaler = apply_scaling(X_train_raw, scale_cols, fit=True)
        X_val, _ = apply_scaling(X_val_raw, scale_cols, scaler=fold_scaler, fit=False)

        est = model.estimator if hasattr(model, "estimator") else model
        fold_model = clone(est)
        fold_model.fit(X_train, y_train_log)
        y_pred = predict_model(fold_model, X_val)
        fold_metrics.append(compute_metrics(y_val_raw, y_pred, n_features=X_raw.shape[1]))

    return {
        "mae": float(np.mean([m["mae"] for m in fold_metrics])),
        "rmse": float(np.mean([m["rmse"] for m in fold_metrics])),
        "r2": float(np.mean([m["r2"] for m in fold_metrics])),
    }


def get_candidates():
    return {
        "linear": LinearRegression(),
        "lasso": LassoCV(alphas=np.logspace(-3, 1, 30), cv=4, max_iter=10000, random_state=42),
        "elasticnet": ElasticNetCV(
            l1_ratio=[0.1, 0.5, 0.7, 0.9, 0.95],
            alphas=np.logspace(-3, 1, 20),
            cv=4,
            max_iter=10000,
            random_state=42,
        ),
        "random_forest": RandomForestRegressor(
            n_estimators=300,
            max_depth=16,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        ),
        "gradient_boosting": GradientBoostingRegressor(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.9,
            random_state=42,
        ),
    }


def tune_top_model(model_name, X, y_log, y_raw):
    tscv = TimeSeriesSplit(n_splits=4)

    if model_name == "random_forest":
        search = RandomizedSearchCV(
            RandomForestRegressor(random_state=42, n_jobs=-1),
            {
                "n_estimators": [200, 400, 600],
                "max_depth": [8, 12, 16, 20, None],
                "min_samples_split": [2, 5, 8],
                "min_samples_leaf": [1, 2, 4],
                "max_features": ["sqrt", "log2", None],
            },
            n_iter=20,
            cv=tscv,
            scoring="neg_root_mean_squared_error",
            random_state=42,
            n_jobs=-1,
        )
    elif model_name == "gradient_boosting":
        search = RandomizedSearchCV(
            GradientBoostingRegressor(random_state=42),
            {
                "n_estimators": [200, 300, 500],
                "max_depth": [3, 4, 5, 6],
                "learning_rate": [0.03, 0.05, 0.08, 0.1],
                "subsample": [0.8, 0.9, 1.0],
            },
            n_iter=20,
            cv=tscv,
            scoring="neg_root_mean_squared_error",
            random_state=42,
            n_jobs=-1,
        )
    else:
        return get_candidates()[model_name]

    search.fit(X, y_log)
    return search.best_estimator_


def run_training_round(round_name="round1", tune=False, top_model_name=None, feature_config=None):
    df = load_and_clean(DATA_PATH)
    results = []

    configs = [feature_config] if feature_config else FEATURE_CONFIGS
    candidates = get_candidates()

    for cfg in configs:
        X, X_raw, y_log, y_raw, scaler, scale_cols = prepare_dataset(df, cfg)
        feature_names = X.columns.tolist()

        model_set = candidates
        if tune and top_model_name:
            model_set = {top_model_name: tune_top_model(top_model_name, X, y_log, y_raw)}

        for model_name, model in model_set.items():
            if model_name == "linear":
                rfecv = RFECV(
                    estimator=LinearRegression(),
                    step=1,
                    cv=TimeSeriesSplit(n_splits=4),
                    scoring="neg_root_mean_squared_error",
                    min_features_to_select=3,
                    n_jobs=-1,
                )
                rfecv.fit(X, y_log)
                selected = X.columns[rfecv.support_].tolist()
                X_sel = X[selected]
                X_raw_sel = X_raw[selected]
                sel_scale_cols = get_scale_columns(X_raw_sel)
                metrics = cv_evaluate(
                    LinearRegression(), X_raw_sel, y_log, y_raw, scale_cols=sel_scale_cols
                )
                final_model = LinearRegression().fit(X_sel, y_log)
                feature_names = selected
                X = X_sel
                scale_cols = sel_scale_cols
            else:
                metrics = cv_evaluate(model, X_raw, y_log, y_raw, scale_cols=scale_cols)
                final_model = model.fit(X, y_log)

            results.append(
                {
                    "round": round_name,
                    "feature_config": cfg,
                    "model_name": model_name,
                    "cv_metrics": metrics,
                    "model": final_model,
                    "X": X,
                    "y_log": y_log,
                    "y_raw": y_raw,
                    "scaler": scaler,
                    "scale_cols": scale_cols,
                    "feature_names": feature_names,
                    "dummy_columns": get_dummy_columns(build_features(df, cfg)),
                    "df": df,
                    "normalized": True,
                }
            )
            print(
                f"[{round_name}] {cfg['name']:12} | {model_name:18} | "
                f"CV RMSE={metrics['rmse']:8.1f} | CV MAE={metrics['mae']:8.1f} | CV R2={metrics['r2']:.4f}"
            )

    best = min(results, key=lambda r: r["cv_metrics"]["rmse"])
    return best, results


def save_artifact(best, cv_results_summary):
    df = best["df"]
    casual_ratio = (df["casual"] / df["cnt"]).median()
    registered_ratio = (df["registered"] / df["cnt"]).median()

    X_raw_final = build_features(df, best["feature_config"])[best["feature_names"]]
    scale_cols = get_scale_columns(X_raw_final)
    _, scaler = apply_scaling(X_raw_final, scale_cols, fit=True)

    artifact = {
        "model_type": best["model_name"],
        "target": TARGET,
        "use_log_transform": True,
        "feature_config": best["feature_config"],
        "selected_features": best["feature_names"],
        "scale_columns": scale_cols,
        "scaler_mean": scaler.mean_.tolist() if scale_cols else [],
        "scaler_scale": scaler.scale_.tolist() if scale_cols else [],
        "dummy_columns": best["dummy_columns"],
        "base_features": [
            "temp", "hum", "windspeed", "leaflets", "yr", "mnth",
            "holiday", "weekday", "workingday", "price_reduction", "season", "weathersit",
        ],
        "casual_ratio": float(casual_ratio),
        "registered_ratio": float(registered_ratio),
        "base_year": 2011,
        "cv_metrics": best["cv_metrics"],
        "normalized": True,
        "model": best["model"],
    }

    model_path = os.path.join(OUTPUT_DIR, "model_cnt.pkl")
    joblib.dump(artifact, model_path)

    train_pred = predict_model(best["model"], best["X"])
    train_metrics = compute_metrics(best["y_raw"], train_pred, len(best["feature_names"]))

    config = {
        "model_type": best["model_name"],
        "feature_config": best["feature_config"],
        "primary_target": "cnt",
        "selected_features": best["feature_names"],
        "train_size": len(df),
        "results": {"cnt": {"train": train_metrics, "cv": best["cv_metrics"]}},
        "cv_comparison": cv_results_summary,
    }
    with open(os.path.join(OUTPUT_DIR, "model_config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, default=str)

    print(f"\nMejor modelo: {best['model_name']} + {best['feature_config']['name']}")
    print(f"CV RMSE={best['cv_metrics']['rmse']:.2f} | Guardado en {model_path}")
    return artifact


def main():
    from data_analysis import run_full_analysis

    print("=== Paso 0: análisis de sesgos y outliers ===")
    bias_report = run_full_analysis(DATA_PATH, "tests/BikeRentalDaily_test.csv")
    print()

    print("=== Ronda 1: corrección base + comparación de modelos ===")
    best1, results1 = run_training_round("round1")

    print("\n=== Ronda 2: mejor config de features ===")
    best_cfg = best1["feature_config"]
    best2, results2 = run_training_round("round2", feature_config=best_cfg)

    print("\n=== Ronda 3: afinación de hiperparámetros ===")
    best3, results3 = run_training_round(
        "round3",
        tune=True,
        top_model_name=best2["model_name"],
        feature_config=best_cfg,
    )

    all_results = results1 + results2 + results3
    summary = [
        {
            "round": r["round"],
            "model": r["model_name"],
            "features": r["feature_config"]["name"],
            **r["cv_metrics"],
        }
        for r in all_results
    ]

    final_best = min([best1, best2, best3], key=lambda r: r["cv_metrics"]["rmse"])
    artifact = save_artifact(final_best, summary)
    artifact["bias_outlier_summary"] = bias_report["summary"]
    joblib.dump(artifact, os.path.join(OUTPUT_DIR, "model_cnt.pkl"))


def build_model_from_experiment(model_type, model_params=None):
    """Instancia un estimador según tipo y parámetros del experimento."""
    model_params = model_params or {}
    if model_type == "gradient_boosting":
        return GradientBoostingRegressor(random_state=42, **model_params)
    if model_type == "gradient_boosting_huber":
        params = {"loss": "huber", "random_state": 42, **model_params}
        return GradientBoostingRegressor(**params)
    if model_type == "hist_gradient_boosting":
        return HistGradientBoostingRegressor(random_state=42, **model_params)
    if model_type == "huber":
        return HuberRegressor(max_iter=1000, **model_params)
    if model_type == "lasso":
        return LassoCV(alphas=np.logspace(-3, 1, 30), cv=4, max_iter=10000, random_state=42)
    if model_type == "elasticnet":
        return ElasticNetCV(
            l1_ratio=[0.1, 0.5, 0.7, 0.9, 0.95],
            alphas=np.logspace(-3, 1, 20),
            cv=4,
            max_iter=10000,
            random_state=42,
        )
    if model_type == "random_forest":
        return RandomForestRegressor(random_state=42, n_jobs=-1, **model_params)
    return get_candidates().get(model_type, GradientBoostingRegressor(random_state=42))


def learn_weekend_calibration(df, X_raw, y_raw, model, scale_cols):
    """Factor multiplicativo para fines de semana aprendido en CV temporal."""
    from sklearn.base import clone
    from sklearn.model_selection import TimeSeriesSplit

    tscv = TimeSeriesSplit(n_splits=4)
    ratios = []
    workingday = df["workingday"].values

    for train_idx, val_idx in tscv.split(X_raw):
        if not np.any(workingday[val_idx] == 0):
            continue
        X_train_raw = X_raw.iloc[train_idx]
        X_val_raw = X_raw.iloc[val_idx]
        y_val_raw = y_raw[val_idx]

        X_train, fold_scaler = apply_scaling(X_train_raw, scale_cols, fit=True)
        X_val, _ = apply_scaling(X_val_raw, scale_cols, scaler=fold_scaler, fit=False)

        fold_model = clone(model)
        fold_model.fit(X_train, np.log1p(y_raw[train_idx]))
        y_pred = predict_model(fold_model, X_val)

        weekend_mask = workingday[val_idx] == 0
        if weekend_mask.sum() == 0:
            continue
        actual = y_val_raw[weekend_mask]
        pred = y_pred[weekend_mask]
        valid = pred > 1
        if valid.sum() > 0:
            ratios.extend((actual[valid] / pred[valid]).tolist())

    if not ratios:
        return 1.0
    return float(np.median(ratios))


def _fit_single_model(df, feature_config, model, winsorize_pct=None):
    work_df = df
    winsor_cap = None
    if winsorize_pct:
        work_df, winsor_cap = winsorize_target(df, winsorize_pct)

    X, X_raw, y_log, y_raw, scaler, scale_cols = prepare_dataset(work_df, feature_config)
    model.fit(X, y_log)
    return {
        "model": model,
        "feature_config": feature_config,
        "feature_names": X.columns.tolist(),
        "dummy_columns": get_dummy_columns(build_features(work_df, feature_config)),
        "scaler": scaler,
        "scale_cols": scale_cols,
        "X": X,
        "X_raw": X_raw,
        "y_log": y_log,
        "y_raw": y_raw,
        "df": work_df,
        "winsor_cap": winsor_cap,
        "cv_metrics": cv_evaluate(model, X_raw, y_log, y_raw, scale_cols=scale_cols),
    }


def train_dual_model(df, feature_config, model_factory):
    """Modelos separados para días laborables y fines de semana."""
    segments = {}
    for wd, label in [(1, "workingday"), (0, "weekend")]:
        subset = df[df["workingday"] == wd].copy()
        if len(subset) < 10:
            continue
        model = model_factory()
        fitted = _fit_single_model(subset, feature_config, model)
        sc = fitted["scaler"]
        fitted["scaler_mean"] = sc.mean_.tolist() if fitted["scale_cols"] else []
        fitted["scaler_scale"] = sc.scale_.tolist() if fitted["scale_cols"] else []
        segments[label] = fitted

    primary = segments.get("workingday") or segments.get("weekend")
    train_stats = extract_train_stats(df)
    return {
        "model_type": "dual",
        "target": TARGET,
        "use_log_transform": True,
        "feature_config": feature_config,
        "segments": segments,
        "selected_features": primary["feature_names"],
        "scale_columns": primary["scale_cols"],
        "scaler_mean": primary["scaler"].mean_.tolist() if primary["scale_cols"] else [],
        "scaler_scale": primary["scaler"].scale_.tolist() if primary["scale_cols"] else [],
        "dummy_columns": primary["dummy_columns"],
        "base_features": [
            "temp", "hum", "windspeed", "leaflets", "yr", "mnth",
            "holiday", "weekday", "workingday", "price_reduction", "season", "weathersit",
        ],
        "train_stats": train_stats,
        "cv_metrics": primary["cv_metrics"],
        "normalized": True,
        "df": df,
    }


def train_two_stage_model(df, feature_config, model_factory, peak_percentile=95):
    """Clasificador de día pico + regresores para demanda normal y extrema."""
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.base import clone

    peak_threshold = float(np.percentile(df[TARGET].astype(float), peak_percentile))
    is_peak = (df[TARGET].astype(float) >= peak_threshold).astype(int)

    X, X_raw, y_log, y_raw, scaler, scale_cols = prepare_dataset(df, feature_config)
    clf = GradientBoostingClassifier(
        n_estimators=100, max_depth=3, random_state=42
    )
    clf.fit(X, is_peak)

    normal_mask = is_peak == 0
    peak_mask = is_peak == 1
    reg_normal = model_factory()
    reg_peak = clone(model_factory())

    reg_normal.fit(X[normal_mask], y_log[normal_mask])
    if peak_mask.sum() >= 5:
        reg_peak.fit(X[peak_mask], y_log[peak_mask])
    else:
        reg_peak = reg_normal

    train_stats = extract_train_stats(df)
    return {
        "model_type": "two_stage",
        "target": TARGET,
        "use_log_transform": True,
        "feature_config": feature_config,
        "peak_classifier": clf,
        "regressor_normal": reg_normal,
        "regressor_peak": reg_peak,
        "peak_threshold": peak_threshold,
        "selected_features": X.columns.tolist(),
        "scale_columns": scale_cols,
        "scaler_mean": scaler.mean_.tolist() if scale_cols else [],
        "scaler_scale": scaler.scale_.tolist() if scale_cols else [],
        "dummy_columns": get_dummy_columns(build_features(df, feature_config)),
        "base_features": [
            "temp", "hum", "windspeed", "leaflets", "yr", "mnth",
            "holiday", "weekday", "workingday", "price_reduction", "season", "weathersit",
        ],
        "train_stats": train_stats,
        "cv_metrics": cv_evaluate(reg_normal, X_raw, y_log, y_raw, scale_cols=scale_cols),
        "normalized": True,
        "df": df,
        "model": reg_normal,
    }


def train_candidate(experiment, df=None, train_stats=None):
    """
    Entrena un candidato según configuración de experimento.
    Retorna artifact listo para guardar/evaluar.
    """
    if df is None:
        df = load_and_clean(DATA_PATH)
    if train_stats is None:
        train_stats = extract_train_stats(df)
    feature_config = experiment.get("feature_config", FEATURE_CONFIGS[3])
    model_type = experiment.get("model_type", "gradient_boosting")
    model_params = experiment.get("model_params", {})
    winsorize_pct = experiment.get("winsorize_pct")

    def model_factory():
        return build_model_from_experiment(model_type, model_params)

    if experiment.get("dual_model"):
        artifact = train_dual_model(df, feature_config, model_factory)
        artifact["experiment"] = experiment["name"]
        artifact["hypothesis"] = experiment.get("hypothesis", "")
        return artifact

    if experiment.get("two_stage"):
        artifact = train_two_stage_model(
            df, feature_config, model_factory,
            peak_percentile=experiment.get("peak_percentile", 95),
        )
        artifact["experiment"] = experiment["name"]
        artifact["hypothesis"] = experiment.get("hypothesis", "")
        return artifact

    fitted = _fit_single_model(df, feature_config, model_factory(), winsorize_pct=winsorize_pct)
    df = fitted["df"]
    casual_ratio = (df["casual"] / df["cnt"]).median()
    registered_ratio = (df["registered"] / df["cnt"]).median()

    weekend_factor = 1.0
    if experiment.get("weekend_calibration"):
        weekend_factor = learn_weekend_calibration(
            df, fitted["X_raw"], fitted["y_raw"], fitted["model"], fitted["scale_cols"]
        )

    artifact = {
        "model_type": model_type,
        "target": TARGET,
        "use_log_transform": True,
        "feature_config": feature_config,
        "selected_features": fitted["feature_names"],
        "scale_columns": fitted["scale_cols"],
        "scaler_mean": fitted["scaler"].mean_.tolist() if fitted["scale_cols"] else [],
        "scaler_scale": fitted["scaler"].scale_.tolist() if fitted["scale_cols"] else [],
        "dummy_columns": fitted["dummy_columns"],
        "base_features": [
            "temp", "hum", "windspeed", "leaflets", "yr", "mnth",
            "holiday", "weekday", "workingday", "price_reduction", "season", "weathersit",
        ],
        "casual_ratio": float(casual_ratio),
        "registered_ratio": float(registered_ratio),
        "base_year": 2011,
        "cv_metrics": fitted["cv_metrics"],
        "normalized": True,
        "model": fitted["model"],
        "train_stats": train_stats,
        "weekend_calibration_factor": weekend_factor,
        "winsor_cap": fitted["winsor_cap"],
        "experiment": experiment["name"],
        "hypothesis": experiment.get("hypothesis", ""),
        "df": df,
    }
    return artifact


def save_candidate_artifact(artifact, cv_results_summary=None):
    """Persiste artifact como modelo principal."""
    model_path = os.path.join(OUTPUT_DIR, "model_cnt.pkl")
    joblib.dump(artifact, model_path)

    df = artifact.get("df")
    if df is None:
        df = load_and_clean(DATA_PATH)
    if artifact.get("model_type") in ("dual", "ensemble"):
        train_pred = predict_model_batch(df, artifact)
        y_raw = df[TARGET].astype(float).values
        n_feat = len(artifact.get("selected_features", []))
        train_metrics = compute_metrics(y_raw, train_pred, n_feat)
    else:
        from model_inference import build_inference_matrix

        X = build_inference_matrix(df, artifact)
        train_pred = predict_model(artifact["model"], X)
        y_raw = df[TARGET].astype(float).values
        train_metrics = compute_metrics(y_raw, train_pred, len(artifact["selected_features"]))

    config = {
        "model_type": artifact.get("model_type"),
        "feature_config": artifact.get("feature_config"),
        "primary_target": "cnt",
        "selected_features": artifact.get("selected_features"),
        "train_size": len(df),
        "experiment": artifact.get("experiment"),
        "results": {"cnt": {"train": train_metrics, "cv": artifact.get("cv_metrics", {})}},
        "cv_comparison": cv_results_summary or [],
        "weekend_calibration_factor": artifact.get("weekend_calibration_factor", 1.0),
        "winsor_cap": artifact.get("winsor_cap"),
        "ensemble_strategy": artifact.get("strategy"),
        "ensemble_members": artifact.get("member_names", []),
    }
    with open(os.path.join(OUTPUT_DIR, "model_config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, default=str)

    print(f"Modelo guardado: {artifact.get('experiment', artifact.get('model_type'))} -> {model_path}")
    return artifact


def predict_model_batch(df, artifact):
    """Predicción batch soportando modelos simples, duales y two-stage."""
    from model_inference import predict_cnt

    return predict_cnt(df, artifact)


if __name__ == "__main__":
    main()
