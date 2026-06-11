"""
Evaluación del modelo en el conjunto externo tests/BikeRentalDaily_test.csv.
"""
import json
import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from model_inference import predict_cnt
from preprocessing import TARGET, load_and_clean
from train_model import adjusted_r2, compute_metrics

DATA_PATH = "data/BikeRentalDaily_train.csv"
TEST_PATH = "tests/BikeRentalDaily_test.csv"
MODEL_PATH = "outputs/model_cnt.pkl"
OUTPUT_DIR = "outputs"
SEGMENT_COLS = ["season", "weathersit", "workingday", "yr"]


def compute_robust_metrics(y_true, y_pred):
    """Métricas robustas a outliers: MAE mediana y MAPE mediana."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.maximum(np.asarray(y_pred, dtype=float), 0)
    errors = np.abs(y_true - y_pred)
    pct_errors = errors / np.maximum(y_true, 1) * 100
    return {
        "mae_median": float(np.median(errors)),
        "mape_median": float(np.median(pct_errors)),
    }


def composite_from_predictions(df, y_true, y_pred):
    """Score compuesto a partir de predicciones y dataframe con segmentos."""
    global_metrics = compute_metrics(y_true, y_pred)
    segments = {}
    for col in SEGMENT_COLS:
        segments[col] = segment_metrics(df, y_true, y_pred, col)

    high_demand = np.asarray(y_true, dtype=float) > 15000
    if high_demand.sum() > 0:
        outlier_m = compute_metrics(y_true[high_demand], y_pred[high_demand])
        rmse_outlier_pct = outlier_m["rmse"] / max(float(np.mean(y_true[high_demand])), 1) * 100
    else:
        rmse_outlier_pct = 0.0

    score_info = composite_score(global_metrics, segments)
    score_info["rmse_outlier_pct"] = rmse_outlier_pct
    score_info["score"] = (
        0.4 * score_info["mae_global"]
        + 0.3 * score_info["mae_weekend"]
        + 0.2 * score_info["mae_yr2012"]
        + 0.1 * rmse_outlier_pct
    )
    return score_info, global_metrics, segments


def composite_score(global_metrics, segments, outlier_threshold=15000):
    """Score compuesto para selección de candidatos en refinamiento."""
    mae_global = global_metrics["mae"]

    mae_weekend = mae_global
    mae_yr2012 = mae_global
    for item in segments.get("workingday", []):
        if item["value"] == 0:
            mae_weekend = item["mae"]
    for item in segments.get("yr", []):
        if item["value"] == 1:
            mae_yr2012 = item["mae"]

    rmse_outlier_pct = 0.0
    return {
        "score": (
            0.4 * mae_global
            + 0.3 * mae_weekend
            + 0.2 * mae_yr2012
            + 0.1 * rmse_outlier_pct
        ),
        "mae_global": mae_global,
        "mae_weekend": mae_weekend,
        "mae_yr2012": mae_yr2012,
        "rmse_outlier_pct": rmse_outlier_pct,
    }


def evaluate_artifact(artifact, save_outputs=True, model_path=MODEL_PATH):
    """Evalúa un artifact en test externo; retorna métricas y score compuesto."""
    train_df = load_and_clean(DATA_PATH)
    train_dates = set(train_df["dteday"])
    train_stats = artifact.get("train_stats")
    test_df = load_and_clean(TEST_PATH, train_dates=train_dates, train_stats=train_stats)

    y_true = test_df[TARGET].astype(float).values
    y_pred = predict_cnt(test_df, artifact)
    global_metrics = compute_metrics(y_true, y_pred, len(artifact.get("selected_features", [])))
    global_metrics["r2_adj"] = adjusted_r2(
        global_metrics["r2"], len(y_true), len(artifact.get("selected_features", []))
    )
    global_metrics.update(compute_robust_metrics(y_true, y_pred))

    segments = {}
    for col in SEGMENT_COLS:
        segments[col] = segment_metrics(test_df, y_true, y_pred, col)

    # Error en días de demanda extrema (cnt > umbral)
    high_demand = y_true > 15000
    if high_demand.sum() > 0:
        outlier_m = compute_metrics(y_true[high_demand], y_pred[high_demand])
        global_metrics["outlier_mae"] = outlier_m["mae"]
        global_metrics["outlier_count"] = int(high_demand.sum())
        rmse_outlier_pct = outlier_m["rmse"] / max(float(np.mean(y_true[high_demand])), 1) * 100
    else:
        rmse_outlier_pct = 0.0

    score_info = composite_score(global_metrics, segments)
    score_info["rmse_outlier_pct"] = rmse_outlier_pct
    score_info["score"] = (
        0.4 * score_info["mae_global"]
        + 0.3 * score_info["mae_weekend"]
        + 0.2 * score_info["mae_yr2012"]
        + 0.1 * rmse_outlier_pct
    )

    # Métricas sin el outlier más extremo del test
    if len(y_true) > 1:
        max_idx = int(np.argmax(y_true))
        mask = np.ones(len(y_true), dtype=bool)
        mask[max_idx] = False
        without_outlier = compute_metrics(y_true[mask], y_pred[mask])
        global_metrics["without_top_outlier"] = without_outlier

    train_pred = predict_cnt(train_df, artifact)
    train_y = train_df[TARGET].astype(float).values
    train_metrics = compute_metrics(train_y, train_pred, len(artifact.get("selected_features", [])))
    global_metrics["r2_gap_train_test"] = train_metrics["r2"] - global_metrics["r2"]

    report = {
        "model_type": artifact.get("model_type"),
        "feature_config": artifact.get("feature_config"),
        "experiment": artifact.get("experiment"),
        "test_size": len(test_df),
        "excluded_overlap_dates": 132 - len(test_df),
        "global": global_metrics,
        "segments": segments,
        "composite_score": score_info,
        "cv_metrics": artifact.get("cv_metrics", {}),
        "train_metrics": train_metrics,
    }

    if save_outputs:
        predictions_df = test_df[
            ["dteday", "season", "yr", "mnth", "weekday", "workingday", "weathersit", TARGET]
        ].copy()
        predictions_df["predicho"] = y_pred
        predictions_df["error"] = predictions_df[TARGET] - predictions_df["predicho"]
        predictions_df["error_abs"] = predictions_df["error"].abs()
        predictions_df.to_csv(os.path.join(OUTPUT_DIR, "test_predictions.csv"), index=False)

        with open(os.path.join(OUTPUT_DIR, "test_metrics.json"), "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

    return report


def segment_metrics(df, y_true, y_pred, column):
    rows = []
    for value in sorted(df[column].dropna().unique()):
        mask = df[column] == value
        if mask.sum() < 2:
            continue
        m = compute_metrics(y_true[mask], y_pred[mask])
        rows.append({"segment": column, "value": int(value), "count": int(mask.sum()), **m})
    return rows


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    artifact = joblib.load(MODEL_PATH)
    report = evaluate_artifact(artifact, save_outputs=True)

    train_df = load_and_clean(DATA_PATH)
    train_dates = set(train_df["dteday"])
    test_df = load_and_clean(
        TEST_PATH, train_dates=train_dates, train_stats=artifact.get("train_stats")
    )
    y_true = test_df[TARGET].astype(float).values
    y_pred = predict_cnt(test_df, artifact)
    global_metrics = report["global"]
    segments = report["segments"]

    plt.figure(figsize=(10, 5))
    plt.plot(y_true, label="Real", linewidth=2)
    plt.plot(y_pred, label="Predicho", linewidth=2)
    plt.title("Test externo: cnt real vs predicho")
    plt.xlabel("Índice")
    plt.ylabel("cnt")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "test_predictions_vs_actual.png"), dpi=300)
    plt.close()

    seg_rows = []
    for col, items in segments.items():
        for item in items:
            seg_rows.append({**item, "segment_name": col})
    seg_df = pd.DataFrame(seg_rows)

    if not seg_df.empty:
        plt.figure(figsize=(12, 6))
        seg_df["label"] = seg_df["segment_name"] + "=" + seg_df["value"].astype(str)
        sns.barplot(data=seg_df, x="label", y="rmse", color="#3d7aed")
        plt.title("RMSE por segmento (test externo)")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "test_error_by_segment.png"), dpi=300)
        plt.close()

    print("=== Evaluación test externo ===")
    print(f"Filas evaluadas: {len(test_df)}")
    print(
        f"MAE={global_metrics['mae']:.2f} | RMSE={global_metrics['rmse']:.2f} | "
        f"R2={global_metrics['r2']:.4f} | R2_adj={global_metrics['r2_adj']:.4f}"
    )
    print(
        f"MAE mediana={global_metrics.get('mae_median', 0):.2f} | "
        f"MAPE mediana={global_metrics.get('mape_median', 0):.1f}%"
    )
    cs = report.get("composite_score", {})
    print(f"Score compuesto={cs.get('score', 0):.2f}")
    print(f"Reporte: {OUTPUT_DIR}/test_metrics.json")


if __name__ == "__main__":
    main()
