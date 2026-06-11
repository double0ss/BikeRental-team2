"""
Análisis de sesgos y valores atípicos antes del entrenamiento.
Genera reportes JSON y gráficos en outputs/.
"""
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

from preprocessing import TARGET, load_and_clean

OUTPUT_DIR = "outputs"
NUMERIC_COLS = ["cnt", "temp", "hum", "windspeed", "leaflets", "casual", "registered"]
CATEGORICAL_COLS = ["season", "yr", "mnth", "holiday", "weekday", "workingday", "weathersit", "price_reduction"]
BIAS_THRESHOLD = 0.35  # desbalance > 65/35 se considera sesgo moderado


def detect_outliers_iqr(series, column, k=1.5):
    clean = series.dropna().astype(float)
    q1, q3 = clean.quantile(0.25), clean.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - k * iqr
    upper = q3 + k * iqr
    mask = (series < lower) | (series > upper)
    return {
        "method": "IQR",
        "column": column,
        "q1": float(q1),
        "q3": float(q3),
        "lower_bound": float(lower),
        "upper_bound": float(upper),
        "count": int(mask.sum()),
        "pct": float(mask.mean() * 100),
        "indices": series[mask].index.tolist(),
        "values": series[mask].tolist(),
    }


def detect_outliers_zscore(series, column, threshold=3.0):
    clean = series.dropna().astype(float)
    if len(clean) < 3:
        return {"method": "zscore", "column": column, "count": 0, "indices": [], "values": []}
    z = np.abs(stats.zscore(clean))
    out_idx = clean.index[z > threshold]
    return {
        "method": "zscore",
        "column": column,
        "threshold": threshold,
        "count": int(len(out_idx)),
        "pct": float(len(out_idx) / len(series) * 100),
        "indices": out_idx.tolist(),
        "values": series.loc[out_idx].tolist(),
    }


def distribution_stats(df, column):
    series = df[column].dropna().astype(float)
    return {
        "mean": float(series.mean()),
        "median": float(series.median()),
        "std": float(series.std()),
        "min": float(series.min()),
        "max": float(series.max()),
        "skewness": float(stats.skew(series)),
        "kurtosis": float(stats.kurtosis(series)),
    }


def categorical_bias(df, column):
    counts = df[column].value_counts(dropna=False).sort_index()
    total = len(df)
    proportions = {str(k): float(v / total) for k, v in counts.items()}
    dominant = counts.idxmax()
    dominant_pct = float(counts.max() / total)
    biased = dominant_pct > (0.5 + BIAS_THRESHOLD / 2)
    return {
        "column": column,
        "counts": {str(k): int(v) for k, v in counts.items()},
        "proportions": proportions,
        "dominant_class": str(dominant),
        "dominant_pct": dominant_pct,
        "has_bias": biased,
    }


def temporal_bias(df):
    monthly = df.groupby("mnth")[TARGET].agg(["mean", "count"]).round(2)
    seasonal = df.groupby("season")[TARGET].agg(["mean", "count"]).round(2)
    by_year = df.groupby("yr")[TARGET].agg(["mean", "count"]).round(2)

    month_range = monthly["mean"].max() - monthly["mean"].min()
    return {
        "cnt_by_month": {int(k): v for k, v in monthly["mean"].to_dict().items()},
        "cnt_by_season": {int(k): float(v) for k, v in seasonal["mean"].to_dict().items()},
        "cnt_by_year": {int(k): float(v) for k, v in by_year["mean"].to_dict().items()},
        "month_demand_range": float(month_range),
        "peak_month": int(monthly["mean"].idxmax()),
        "low_month": int(monthly["mean"].idxmin()),
        "temporal_bias_note": (
            "Demanda muy estacional: picos en verano/otoño y mínimos en invierno."
            if month_range > 3000
            else "Variación estacional moderada."
        ),
    }


def missing_bias(raw_df):
    missing = raw_df.isnull().sum()
    missing = missing[missing > 0]
    return {
        "columns_with_missing": {col: int(n) for col, n in missing.items()},
        "total_rows": len(raw_df),
    }


def analyze_dataset(path, label, train_dates=None):
    raw = pd.read_csv(path, sep=";")
    raw.rename(columns={"price reduction": "price_reduction"}, inplace=True)
    df = load_and_clean(path, train_dates=train_dates)

    outliers = {}
    for col in NUMERIC_COLS:
        if col not in df.columns:
            continue
        outliers[col] = {
            "iqr": detect_outliers_iqr(df[col], col),
            "zscore": detect_outliers_zscore(df[col], col),
            "distribution": distribution_stats(df, col),
        }

    categorical = {col: categorical_bias(df, col) for col in CATEGORICAL_COLS if col in df.columns}

    cnt_extreme = df.nlargest(5, TARGET)[
        ["dteday", TARGET, "temp", "hum", "weathersit", "holiday", "price_reduction", "season"]
    ]
    extreme_rows = cnt_extreme.copy()
    extreme_rows["dteday"] = extreme_rows["dteday"].dt.strftime("%Y-%m-%d")

    return {
        "label": label,
        "path": path,
        "rows": len(df),
        "missing_raw": missing_bias(raw),
        "outliers": outliers,
        "categorical_bias": categorical,
        "temporal_bias": temporal_bias(df),
        "top_cnt_rows": extreme_rows.to_dict(orient="records"),
        "recommendations": build_recommendations(outliers, categorical, temporal_bias(df)),
    }


def build_recommendations(outliers, categorical, temporal):
    recs = []

    cnt_iqr = outliers.get("cnt", {}).get("iqr", {})
    if cnt_iqr.get("count", 0) > 0:
        recs.append(
            f"cnt tiene {cnt_iqr['count']} outliers IQR (máx. {max(cnt_iqr.get('values', [0])):.0f}). "
            "Se aplicará transformación log1p(cnt) para reducir su impacto."
        )

    cnt_skew = outliers.get("cnt", {}).get("distribution", {}).get("skewness", 0)
    if abs(cnt_skew) > 1:
        recs.append(f"cnt presenta asimetría alta (skew={cnt_skew:.2f}); la escala logarítmica es adecuada.")

    ws = categorical.get("weathersit", {})
    if ws.get("dominant_pct", 0) > 0.6:
        recs.append(
            f"Sesgo en weathersit: {ws['dominant_pct']*100:.0f}% de días con clima '{ws['dominant_class']}'."
        )
    if ws.get("counts", {}).get("4", 0) == 0:
        recs.append("weathersit=4 (lluvia fuerte) no aparece en train; el modelo no podrá aprender ese patrón.")

    wd = categorical.get("workingday", {})
    if wd.get("has_bias"):
        recs.append(
            f"Sesgo día laborable: {wd['dominant_pct']*100:.0f}% son días laborables."
        )

    pr = categorical.get("price_reduction", {})
    if pr.get("dominant_pct", 0) > 0.8:
        recs.append(
            f"Promociones raras: solo {100-pr['dominant_pct']*100:.0f}% de días con price_reduction=1."
        )

    recs.append(temporal.get("temporal_bias_note", ""))
    recs.append("Variables continuas se normalizan con StandardScaler antes del entrenamiento.")

    return recs


def save_plots(train_report, test_report=None):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    train_df = load_and_clean(train_report["path"])

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    sns.histplot(train_df[TARGET], bins=40, kde=True, ax=axes[0, 0], color="#3d7aed")
    axes[0, 0].set_title("Distribución de cnt (train)")
    axes[0, 0].axvline(train_df[TARGET].quantile(0.75), color="red", linestyle="--", label="Q3")
    axes[0, 0].legend()

    sns.boxplot(data=train_df[["temp", "hum", "windspeed", "leaflets"]], ax=axes[0, 1])
    axes[0, 1].set_title("Outliers variables meteorológicas")

    cnt_iqr = train_report["outliers"]["cnt"]["iqr"]
    axes[1, 0].scatter(
        range(len(train_df)),
        train_df[TARGET],
        alpha=0.5,
        s=15,
        label="Normal",
    )
    out_idx = cnt_iqr.get("indices", [])
    if out_idx:
        axes[1, 0].scatter(
            out_idx,
            train_df.loc[out_idx, TARGET],
            color="red",
            s=60,
            label=f"Outliers IQR ({len(out_idx)})",
        )
    axes[1, 0].axhline(cnt_iqr["upper_bound"], color="orange", linestyle="--", label="Límite IQR superior")
    axes[1, 0].set_title("Outliers en cnt")
    axes[1, 0].legend(fontsize=8)

    monthly = train_df.groupby("mnth")[TARGET].mean()
    monthly.plot(kind="bar", ax=axes[1, 1], color="#3d7aed")
    axes[1, 1].set_title("Sesgo temporal: cnt promedio por mes")
    axes[1, 1].set_xlabel("Mes")

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "bias_outlier_analysis.png"), dpi=300)
    plt.close()

    cat_rows = []
    for col, info in train_report["categorical_bias"].items():
        cat_rows.append({"variable": col, "dominante": info["dominant_class"], "pct": info["dominant_pct"]})
    cat_df = pd.DataFrame(cat_rows)
    plt.figure(figsize=(10, 4))
    sns.barplot(data=cat_df, x="variable", y="pct", color="#3d7aed")
    plt.axhline(0.5, color="red", linestyle="--", label="50%")
    plt.title("Sesgo categórico: proporción de clase dominante")
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "categorical_bias.png"), dpi=300)
    plt.close()


def run_full_analysis(train_path="data/BikeRentalDaily_train.csv", test_path="tests/BikeRentalDaily_test.csv"):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    train_report = analyze_dataset(train_path, "train")
    train_dates = set(load_and_clean(train_path)["dteday"])
    test_report = analyze_dataset(test_path, "test", train_dates=train_dates)

    report = {
        "train": train_report,
        "test": test_report,
        "summary": {
            "train_cnt_outliers_iqr": train_report["outliers"]["cnt"]["iqr"]["count"],
            "train_cnt_skewness": train_report["outliers"]["cnt"]["distribution"]["skewness"],
            "test_cnt_outliers_iqr": test_report["outliers"]["cnt"]["iqr"]["count"],
            "recommendations": train_report["recommendations"],
        },
    }

    save_plots(train_report, test_report)

    report_path = os.path.join(OUTPUT_DIR, "bias_outlier_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)

    print("=== Análisis de sesgos y outliers ===")
    print(f"Train: {train_report['rows']} filas | cnt outliers IQR: {train_report['outliers']['cnt']['iqr']['count']}")
    print(f"  Skewness cnt: {train_report['outliers']['cnt']['distribution']['skewness']:.2f}")
    print(f"  Pico mensual: mes {train_report['temporal_bias']['peak_month']} | Mínimo: mes {train_report['temporal_bias']['low_month']}")
    print(f"Test:  {test_report['rows']} filas | cnt outliers IQR: {test_report['outliers']['cnt']['iqr']['count']}")
    print("\nSesgos detectados:")
    for col, info in train_report["categorical_bias"].items():
        if info["has_bias"]:
            print(f"  - {col}: clase {info['dominant_class']} domina ({info['dominant_pct']*100:.1f}%)")
    print("\nRecomendaciones:")
    for r in train_report["recommendations"]:
        print(f"  • {r}")
    print(f"\nReporte: {report_path}")
    print(f"Gráficos: {OUTPUT_DIR}/bias_outlier_analysis.png, categorical_bias.png")
    return report


if __name__ == "__main__":
    run_full_analysis()
