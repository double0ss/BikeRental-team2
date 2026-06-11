"""
Entrenamiento del modelo de regresión lineal múltiple para predecir cnt.
Metodología: limpieza, correlaciones, VIF, RFE, validación de supuestos.
"""
import json
import os
import warnings

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.feature_selection import RFECV
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import durbin_watson
import statsmodels.api as sm

warnings.filterwarnings("ignore")
plt.rcParams["figure.figsize"] = (12, 6)
sns.set_theme(style="whitegrid")

DATA_PATH = "data/BikeRentalDaily_train.csv"
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

TARGET = "cnt"
NUMERIC_FEATURES = ["temp", "hum", "windspeed", "leaflets", "yr", "mnth", "holiday", "weekday", "workingday", "price_reduction"]
CATEGORICAL_FEATURES = ["season", "weathersit"]
BASE_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def month_to_season(month):
    if month in (12, 1, 2):
        return 4
    if month in (3, 4, 5):
        return 1
    if month in (6, 7, 8):
        return 2
    return 3


def load_and_clean(path):
    df = pd.read_csv(path, sep=";")
    df.rename(columns={"price reduction": "price_reduction"}, inplace=True)
    df["dteday"] = pd.to_datetime(df["dteday"], format="%d.%m.%Y")

    df["season"] = df["season"].fillna(df["dteday"].dt.month.map(month_to_season))
    df.loc[df["weekday"] == -1, "weekday"] = df.loc[df["weekday"] == -1, "dteday"].dt.dayofweek

    df["hum"] = df["hum"].clip(0, 100)
    df["windspeed"] = df["windspeed"].replace(-1.0, np.nan)

    for col in ["hum", "windspeed"]:
        df[col] = df[col].fillna(df[col].median())

    df["season"] = df["season"].astype(int)
    df["weathersit"] = df["weathersit"].astype(int)
    df = df.sort_values("dteday").reset_index(drop=True)
    return df


def build_feature_matrix(df):
    work = df[BASE_FEATURES].copy()
    work["temp_x_weathersit"] = work["temp"] * work["weathersit"]
    work = pd.get_dummies(work, columns=CATEGORICAL_FEATURES, drop_first=True)
    return work


def adjusted_r2(r2, n, p):
    if n <= p + 1:
        return r2
    return 1 - (1 - r2) * (n - 1) / (n - p - 1)


def compute_vif(X):
    vif_data = pd.DataFrame()
    vif_data["variable"] = X.columns
    vif_data["vif"] = [
        variance_inflation_factor(X.values.astype(float), i) for i in range(X.shape[1])
    ]
    return vif_data.sort_values("vif", ascending=False)


def drop_high_vif(X, threshold=10.0):
    current = X.copy()
    dropped = []
    while current.shape[1] > 1:
        vif_df = compute_vif(current)
        max_vif = vif_df.iloc[0]
        if max_vif["vif"] <= threshold:
            break
        dropped.append(max_vif["variable"])
        current = current.drop(columns=[max_vif["variable"]])
    return current, dropped


def evaluate_split(y_true, y_pred, split_name, n_features):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    adj = adjusted_r2(r2, len(y_true), n_features)
    print(f"{split_name:8} | MAE: {mae:8.2f} | RMSE: {rmse:8.2f} | R²: {r2:.4f} | R² adj: {adj:.4f}")
    return {"mae": mae, "rmse": rmse, "r2": r2, "r2_adj": adj}


def save_correlation_plots(df):
    continuous = ["cnt", "temp", "hum", "windspeed"]
    pearson = df[continuous].corr(method="pearson")
    plt.figure(figsize=(8, 6))
    sns.heatmap(pearson, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1)
    plt.title("Correlación de Pearson (variables continuas)")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/chart_correlation.png", dpi=300)
    plt.close()

    ordinal = ["cnt", "season", "weathersit", "weekday", "workingday"]
    spearman = df[ordinal].corr(method="spearman")
    plt.figure(figsize=(8, 6))
    sns.heatmap(spearman, annot=True, fmt=".2f", cmap="viridis", vmin=-1, vmax=1)
    plt.title("Correlación de Spearman (variables ordinales)")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/chart_correlation_spearman.png", dpi=300)
    plt.close()


def save_eda_plots(df):
    plt.figure()
    sns.histplot(df["cnt"], bins=30, kde=True)
    plt.title("Distribución de cnt")
    plt.xlabel("cnt")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/chart_cnt_hist.png", dpi=300)
    plt.close()

    plt.figure()
    sns.boxplot(y=df["cnt"])
    plt.title("Boxplot de cnt")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/chart_cnt_box.png", dpi=300)
    plt.close()

    plt.figure(figsize=(10, 5))
    sns.boxplot(data=df[["temp", "hum", "windspeed"]])
    plt.title("Boxplot variables numéricas")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/chart_numeric_box.png", dpi=300)
    plt.close()

    plt.figure()
    df.groupby("yr")["cnt"].mean().plot(kind="bar", color="#3d7aed")
    plt.title("cnt promedio por año")
    plt.xlabel("yr")
    plt.ylabel("cnt promedio")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/chart_cnt_year.png", dpi=300)
    plt.close()

    plt.figure()
    df.groupby("weathersit")["cnt"].mean().plot(kind="bar", color="#3d7aed")
    plt.title("cnt promedio por clima")
    plt.xlabel("weathersit")
    plt.ylabel("cnt promedio")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/chart_cnt_weather.png", dpi=300)
    plt.close()


def save_diagnostics(y_true, y_pred, residuals, split_name):
    fitted = y_pred

    plt.figure()
    plt.scatter(fitted, residuals, alpha=0.6)
    plt.axhline(0, color="red", linestyle="--")
    plt.xlabel("Valores ajustados")
    plt.ylabel("Residuos")
    plt.title(f"Homocedasticidad ({split_name})")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/residuals_vs_fitted_{split_name}.png", dpi=300)
    plt.close()

    plt.figure()
    stats.probplot(residuals, dist="norm", plot=plt)
    plt.title(f"Q-Q plot de residuos ({split_name})")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/qq_plot_{split_name}.png", dpi=300)
    plt.close()


def main():
    print("Cargando y limpiando datos...")
    df = load_and_clean(DATA_PATH)
    print(f"Dataset preparado: {df.shape[0]} filas")

    save_correlation_plots(df)
    save_eda_plots(df)

    pearson_cnt = df[["cnt", "temp", "hum", "windspeed"]].corr(method="pearson")["cnt"]
    spearman_cnt = df[["cnt", "season", "weathersit", "weekday", "workingday"]].corr(method="spearman")["cnt"]
    print("\nCorrelaciones con cnt:")
    print("Pearson:", pearson_cnt.drop("cnt").round(3).to_dict())
    print("Spearman:", spearman_cnt.drop("cnt").round(3).to_dict())

    X = build_feature_matrix(df).astype(float)
    y_raw = df[TARGET].astype(float)
    y = np.log1p(y_raw)

    split_idx = int(len(df) * 0.8)
    X_train_raw = X.iloc[:split_idx].copy()
    X_test_raw = X.iloc[split_idx:].copy()
    y_train = y.iloc[:split_idx]
    y_test_raw = y_raw.iloc[split_idx:]

    # Escalar solo variables meteorológicas; yr queda en años desde 2011 para extrapolar
    scale_cols = [c for c in ["temp", "hum", "windspeed", "leaflets", "temp_x_weathersit"] if c in X.columns]
    scaler = StandardScaler()
    X_train = X_train_raw.copy()
    X_test = X_test_raw.copy()
    X_train[scale_cols] = scaler.fit_transform(X_train[scale_cols])
    X_test[scale_cols] = scaler.transform(X_test[scale_cols])

    print("\nCalculando VIF...")
    vif_initial = compute_vif(X_train)
    vif_initial.to_csv(f"{OUTPUT_DIR}/vif_initial.csv", index=False)
    print(vif_initial.head(10).to_string(index=False))

    kept_cols = drop_high_vif(X_train, threshold=10.0)[0].columns.tolist()
    dropped_vif = [c for c in X_train.columns if c not in kept_cols]
    print(f"Variables eliminadas por VIF > 10: {dropped_vif}")

    X_train_vif = X_train[kept_cols]
    X_test_vif = X_test[kept_cols]

    print("\nSelección de características con RFECV...")
    tscv = TimeSeriesSplit(n_splits=4)
    rfecv = RFECV(
        estimator=LinearRegression(),
        step=1,
        cv=tscv,
        scoring="neg_root_mean_squared_error",
        min_features_to_select=3,
        n_jobs=-1,
    )
    rfecv.fit(X_train_vif, y_train)
    selected_features = X_train_vif.columns[rfecv.support_].tolist()
    print(f"Variables seleccionadas por RFE ({len(selected_features)}): {selected_features}")

    X_train_final = X_train_vif[selected_features].astype(float)
    X_test_final = X_test_vif[selected_features].astype(float)

    model = LinearRegression()
    model.fit(X_train_final, y_train)

    y_pred_train_log = model.predict(X_train_final)
    y_pred_test_log = model.predict(X_test_final)
    y_pred_train = np.expm1(y_pred_train_log)
    y_pred_test = np.expm1(y_pred_test_log)
    y_train_raw = np.expm1(y_train)
    y_test_raw_values = y_test_raw.values

    print("\nResultados (escala original de cnt):")
    train_metrics = evaluate_split(y_train_raw, y_pred_train, "Train", len(selected_features))
    test_metrics = evaluate_split(y_test_raw_values, y_pred_test, "Test ", len(selected_features))

    residuals_test = y_test_raw_values - y_pred_test
    save_diagnostics(y_test_raw_values, y_pred_test, residuals_test, "test")

    dw = durbin_watson(residuals_test)
    shapiro_stat, shapiro_p = stats.shapiro(residuals_test[: min(500, len(residuals_test))])
    print(f"\nValidación de supuestos (test):")
    print(f"  Durbin-Watson: {dw:.4f} (≈2 indica independencia)")
    print(f"  Shapiro-Wilk: stat={shapiro_stat:.4f}, p={shapiro_p:.4g}")

    X_sm = sm.add_constant(X_train_final.values.astype(float))
    ols = sm.OLS(y_train.values.astype(float), X_sm).fit()
    with open(f"{OUTPUT_DIR}/ols_summary.txt", "w", encoding="utf-8") as f:
        f.write(ols.summary().as_text())

    coef_df = pd.DataFrame({
        "feature": ["intercept"] + selected_features,
        "coefficient": [model.intercept_] + list(model.coef_),
        "p_value": list(ols.pvalues),
    })
    coef_df.to_csv(f"{OUTPUT_DIR}/coefficients.csv", index=False)

    plt.figure()
    plt.plot(y_test_raw_values, label="Real", linewidth=2)
    plt.plot(y_pred_test, label="Predicho", linewidth=2)
    plt.title("cnt real vs predicho (test)")
    plt.xlabel("Días")
    plt.ylabel("cnt")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/predictions_vs_actual.png", dpi=300)
    plt.close()

    coef_plot = coef_df[coef_df["feature"] != "intercept"].copy()
    coef_plot["abs_coef"] = coef_plot["coefficient"].abs()
    coef_plot = coef_plot.sort_values("abs_coef", ascending=False).head(10)
    plt.figure(figsize=(10, 5))
    sns.barplot(data=coef_plot, x="coefficient", y="feature", palette="viridis")
    plt.title("Coeficientes del modelo lineal (top 10)")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/feature_importance.png", dpi=300)
    plt.close()

    casual_ratio = (df.iloc[:split_idx]["casual"] / df.iloc[:split_idx]["cnt"]).median()
    registered_ratio = (df.iloc[:split_idx]["registered"] / df.iloc[:split_idx]["cnt"]).median()

    artifact = {
        "model_type": "linear_regression",
        "target": TARGET,
        "use_log_transform": True,
        "selected_features": selected_features,
        "dropped_vif": dropped_vif,
        "scale_columns": scale_cols,
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "intercept": float(model.intercept_),
        "coefficients": {f: float(c) for f, c in zip(selected_features, model.coef_)},
        "dummy_columns": [c for c in X.columns if c.startswith(("season_", "weathersit_"))],
        "base_features": BASE_FEATURES,
        "base_year": 2011,
        "casual_ratio": float(casual_ratio),
        "registered_ratio": float(registered_ratio),
        "metrics": {"train": train_metrics, "test": test_metrics},
        "assumptions": {"durbin_watson": float(dw), "shapiro_stat": float(shapiro_stat), "shapiro_p": float(shapiro_p)},
        "correlations": {
            "pearson": pearson_cnt.drop("cnt").round(4).to_dict(),
            "spearman": spearman_cnt.drop("cnt").round(4).to_dict(),
        },
    }

    model_path = f"{OUTPUT_DIR}/model_cnt_linear.pkl"
    joblib.dump(artifact, model_path)

    config_path = f"{OUTPUT_DIR}/model_config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "model_type": "linear_regression",
                "targets": ["casual", "registered", "cnt"],
                "primary_target": "cnt",
                "selected_features": selected_features,
                "train_size": split_idx,
                "test_size": len(df) - split_idx,
                "results": {
                    "cnt": {
                        "mae_test": test_metrics["mae"],
                        "rmse_test": test_metrics["rmse"],
                        "r2_test": test_metrics["r2"],
                        "r2_adj_test": test_metrics["r2_adj"],
                        "mae_train": train_metrics["mae"],
                        "rmse_train": train_metrics["rmse"],
                        "r2_train": train_metrics["r2"],
                        "r2_adj_train": train_metrics["r2_adj"],
                    }
                },
            },
            f,
            indent=2,
        )

    print(f"\nModelo guardado en: {model_path}")
    print(f"Configuración guardada en: {config_path}")
    print("Gráficos y reportes guardados en outputs/")
    print("Entrenamiento completado.")


if __name__ == "__main__":
    main()
