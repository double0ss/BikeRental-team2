"""
Preprocesamiento unificado para entrenamiento, evaluación y predicción de cnt.
Especificación season: 1=invierno, 2=primavera, 3=verano, 4=otoño.
"""
import numpy as np
import pandas as pd

TARGET = "cnt"
EXCLUDE_COLS = ["instant", "dteday", "casual", "registered", "atemp", "cnt"]

NUMERIC_FEATURES = [
    "temp",
    "hum",
    "windspeed",
    "leaflets",
    "yr",
    "mnth",
    "holiday",
    "weekday",
    "workingday",
    "price_reduction",
]
CATEGORICAL_FEATURES = ["season", "weathersit"]
DUMMY_PREFIXES = ("season_", "weathersit_")
# Compatibilidad con código legado; usar get_scale_columns() para normalización completa
SCALE_COLUMNS = ["temp", "hum", "windspeed", "leaflets"]
CYCLICAL_FEATURES = ["mnth_sin", "mnth_cos", "weekday_sin", "weekday_cos"]


def month_to_season(month):
    """Mapeo alineado con el dataset real (hemisferio norte)."""
    if month in (12, 1, 2, 3):
        return 1
    if month in (4, 5, 6):
        return 2
    if month in (7, 8, 9):
        return 3
    return 4


def extract_train_stats(df):
    """Estadísticas de imputación calculadas solo sobre train."""
    return {
        "medians": {
            "hum": float(df["hum"].median()),
            "windspeed": float(df["windspeed"].median()),
        }
    }


def winsorize_target(df, percentile=99):
    """Recorta cnt en train al percentil indicado para reducir influencia de outliers."""
    work = df.copy()
    cap = float(np.percentile(work[TARGET].astype(float), percentile))
    work[TARGET] = work[TARGET].astype(float).clip(upper=cap)
    return work, cap


def load_and_clean(path, train_dates=None, train_stats=None):
    df = pd.read_csv(path, sep=";")
    df.rename(columns={"price reduction": "price_reduction"}, inplace=True)
    df["dteday"] = pd.to_datetime(df["dteday"], format="%d.%m.%Y")

    if train_dates is not None:
        df = df[~df["dteday"].isin(train_dates)].copy()

    df["season"] = df["season"].fillna(df["dteday"].dt.month.map(month_to_season))
    invalid_weekday = df["weekday"] == -1
    if invalid_weekday.any():
        df.loc[invalid_weekday, "weekday"] = df.loc[invalid_weekday, "dteday"].dt.dayofweek
        df.loc[invalid_weekday, "weekday"] = (df.loc[invalid_weekday, "weekday"] + 1) % 7

    df["hum"] = pd.to_numeric(df["hum"], errors="coerce").clip(0, 100)
    df["windspeed"] = pd.to_numeric(df["windspeed"], errors="coerce").replace(-1.0, np.nan)

    medians = (train_stats or {}).get("medians", {})
    for col in ["hum", "windspeed"]:
        if df[col].isna().any():
            fill_value = medians.get(col, df[col].median())
            df[col] = df[col].fillna(fill_value)

    for col in ["season", "weathersit", "holiday", "workingday", "price_reduction", "yr", "mnth", "weekday"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    df = df.sort_values("dteday").reset_index(drop=True)
    return df


def add_engineered_features(df, feature_config=None):
    feature_config = feature_config or {}
    work = df.copy()
    work["mnth_sin"] = np.sin(2 * np.pi * work["mnth"] / 12)
    work["mnth_cos"] = np.cos(2 * np.pi * work["mnth"] / 12)
    work["weekday_sin"] = np.sin(2 * np.pi * work["weekday"] / 7)
    work["weekday_cos"] = np.cos(2 * np.pi * work["weekday"] / 7)
    work["temp_x_weathersit"] = work["temp"] * work["weathersit"]

    if feature_config.get("weekend_features", False):
        work["is_weekend"] = (1 - work["workingday"]).astype(float)
        work["workingday_x_temp"] = work["workingday"] * work["temp"]

    if feature_config.get("heat_index", False):
        # Índice de calor simplificado (evita usar atemp crudo)
        work["heat_index"] = work["temp"] + 0.33 * work["hum"]

    return work


def build_features(df, feature_config=None):
    feature_config = feature_config or {
        "cyclical": True,
        "interaction": True,
        "leaflets": True,
    }

    work = add_engineered_features(df, feature_config)
    base_numeric = [c for c in NUMERIC_FEATURES if c in work.columns]
    if not feature_config.get("leaflets", True) and "leaflets" in base_numeric:
        base_numeric.remove("leaflets")

    cols = base_numeric + CATEGORICAL_FEATURES
    if feature_config.get("cyclical", True):
        cols += CYCLICAL_FEATURES
    if feature_config.get("interaction", True):
        cols.append("temp_x_weathersit")
    if feature_config.get("weekend_features", False):
        cols += ["is_weekend", "workingday_x_temp"]
    if feature_config.get("heat_index", False):
        cols.append("heat_index")

    work = work[cols].copy()
    work = pd.get_dummies(work, columns=CATEGORICAL_FEATURES, drop_first=True)
    return work.astype(float)


def get_dummy_columns(feature_matrix):
    return [c for c in feature_matrix.columns if c.startswith(DUMMY_PREFIXES)]


def get_scale_columns(feature_matrix):
    """Columnas continuas a normalizar (todas excepto dummies one-hot)."""
    return [c for c in feature_matrix.columns if not c.startswith(DUMMY_PREFIXES)]


def apply_scaling(frame, scale_cols, scaler=None, fit=False):
    """Aplica StandardScaler a las columnas indicadas."""
    from sklearn.preprocessing import StandardScaler

    result = frame.copy()
    if not scale_cols:
        return result, scaler

    sc = scaler or StandardScaler()
    cols = [c for c in scale_cols if c in result.columns]
    if not cols:
        return result, sc

    values = result[cols].to_numpy(dtype=float)
    if fit:
        result[cols] = sc.fit_transform(values)
    else:
        result[cols] = sc.transform(values)
    return result, sc


def get_feature_columns(feature_config=None):
    dummy_df = build_features(
        pd.DataFrame(
            {
                "temp": [20.0],
                "hum": [50.0],
                "windspeed": [0.1],
                "leaflets": [500],
                "yr": [1],
                "mnth": [6],
                "holiday": [0],
                "weekday": [1],
                "workingday": [1],
                "price_reduction": [0],
                "season": [2],
                "weathersit": [1],
            }
        ),
        feature_config=feature_config,
    )
    return dummy_df.columns.tolist()
