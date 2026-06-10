import pandas as pd
import numpy as np
import os
import warnings
from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Optional plotting and model serialization libs — fallbacks allowed
plotting_available = False
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    plotting_available = True
except Exception:
    plt = None
    sns = None

try:
    import joblib
    joblib_available = True
except Exception:
    import pickle as joblib
    joblib_available = False

warnings.filterwarnings('ignore')
if plotting_available:
    plt.rcParams['figure.figsize'] = (12, 6)
if sns is not None:
    try:
        sns.set_theme(style="whitegrid")
    except Exception:
        pass

# 📂 Configuración de rutas
DATA_PATH = "data/BikeRentalDaily_train.csv"
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =============================================================================
# 1️⃣ CARGA Y LIMPIEZA ROBUSTA
# =============================================================================
print("📥 Cargando dataset...")
df = pd.read_csv(DATA_PATH, sep=';')
df.rename(columns={'price reduction': 'price_reduction'}, inplace=True)
df['dteday'] = pd.to_datetime(df['dteday'], format='%d.%m.%Y')

# Completar season por mes (Hemisferio Norte)
def month_to_season(m):
    if m in [1, 2, 3]: return 1
    elif m in [4, 5, 6]: return 2
    elif m in [7, 8, 9]: return 3
    else: return 4
df['season'] = df['season'].fillna(df['dteday'].dt.month.apply(month_to_season))

# Corregir weekday = -1
df.loc[df['weekday'] == -1, 'weekday'] = df.loc[df['weekday'] == -1, 'dteday'].dt.dayofweek

# Limpiar anomalías en hum y windspeed
df['hum'] = df['hum'].clip(upper=100.0, lower=0.0)
df['windspeed'] = df['windspeed'].replace(-1.0, np.nan)

# Imputar numéricos con mediana (robusto a outliers)
num_cols = ['temp', 'hum', 'windspeed', 'leaflets']
df[num_cols] = df[num_cols].fillna(df[num_cols].median())

# =============================================================================
# 2️⃣ FEATURE ENGINEERING AVANZADO
# =============================================================================
print("⚙️ Aplicando Feature Engineering...")

# Codificación cíclica para estacionalidad (evita saltos bruscos Dic->Ene o Dom->Lun)
df['mnth_sin'] = np.sin(2 * np.pi * df['mnth'] / 12)
df['mnth_cos'] = np.cos(2 * np.pi * df['mnth'] / 12)
df['weekday_sin'] = np.sin(2 * np.pi * df['weekday'] / 7)
df['weekday_cos'] = np.cos(2 * np.pi * df['weekday'] / 7)

df['is_weekend'] = df['weekday'].isin([5, 6]).astype(int)

# Eliminar columnas redundantes, temporales o con fuga de datos
cols_to_drop = ['instant', 'atemp', 'casual', 'registered']

# Validar integridad básica
assert df['cnt'].notna().all(), "❌ Valores nulos en 'cnt'"
print(f"✅ Dataset inicialmente preparado. Shape: {df.shape}")

# =============================================================================
# 3️⃣ DIVISIÓN TEMPORAL (CRÍTICO EN SERIES)
# =============================================================================
print("📅 Dividiendo datos cronológicamente...")
df = df.sort_values('dteday').reset_index(drop=True)

# Ahora que ya ordenamos por fecha, podemos eliminar columnas temporales
df.drop(columns=cols_to_drop + ['dteday', 'mnth', 'weekday'], inplace=True, errors='ignore')
print(f"✅ Dataset final preparado. Shape: {df.shape}")
split_idx = int(len(df) * 0.8)

train_df = df.iloc[:split_idx]
test_df  = df.iloc[split_idx:]

X_train = train_df.drop(columns=['cnt'])
y_train = train_df['cnt']
X_test  = test_df.drop(columns=['cnt'])
y_test  = test_df['cnt']

# =============================================================================
# 4️⃣ NORMALIZACIÓN + PIPELINE (Evita Data Leakage)
# =============================================================================
print("📐 Configurando Pipeline con Normalización...")

# Escalar solo variables continuas. Los árboles no lo requieren, pero se aplica por solicitud.
numeric_features = ['temp', 'hum', 'windspeed', 'leaflets']
categorical_features = ['season', 'yr', 'holiday', 'workingday', 'weathersit', 'price_reduction', 
                        'is_weekend', 'mnth_sin', 'mnth_cos', 'weekday_sin', 'weekday_cos']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features)
    ],
    remainder='passthrough'
)

# Modelo Random Forest optimizado
model = RandomForestRegressor(
    n_estimators=500,
    max_depth=20,
    min_samples_split=5,
    min_samples_leaf=2,
    max_features='sqrt',
    random_state=42,
    n_jobs=-1,
    verbose=0
)

pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', model)
])

# =============================================================================
# 5️⃣ ENTRENAMIENTO Y EVALUACIÓN
# =============================================================================
print("🚀 Entrenando modelo...")
# Optimize hyperparameters with TimeSeries-aware CV
param_dist = {
    'regressor__n_estimators': [100, 200, 400, 600],
    'regressor__max_depth': [8, 12, 16, 20, None],
    'regressor__min_samples_split': [2, 5, 8],
    'regressor__min_samples_leaf': [1, 2, 4],
    'regressor__max_features': ['sqrt', 'log2', None]
}

tscv = TimeSeriesSplit(n_splits=4)
search = RandomizedSearchCV(
    pipeline,
    param_distributions=param_dist,
    n_iter=20,
    cv=tscv,
    scoring='neg_mean_squared_error',
    random_state=42,
    n_jobs=-1,
    verbose=1
)

search.fit(X_train, y_train)
pipeline = search.best_estimator_
print(f"🔎 Mejor parámetro encontrado: {search.best_params_}")

def evaluate(y_true, y_pred, split_name):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    print(f"{split_name:8} | MAE: {mae:6.2f} | RMSE: {rmse:6.2f} | R²: {r2:.4f}")
    return mae, rmse, r2

print("\n📊 Resultados de Evaluación:")
evaluate(y_train, pipeline.predict(X_train), "🟦 Train")
evaluate(y_test,  pipeline.predict(X_test), "🟧 Test ")

# =============================================================================
# 6️⃣ VISUALIZACIÓN Y GUARDADO
# =============================================================================
print("📈 Generando gráficas y guardando modelo...")

y_pred_test = pipeline.predict(X_test)

# 6.1 Predicciones vs Realidad
plt.figure()
plt.plot(y_test.values, label='Real', alpha=0.8, linewidth=2)
plt.plot(y_pred_test, label='Predicho', alpha=0.8, linewidth=2)
plt.title("Alquileres Reales vs Predichos (Test Set)")
plt.xlabel("Días (Test)")
plt.ylabel("cnt")
plt.legend()
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/predictions_vs_actual.png", dpi=300)
plt.close()

# 6.2 Importancia de Variables
importances = pipeline.named_steps['regressor'].feature_importances_
# Reconstruir nombres tras ColumnTransformer
feature_names = numeric_features + categorical_features
feat_imp = pd.DataFrame({'Feature': feature_names, 'Importance': importances}).sort_values('Importance', ascending=False)

plt.figure(figsize=(10, 5))
sns.barplot(data=feat_imp.head(10), x='Importance', y='Feature', palette='viridis')
plt.title("🌲 Top 10 Variables Más Importantes (Random Forest)")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/feature_importance.png", dpi=300)
plt.close()

# 6.3 Guardar Pipeline Completo
model_path = f"{OUTPUT_DIR}/bike_sharing_pipeline.pkl"
joblib.dump(pipeline, model_path)
print(f"\n✅ Modelo guardado en: {model_path}")
print("📁 Gráficas guardadas en: outputs/")
print("🎉 ¡Proyecto completado con éxito!")