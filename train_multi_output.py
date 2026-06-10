"""
Multi-output Model Training: casual, registered, cnt
Entrena 3 modelos independientes usando Random Forest
con el dataset limpio y feature engineering optimizado.
"""
import pandas as pd
import numpy as np
import os
import warnings
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

warnings.filterwarnings('ignore')
plt.rcParams['figure.figsize'] = (12, 6)
sns.set_theme(style="whitegrid")

# 📂 Configuración de rutas
DATA_PATH = "data/BikeRentalDaily_train_clean.csv"
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =============================================================================
# 1️⃣ CARGA DEL DATASET LIMPIO
# =============================================================================
print("📥 Cargando dataset limpio...")
df = pd.read_csv(DATA_PATH, sep=';')
df.rename(columns={'price reduction': 'price_reduction'}, inplace=True)
df['dteday'] = pd.to_datetime(df['dteday'], format='%d.%m.%Y')

print(f"✅ Dataset cargado. Shape: {df.shape}")
print(f"   Columnas: {df.columns.tolist()}")
print(f"   Variables objetivo: casual, registered, cnt")

# =============================================================================
# 2️⃣ FEATURE ENGINEERING
# =============================================================================
print("\n⚙️ Aplicando Feature Engineering...")

# Codificación cíclica para estacionalidad
df['mnth_sin'] = np.sin(2 * np.pi * df['mnth'] / 12)
df['mnth_cos'] = np.cos(2 * np.pi * df['mnth'] / 12)
df['weekday_sin'] = np.sin(2 * np.pi * df['weekday'] / 7)
df['weekday_cos'] = np.cos(2 * np.pi * df['weekday'] / 7)
df['is_weekend'] = df['weekday'].isin([5, 6]).astype(int)

# Columnas a eliminar (temporales y redundantes)
cols_to_drop = ['instant', 'dteday', 'mnth', 'weekday']

# Ordenar por fecha para División Temporal
df = df.sort_values('dteday').reset_index(drop=True) if 'dteday' in df.columns else df.reset_index(drop=True)

# Preparar features X
X = df.drop(columns=cols_to_drop + ['casual', 'registered', 'cnt'], errors='ignore')
print(f"✅ Features preparadas. Shape: {X.shape}")
print(f"   Columnas de entrada: {X.columns.tolist()}")

# División temporal 80-20
split_idx = int(len(X) * 0.8)
X_train = X.iloc[:split_idx]
X_test = X.iloc[split_idx:]

# Definir features para preprocessing
numeric_features = ['temp', 'hum', 'windspeed', 'leaflets']
categorical_features = [col for col in X.columns if col not in numeric_features]

# =============================================================================
# 3️⃣ CONFIGURAR PIPELINE
# =============================================================================
print("\n📐 Configurando Pipeline...")

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features)
    ],
    remainder='passthrough'
)

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
# 4️⃣ ENTRENAR 3 MODELOS (uno para cada variable objetivo)
# =============================================================================
targets = ['casual', 'registered', 'cnt']
models = {}
predictions = {}
results = {}

for target in targets:
    print(f"\n{'='*70}")
    print(f"🚀 Entrenando modelo para: {target.upper()}")
    print(f"{'='*70}")
    
    # Preparar datos objetivo
    y_train = df.iloc[:split_idx][target]
    y_test = df.iloc[split_idx:][target]
    
    # Hyperparameter optimization con TimeSeriesSplit
    param_dist = {
        'regressor__n_estimators': [100, 200, 400],
        'regressor__max_depth': [8, 12, 16, 20],
        'regressor__min_samples_split': [2, 5, 8],
        'regressor__min_samples_leaf': [1, 2, 4],
        'regressor__max_features': ['sqrt', 'log2']
    }
    
    tscv = TimeSeriesSplit(n_splits=4)
    search = RandomizedSearchCV(
        pipeline,
        param_distributions=param_dist,
        n_iter=15,
        cv=tscv,
        scoring='neg_mean_squared_error',
        random_state=42,
        n_jobs=-1,
        verbose=0
    )
    
    print(f"   🔍 Buscando mejores hiperparámetros...")
    search.fit(X_train, y_train)
    best_model = search.best_estimator_
    
    print(f"   ✅ Mejores parámetros encontrados:")
    for param, value in search.best_params_.items():
        param_name = param.split('__')[-1]
        print(f"      • {param_name}: {value}")
    
    # Predicciones
    y_pred_train = best_model.predict(X_train)
    y_pred_test = best_model.predict(X_test)
    
    # Evaluación
    mae_train = mean_absolute_error(y_train, y_pred_train)
    rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
    r2_train = r2_score(y_train, y_pred_train)
    
    mae_test = mean_absolute_error(y_test, y_pred_test)
    rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
    r2_test = r2_score(y_test, y_pred_test)
    
    print(f"\n   📊 Resultados de Evaluación:")
    print(f"      🟦 Train: MAE={mae_train:7.2f} | RMSE={rmse_train:7.2f} | R²={r2_train:.4f}")
    print(f"      🟧 Test:  MAE={mae_test:7.2f}  | RMSE={rmse_test:7.2f}  | R²={r2_test:.4f}")
    
    # Guardar modelo y resultados
    models[target] = best_model
    predictions[target] = {'train': y_pred_train, 'test': y_pred_test}
    results[target] = {
        'mae_train': mae_train, 'rmse_train': rmse_train, 'r2_train': r2_train,
        'mae_test': mae_test, 'rmse_test': rmse_test, 'r2_test': r2_test,
        'y_test': y_test, 'y_pred_test': y_pred_test,
        'best_params': search.best_params_
    }

# =============================================================================
# 5️⃣ VISUALIZACIONES COMPARATIVAS
# =============================================================================
print(f"\n{'='*70}")
print("📈 Generando visualizaciones...")
print(f"{'='*70}")

# 5.1 Comparar Predicciones vs Realidad para cada variable
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for idx, target in enumerate(targets):
    y_test = results[target]['y_test']
    y_pred = results[target]['y_pred_test']
    
    axes[idx].plot(y_test.values, label='Real', alpha=0.8, linewidth=2)
    axes[idx].plot(y_pred, label='Predicho', alpha=0.8, linewidth=2)
    axes[idx].set_title(f"{target.upper()}\nR² Test: {results[target]['r2_test']:.4f}")
    axes[idx].set_xlabel("Días (Test Set)")
    axes[idx].set_ylabel("Cantidad")
    axes[idx].legend()
    axes[idx].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/predictions_vs_actual_multi.png", dpi=300, bbox_inches='tight')
plt.close()
print("   ✅ Gráfica: predictions_vs_actual_multi.png")

# 5.2 Feature Importance para cada modelo
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
feature_names = numeric_features + categorical_features

for idx, target in enumerate(targets):
    importances = models[target].named_steps['regressor'].feature_importances_
    feat_imp = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values('Importance', ascending=False).head(10)
    
    sns.barplot(data=feat_imp, x='Importance', y='Feature', ax=axes[idx], palette='viridis')
    axes[idx].set_title(f"Top 10 Features - {target.upper()}")

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/feature_importance_multi.png", dpi=300, bbox_inches='tight')
plt.close()
print("   ✅ Gráfica: feature_importance_multi.png")

# 5.3 Scatter: Predicho vs Real
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for idx, target in enumerate(targets):
    y_test = results[target]['y_test'].values
    y_pred = results[target]['y_pred_test']
    
    axes[idx].scatter(y_test, y_pred, alpha=0.5, s=20)
    lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
    axes[idx].plot(lims, lims, 'r--', lw=2)
    axes[idx].set_xlabel("Real")
    axes[idx].set_ylabel("Predicho")
    axes[idx].set_title(f"{target.upper()}\nMAE: {results[target]['mae_test']:.2f}")
    axes[idx].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/scatter_pred_vs_actual.png", dpi=300, bbox_inches='tight')
plt.close()
print("   ✅ Gráfica: scatter_pred_vs_actual.png")

# =============================================================================
# 6️⃣ GUARDAR MODELOS
# =============================================================================
print(f"\n{'='*70}")
print("💾 Guardando modelos...")
print(f"{'='*70}")

# Guardar cada modelo por separado
for target in targets:
    model_path = f"{OUTPUT_DIR}/model_{target}.pkl"
    joblib.dump(models[target], model_path)
    print(f"   ✅ {model_path}")

# Crear archivo de configuración con metadatos
config = {
    'targets': targets,
    'feature_names': feature_names,
    'numeric_features': numeric_features,
    'categorical_features': categorical_features,
    'train_size': split_idx,
    'test_size': len(X) - split_idx,
    'results': {
        target: {
            'mae_test': float(results[target]['mae_test']),
            'rmse_test': float(results[target]['rmse_test']),
            'r2_test': float(results[target]['r2_test']),
            'mae_train': float(results[target]['mae_train']),
            'rmse_train': float(results[target]['rmse_train']),
            'r2_train': float(results[target]['r2_train'])
        }
        for target in targets
    }
}

import json
config_path = f"{OUTPUT_DIR}/model_config.json"
with open(config_path, 'w') as f:
    json.dump(config, f, indent=2)
print(f"   ✅ {config_path}")

# =============================================================================
# 7️⃣ RESUMEN FINAL
# =============================================================================
print(f"\n{'='*70}")
print("📊 RESUMEN DE MODELOS ENTRENADOS")
print(f"{'='*70}\n")

summary_data = []
for target in targets:
    summary_data.append({
        'Target': target.upper(),
        'MAE Test': f"{results[target]['mae_test']:.2f}",
        'RMSE Test': f"{results[target]['rmse_test']:.2f}",
        'R² Test': f"{results[target]['r2_test']:.4f}",
        'R² Train': f"{results[target]['r2_train']:.4f}"
    })

summary_df = pd.DataFrame(summary_data)
print(summary_df.to_string(index=False))

print(f"\n✅ Dataset limpio utilizado: {DATA_PATH}")
print(f"✅ Modelos guardados en: {OUTPUT_DIR}/")
print(f"✅ Configuración guardada en: {config_path}")
print(f"\n🎉 ¡Entrenamiento multi-objetivo completado con éxito!")
