"""
Predictor multi-output: casual, registered, cnt
Carga 3 modelos independientes entrenados con Random Forest
"""
import sys
import json
import os
try:
    import joblib
except Exception:
    import pickle as _pickle
    class _JoblibFallback:
        @staticmethod
        def load(path):
            with open(path, 'rb') as f:
                return _pickle.load(f)
    joblib = _JoblibFallback()
import pandas as pd
import numpy as np

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'outputs')
MODELS_PATHS = {
    'casual': os.path.join(OUTPUT_DIR, 'model_casual.pkl'),
    'registered': os.path.join(OUTPUT_DIR, 'model_registered.pkl'),
    'cnt': os.path.join(OUTPUT_DIR, 'model_cnt.pkl')
}
CONFIG_PATH = os.path.join(OUTPUT_DIR, 'model_config.json')

# Features expected by the pipelines
numeric_features = ['temp', 'hum', 'windspeed', 'leaflets']
categorical_features = ['season', 'yr', 'holiday', 'workingday', 'weathersit', 'atemp', 'price_reduction', 
                        'is_weekend', 'mnth_sin', 'mnth_cos', 'weekday_sin', 'weekday_cos']

# read JSON from stdin
raw = sys.stdin.read()
if not raw:
    print(json.dumps({'error': 'No input provided'}))
    sys.exit(1)

try:
    payload = json.loads(raw)
except Exception as e:
    print(json.dumps({'error': 'Invalid JSON', 'msg': str(e)}))
    sys.exit(1)

# payload can contain raw mnth and weekday; compute engineered features if present
def compute_engineered(d):
    if 'mnth' in d:
        m = float(d.get('mnth', 1))
        d['mnth_sin'] = np.sin(2 * np.pi * m / 12)
        d['mnth_cos'] = np.cos(2 * np.pi * m / 12)
    if 'weekday' in d:
        w = float(d.get('weekday', 0))
        d['weekday_sin'] = np.sin(2 * np.pi * w / 7)
        d['weekday_cos'] = np.cos(2 * np.pi * w / 7)
        d['is_weekend'] = 1 if int(d.get('weekday', 0)) in [5,6] else 0
    
    # Calcular atemp usando Heat Index (sensación térmica)
    if 'temp' in d and 'hum' in d:
        T = float(d.get('temp', 0))
        RH = float(d.get('hum', 50))
        # Convertir temp normalizado [0,1] a °C. Asumiendo rango 0-34.5°C
        T_celsius = T * 34.5
        # Heat Index formula (NOAA)
        T_F = T_celsius * 1.8 + 32
        HI_F = (-42.379 + 2.04901523*T_F + 10.14333127*RH
                - 0.22475541*T_F*RH - 6.83783e-3*T_F**2
                - 5.481717e-2*RH**2 + 1.22874e-3*T_F**2*RH
                + 8.5282e-4*T_F*RH**2 - 1.99e-6*T_F**2*RH**2)
        HI_celsius = (HI_F - 32) / 1.8
        # Normalizar a [0,1] dividiendo por 50 (máximo realista de temperatura aparente)
        d['atemp'] = max(0, min(1, HI_celsius / 50))
    else:
        d['atemp'] = d.get('atemp', 0.5)
    
    return d

payload = compute_engineered(payload)

# Ensure all features exist (fill with 0/median-like defaults)
row = {}
for f in numeric_features:
    val = payload.get(f, 0.0)
    row[f] = float(val) if val is not None else 0.0
for f in categorical_features:
    val = payload.get(f, 0)
    row[f] = val

X = pd.DataFrame([row])

# Load all 3 models
models = {}
try:
    for target, path in MODELS_PATHS.items():
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}")
        models[target] = joblib.load(path)
except Exception as e:
    print(json.dumps({'error': 'Could not load models', 'msg': str(e)}))
    sys.exit(1)

# Make predictions with all 3 models
try:
    predictions = {}
    feature_importance_by_model = {}
    
    for target in ['casual', 'registered', 'cnt']:
        model = models[target]
        
        # Prediction
        pred = model.predict(X)[0]
        predictions[target] = float(pred)
        
        # Feature importance
        importances = []
        feature_names = numeric_features + categorical_features
        regressor = None
        
        if hasattr(model, 'named_steps') and 'regressor' in model.named_steps:
            regressor = model.named_steps['regressor']
        elif hasattr(model, 'steps') and len(model.steps) > 0:
            regressor = model.steps[-1][1]
        
        if regressor is not None and hasattr(regressor, 'feature_importances_'):
            importances = [
                {'nombre': name, 'valor': float(row[name]), 'importancia': float(imp)}
                for name, imp in zip(feature_names, regressor.feature_importances_)
            ]
            importances = sorted(importances, key=lambda x: x['importancia'], reverse=True)[:10]
        
        feature_importance_by_model[target] = importances
    
    # Prepare output
    out = {
        'predicciones': {
            'casual': predictions['casual'],
            'registered': predictions['registered'],
            'cnt': predictions['cnt']
        },
        'entrada': row,
        'desglose': {
            'casual': feature_importance_by_model['casual'],
            'registered': feature_importance_by_model['registered'],
            'cnt': feature_importance_by_model['cnt']
        }
    }
    
    print(json.dumps(out))
    
except Exception as e:
    print(json.dumps({'error': 'Prediction failed', 'msg': str(e)}))
    sys.exit(1)
