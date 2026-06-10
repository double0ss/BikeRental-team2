import pandas as pd
import numpy as np
import os
import warnings
import pickle
from datetime import datetime

warnings.filterwarnings('ignore')

DATA_PATH = "data/BikeRentalDaily_train.csv"
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("📥 Cargando dataset con pandas...")
df = pd.read_csv(DATA_PATH, sep=';')
df.rename(columns={'price reduction': 'price_reduction'}, inplace=True)
# parse date
try:
    df['dteday'] = pd.to_datetime(df['dteday'], format='%d.%m.%Y')
except Exception:
    df['dteday'] = pd.to_datetime(df['dteday'])

# fill season from month
def month_to_season(m):
    if m in [1,2,3]: return 1
    if m in [4,5,6]: return 2
    if m in [7,8,9]: return 3
    return 4

if 'season' in df.columns:
    df['season'] = df['season'].fillna(df['dteday'].dt.month.apply(month_to_season))

# fix weekday -1
if 'weekday' in df.columns:
    df.loc[df['weekday'] == -1, 'weekday'] = df.loc[df['weekday'] == -1, 'dteday'].dt.dayofweek

# clip and replace
if 'hum' in df.columns:
    df['hum'] = df['hum'].clip(0,100)
if 'windspeed' in df.columns:
    df['windspeed'] = df['windspeed'].replace(-1.0, np.nan)

num_cols = [c for c in ['temp','hum','windspeed','leaflets'] if c in df.columns]
for c in num_cols:
    df[c] = df[c].fillna(df[c].median())

# cyclic encodings
if 'mnth' in df.columns:
    df['mnth_sin'] = np.sin(2 * np.pi * df['mnth'] / 12)
    df['mnth_cos'] = np.cos(2 * np.pi * df['mnth'] / 12)
else:
    df['mnth_sin'] = 0
    df['mnth_cos'] = 0

if 'weekday' in df.columns:
    df['weekday_sin'] = np.sin(2 * np.pi * df['weekday'] / 7)
    df['weekday_cos'] = np.cos(2 * np.pi * df['weekday'] / 7)
else:
    df['weekday_sin'] = 0
    df['weekday_cos'] = 0

if 'weekday' in df.columns:
    df['is_weekend'] = df['weekday'].isin([5,6]).astype(int)
else:
    df['is_weekend'] = 0

# choose features available
features = []
for c in ['temp','hum','windspeed','leaflets','season','yr','holiday','workingday','weathersit','price_reduction','is_weekend','mnth_sin','mnth_cos','weekday_sin','weekday_cos']:
    if c in df.columns:
        features.append(c)

print(f"Usando features: {features}")

# drop rows with missing target
df = df[df['cnt'].notna()].copy()

df = df.sort_values('dteday').reset_index(drop=True)

X = df[features].astype(float).values
y = df['cnt'].astype(float).values

# train/test split
split = int(0.8 * len(y))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# scale numeric columns (first up to 4 are numeric)
num_idx = [i for i,f in enumerate(features) if f in ['temp','hum','windspeed','leaflets']]
means = X_train[:, num_idx].mean(axis=0)
stds = X_train[:, num_idx].std(axis=0, ddof=0)
stds[stds==0] = 1.0
X_train[:, num_idx] = (X_train[:, num_idx] - means) / stds
X_test[:, num_idx] = (X_test[:, num_idx] - means) / stds

# add bias column
X_train_aug = np.hstack([np.ones((X_train.shape[0],1)), X_train])
X_test_aug = np.hstack([np.ones((X_test.shape[0],1)), X_test])

# closed-form ridge regression
alpha = 1.0
A = X_train_aug.T @ X_train_aug + alpha * np.eye(X_train_aug.shape[1])
B = X_train_aug.T @ y_train
w = np.linalg.solve(A, B)

# predictions
y_pred_train = X_train_aug @ w
y_pred_test = X_test_aug @ w

# metrics
from math import sqrt

def mae(a,b): return np.mean(np.abs(a-b))
def rmse(a,b): return sqrt(np.mean((a-b)**2))
def r2(a,b): return 1 - np.sum((a-b)**2)/np.sum((a-np.mean(a))**2)

print('📊 Métricas:')
print('Train | MAE: {:.2f} | RMSE: {:.2f} | R2: {:.4f}'.format(mae(y_train,y_pred_train), rmse(y_train,y_pred_train), r2(y_train,y_pred_train)))
print('Test  | MAE: {:.2f} | RMSE: {:.2f} | R2: {:.4f}'.format(mae(y_test,y_pred_test), rmse(y_test,y_pred_test), r2(y_test,y_pred_test)))

# save model params
model_data = {'weights': w, 'features': features, 'means': means.tolist(), 'stds': stds.tolist(), 'alpha': alpha}
with open(os.path.join(OUTPUT_DIR,'ridge_weights_nosklearn.pkl'),'wb') as f:
    pickle.dump(model_data,f)

print('✅ Modelo guardado en outputs/')

# try plotting if matplotlib available
try:
    import matplotlib.pyplot as plt
    plt.figure(figsize=(12,6))
    plt.plot(y_test, label='Real')
    plt.plot(y_pred_test, label='Predicho')
    plt.legend()
    plt.title('Predicciones vs Reales (Ridge, no sklearn)')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR,'pred_vs_actual_nosklearn.png'), dpi=200)
    print('📈 Gráfica guardada en outputs/')
except Exception:
    pass

print('🎉 Proceso completado')
