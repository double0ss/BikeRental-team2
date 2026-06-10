import csv
import math
from datetime import datetime
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

DATA_PATH = "data/BikeRentalDaily_train.csv"
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def try_float(x):
    if x is None or x == "":
        return None
    try:
        return float(x)
    except:
        return None

print("📥 Cargando CSV sin pandas...")
rows = []
with open(DATA_PATH, newline='') as f:
    reader = csv.DictReader(f, delimiter=';')
    for r in reader:
        rows.append(r)

# Collect columns
cols = ['instant','dteday','season','yr','mnth','holiday','weekday','workingday','weathersit',
        'temp','atemp','hum','windspeed','leaflets','price reduction','casual','registered','cnt']

data = {c: [] for c in cols}
for r in rows:
    for c in cols:
        data[c].append(r.get(c, ''))

# parse and convert
N = len(rows)
dteday = [datetime.strptime(x, '%d.%m.%Y') for x in data['dteday']]
season = [try_float(x) for x in data['season']]
yr = [int(try_float(x) or 0) for x in data['yr']]
mnth = [int(try_float(x) or 0) for x in data['mnth']]
holiday = [int(try_float(x) or 0) for x in data['holiday']]
weekday = [int(try_float(x) or 0) for x in data['weekday']]
workingday = [int(try_float(x) or 0) for x in data['workingday']]
weathersit = [int(try_float(x) or 0) for x in data['weathersit']]
temp = [try_float(x) for x in data['temp']]
atemp = [try_float(x) for x in data['atemp']]
hum = [try_float(x) for x in data['hum']]
windspeed = [try_float(x) for x in data['windspeed']]
leaflets = [try_float(x) for x in data['leaflets']]
price_reduction = [try_float(x) for x in data['price reduction']]
cnt = [try_float(x) for x in data['cnt']]

# Impute numeric medians
def median_impute(arr):
    vals = [x for x in arr if x is not None]
    if not vals:
        return [0.0]*len(arr)
    med = float(np.median(vals))
    return [med if x is None else x for x in arr]

temp = median_impute(temp)
hum = median_impute(hum)
windspeed = median_impute(windspeed)
leaflets = median_impute(leaflets)
price_reduction = median_impute(price_reduction)

# season fallback from month if missing
for i in range(N):
    if season[i] is None:
        m = mnth[i]
        if m in [1,2,3]: season[i]=1.0
        elif m in [4,5,6]: season[i]=2.0
        elif m in [7,8,9]: season[i]=3.0
        else: season[i]=4.0

# cyclic encodings
mnth_sin = [math.sin(2*math.pi*m/12) for m in mnth]
mnth_cos = [math.cos(2*math.pi*m/12) for m in mnth]
weekday_sin = [math.sin(2*math.pi*w/7) for w in weekday]
weekday_cos = [math.cos(2*math.pi*w/7) for w in weekday]
is_weekend = [1 if w in (5,6) else 0 for w in weekday]

# Build feature matrix in a fixed order
feature_names = ['temp','hum','windspeed','leaflets','season','yr','holiday','workingday','weathersit','price_reduction','is_weekend','mnth_sin','mnth_cos','weekday_sin','weekday_cos']
X = np.column_stack([
    temp, hum, windspeed, leaflets,
    season, yr, holiday, workingday, weathersit, price_reduction,
    is_weekend, mnth_sin, mnth_cos, weekday_sin, weekday_cos
])
Y = np.array([float(x) for x in cnt])

# sort by date and split
order = np.argsort(dteday)
X = X[order]
Y = Y[order]

split_idx = int(0.8 * len(Y))
X_train, X_test = X[:split_idx], X[split_idx:]
Y_train, Y_test = Y[:split_idx], Y[split_idx:]

# scale numeric first 4 columns
scaler = StandardScaler()
X_train_num = scaler.fit_transform(X_train[:, :4])
X_test_num = scaler.transform(X_test[:, :4])

X_train_proc = np.hstack([X_train_num, X_train[:, 4:]])
X_test_proc = np.hstack([X_test_num, X_test[:, 4:]])

print("🚀 Entrenando RandomForest...")
model = RandomForestRegressor(n_estimators=300, max_depth=20, min_samples_split=5, min_samples_leaf=2, max_features='sqrt', random_state=42, n_jobs=-1)
model.fit(X_train_proc, Y_train)

def evaluate(y_true, y_pred, name):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    print(f"{name}: MAE={mae:.2f}, RMSE={rmse:.2f}, R2={r2:.4f}")
    return mae, rmse, r2

print("📊 Evaluación:")
evaluate(Y_train, model.predict(X_train_proc), 'Train')
evaluate(Y_test, model.predict(X_test_proc), 'Test')

# save model and scaler and feature names
joblib.dump({'scaler': scaler, 'model': model, 'feature_names': feature_names}, os.path.join(OUTPUT_DIR, 'bike_sharing_pipeline_minimal.pkl'))

# plot predictions vs actual
pred = model.predict(X_test_proc)
plt.figure(figsize=(12,6))
plt.plot(Y_test, label='Real')
plt.plot(pred, label='Predicho')
plt.legend()
plt.title('Predicciones vs Reales (sin pandas)')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR,'pred_vs_actual_minimal.png'), dpi=200)
print('✅ Resultados guardados en outputs/')
