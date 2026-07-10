# BikeRental — Predicción de alquiler de bicicletas

Sistema para predecir la demanda diaria de bicicletas (`cnt`) a partir de variables meteorológicas, calendario y promociones. Incluye pipeline de entrenamiento en Python, evaluación en test externo y una interfaz web con API Node.js.

## Requisitos previos

- **Python 3.9 o superior** (`python3 --version`)
- **Node.js 18 o superior** (`node --version`)
- **npm** (incluido con Node.js)

## Estructura del proyecto

```
BikeRental-team2/
├── data/                          # Datos de entrenamiento
│   └── BikeRentalDaily_train.csv
├── tests/                         # Test externo (no usar para entrenar)
│   └── BikeRentalDaily_test.csv
├── outputs/                       # Modelos, métricas y gráficos generados
├── python_api/                    # Scripts llamados por la API
├── node_server/                   # Servidor web y frontend
├── preprocessing.py               # Limpieza y features
├── train_model.py                 # Entrenamiento clásico (comparación de modelos)
├── refine_model.py                # Refinamiento iterativo guiado
├── ensemble_model.py              # Ensemble de candidatos (CV en train)
├── evaluate_model.py              # Evaluación en test externo
└── requirements.txt               # Dependencias Python
```

---

## Paso 1 — Clonar o abrir el repositorio

```bash
cd /ruta/donde/quieras/el/proyecto
git clone <url-del-repositorio> BikeRental-team2
cd BikeRental-team2
```

Si ya tienes el código, entra solo en la carpeta del proyecto.

---

## Paso 2 — Entorno virtual de Python (venv)

Crea y activa un entorno aislado en la raíz del proyecto:

### macOS / Linux

```bash
python3 -m venv venv
source venv/bin/activate
```

### Windows (PowerShell)

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### Windows (CMD)

```cmd
python -m venv venv
venv\Scripts\activate.bat
```

Verás el prefijo `(venv)` en la terminal cuando el entorno esté activo. **Todas las órdenes Python de este README asumen que el venv está activado.**

Actualiza pip e instala dependencias:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

Comprueba la instalación:

```bash
python -c "import sklearn, pandas, joblib; print('OK')"
```

> **Nota:** Cada vez que abras una terminal nueva para trabajar con Python, reactiva el venv con `source venv/bin/activate` (o el comando equivalente en Windows).

---

## Paso 3 — Dependencias de Node.js

En otra terminal (o con el venv activo, no importa para Node):

```bash
cd node_server
npm install
cd ..
```

---

## Paso 4 — Entrenar y evaluar el modelo (Python)

Ejecuta estos comandos **desde la raíz del proyecto** y **con el venv activado**.

### 4.1 Análisis de datos (opcional pero recomendado)

```bash
python data_analysis.py
```

Genera reportes y gráficos en `outputs/` (sesgos, outliers, correlaciones).

### 4.2 Refinamiento iterativo (recomendado)

Prueba varias hipótesis y guarda el mejor modelo en `outputs/model_cnt.pkl`:

```bash
python refine_model.py
```

Registro de experimentos: `outputs/refinement_log.json`.

### 4.3 Ensemble de candidatos (opcional)

Combina los mejores modelos; solo reemplaza el modelo en producción si mejora en validación cruzada (train):

```bash
python ensemble_model.py
```

Salida: `outputs/ensemble_log.json` y `outputs/ensemble_members.pkl`.

### 4.4 Evaluación en test externo

```bash
python evaluate_model.py
```

Métricas en `outputs/test_metrics.json` y gráficos de test en `outputs/`.

### Alternativa: entrenamiento clásico

Si prefieres el flujo original de comparación de algoritmos:

```bash
python train_model.py
python evaluate_model.py
```

---

## Paso 5 — Arrancar la aplicación web

### Opción A — Usar el Python del venv (recomendado)

Con el venv activado, desde la raíz del proyecto:

```bash
cd node_server
npm start
```

El servidor usa `python3` del PATH. Si el venv está activo, usará ese intérprete.

### Opción B — Indicar la ruta explícita al Python del venv

Útil si Node se lanza sin tener el venv activo:

```bash
cd node_server
PYTHON_BIN="../venv/bin/python" npm start
```

En Windows:

```powershell
$env:PYTHON_BIN="..\venv\Scripts\python.exe"
npm start
```

Abre el navegador en:

**http://localhost:3000**

---

## Paso 6 — Usar la interfaz

1. Ve a la sección **Predicción**.
2. Elige un **modelo de predicción** en el desplegable (producción, candidatos del refinamiento o ensembles).
3. Selecciona fecha, temperatura, humedad, viento, clima, etc.
4. Pulsa **Predecir**.

También puedes consultar **Gráficos** y **Análisis de variables** (métricas cargadas desde `outputs/test_metrics.json`).

---

## API REST

| Método | Ruta       | Descripción                          |
|--------|------------|--------------------------------------|
| `GET`  | `/models`  | Lista modelos disponibles            |
| `POST` | `/predict` | Predicción JSON (acepta `model_id`)  |

Ejemplo de predicción con `curl`:

```bash
curl -s -X POST http://localhost:3000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "model_id": "default",
    "prediction_date": "2012-07-15",
    "temp": 24.8,
    "hum": 53,
    "windspeed": 0.25,
    "leaflets": 0,
    "weathersit": 1,
    "holiday": 0,
    "price_reduction": 0
  }'
```

Modelos disponibles (si existen los archivos en `outputs/`):

| `model_id` | Descripción |
|------------|-------------|
| `default` | Modelo en producción (`model_cnt.pkl`) |
| `linear_legacy` | Modelo lineal antiguo |
| `member:<nombre>` | Candidato individual (ej. `member:two_stage_peak`) |
| `ensemble:weighted_average` | Promedio ponderado (pesos CV) |
| `ensemble:median` | Mediana de candidatos |
| `ensemble:segment_routing` | Mejor modelo por segmento (CV) |

---

## Resumen rápido (primera vez)

```bash
# Opción automática (venv + servidor en segundo plano)
chmod +x start_project.sh stop_server.sh
./start_project.sh          # usar bash, no sh

# Detener el servidor
./stop_server.sh
```

O manualmente:

```bash
# 1. Python
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt

# 2. Entrenar
python refine_model.py
python evaluate_model.py

# 3. Node
cd node_server && npm install && npm start

# 4. Navegador → http://localhost:3000
```

---

## Solución de problemas

### `ModuleNotFoundError` al ejecutar scripts Python

- Confirma que el venv está activado: `which python` (debería apuntar a `venv/bin/python`).
- Reinstala: `pip install -r requirements.txt`.

### La API devuelve error al predecir

- Comprueba que existe `outputs/model_cnt.pkl` (ejecuta `refine_model.py` o `train_model.py`).
- Para ensembles o candidatos individuales, ejecuta también `ensemble_model.py`.
- Arranca Node con `PYTHON_BIN` apuntando al Python del venv.

### Puerto 3000 ocupado

```bash
PORT=3001 npm start
```

Luego abre `http://localhost:3001`.

### Gráficos o métricas vacíos en la web

Ejecuta `data_analysis.py`, `evaluate_model.py` y asegúrate de que la carpeta `outputs/` contiene los PNG y JSON generados.

---

## Scripts Python principales

| Script | Función |
|--------|---------|
| `data_analysis.py` | Análisis de sesgos y outliers |
| `train_model.py` | Comparación y selección de modelos (CV) |
| `refine_model.py` | Refinamiento iterativo por fases |
| `ensemble_model.py` | Ensemble con selección por CV |
| `evaluate_model.py` | Métricas en test externo |
| `preprocessing.py` | Módulo de limpieza y features (importado por otros) |

---

## Datos

- **Entrenamiento:** `data/BikeRentalDaily_train.csv` (600 filas) — único conjunto para entrenar.
- **Test externo:** `tests/BikeRentalDaily_test.csv` (131 filas tras excluir solapamientos) — solo para evaluación final, no para entrenar ni elegir hiperparámetros de forma repetida.

---

## Licencia y equipo

Proyecto académico BikeRental — equipo 2.
