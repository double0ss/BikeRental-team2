"""
Predicción de alquiler de bicicletas (cnt) con el modelo seleccionado.
"""
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

from date_features import extrapolation_warning, parse_prediction_date
from model_inference import build_inference_matrix, get_primary_artifact, predict_cnt
from model_registry import resolve_artifact


def build_row_from_payload(payload, artifact):
    row = {}
    for feature in artifact.get("base_features", []):
        if feature in ("season", "weathersit"):
            row[feature] = int(payload.get(feature, 1))
        else:
            row[feature] = float(payload.get(feature, 0))
    return row


def coefficient_breakdown(frame, artifact):
    model = artifact.get("model")
    if model is None and artifact.get("model_type") == "two_stage":
        model = artifact.get("regressor_normal")
    if model is None:
        return []

    items = []
    selected = artifact.get("selected_features", [])

    if hasattr(model, "coef_"):
        coefs = model.coef_
        for feature, coef in zip(selected, coefs):
            value = float(frame[feature].iloc[0])
            items.append(
                {
                    "nombre": feature,
                    "valor": value,
                    "coeficiente": float(coef),
                    "contribucion": float(coef * value),
                    "importancia": float(abs(coef * value)),
                }
            )
    elif hasattr(model, "feature_importances_"):
        for feature, imp in zip(selected, model.feature_importances_):
            value = float(frame[feature].iloc[0])
            items.append(
                {
                    "nombre": feature,
                    "valor": value,
                    "importancia": float(imp),
                }
            )

    items = sorted(items, key=lambda x: x.get("importancia", 0), reverse=True)
    return items[:10]


def run_prediction(payload):
    payload = dict(payload)
    model_id = payload.pop("model_id", None) or payload.pop("modelo_id", None)
    artifact, meta = resolve_artifact(model_id)

    if payload.get("prediction_date"):
        holiday = payload.get("holiday", 0)
        derived = parse_prediction_date(payload["prediction_date"], holiday)
        payload = {**payload, **derived}
    elif "anio" in payload:
        payload["yr"] = float(payload.get("yr", max(0, int(payload["anio"]) - 2011)))

    primary = get_primary_artifact(artifact)
    row = build_row_from_payload(payload, primary)
    df = pd.DataFrame([row])
    preds = predict_cnt(df, artifact)
    cnt_pred = float(preds[0])

    frame = build_inference_matrix(df, primary)
    breakdown = coefficient_breakdown(frame, primary)

    casual_ratio = artifact.get("casual_ratio", primary.get("casual_ratio", 0.25))
    registered_ratio = artifact.get("registered_ratio", primary.get("registered_ratio", 0.75))
    warning = extrapolation_warning(
        int(payload.get("anio", 2011)),
        float(payload.get("yr", 0)),
        int(payload.get("yr_natural", payload.get("anio", 2011) - 2011)),
    )

    return {
        "modelo_id": meta.get("id"),
        "modelo_nombre": meta.get("label"),
        "modelo": artifact.get("model_type", "unknown"),
        "experiment": meta.get("experiment") or artifact.get("experiment"),
        "fecha": payload.get("prediction_date"),
        "aviso": warning,
        "predicciones": {
            "casual": float(cnt_pred * casual_ratio),
            "registered": float(cnt_pred * registered_ratio),
            "cnt": cnt_pred,
        },
        "entrada": row,
        "metricas_modelo": {
            "cv": artifact.get("cv_metrics", {}),
        },
        "variables_seleccionadas": artifact.get("selected_features", []),
        "desglose": {
            "casual": breakdown,
            "registered": breakdown,
            "cnt": breakdown,
        },
    }


raw = sys.stdin.read()
if not raw:
    print(json.dumps({"error": "No input provided"}))
    sys.exit(1)

try:
    payload = json.loads(raw)
except Exception as exc:
    print(json.dumps({"error": "Invalid JSON", "msg": str(exc)}))
    sys.exit(1)

try:
    print(json.dumps(run_prediction(payload)))
except FileNotFoundError as exc:
    print(json.dumps({"error": "Model not found", "msg": str(exc)}))
    sys.exit(1)
except Exception as exc:
    print(json.dumps({"error": "Prediction failed", "msg": str(exc)}))
    sys.exit(1)
