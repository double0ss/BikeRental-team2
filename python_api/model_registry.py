"""
Catálogo de modelos disponibles para predicción en la API y el frontend.
"""
import json
import os

try:
    import joblib
except Exception:
    import pickle as _pickle

    class _JoblibFallback:
        @staticmethod
        def load(path):
            with open(path, "rb") as f:
                return _pickle.load(f)

    joblib = _JoblibFallback()

from ensemble_model import build_ensemble_artifact

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")
MODEL_PATH = os.path.join(OUTPUT_DIR, "model_cnt.pkl")
LEGACY_MODEL_PATH = os.path.join(OUTPUT_DIR, "model_cnt_linear.pkl")
ENSEMBLE_MEMBERS_PATH = os.path.join(OUTPUT_DIR, "ensemble_members.pkl")
ENSEMBLE_LOG_PATH = os.path.join(OUTPUT_DIR, "ensemble_log.json")
REFINEMENT_LOG_PATH = os.path.join(OUTPUT_DIR, "refinement_log.json")

MEMBER_LABELS = {
    "full_heat_index": "Full + heat index",
    "two_stage_peak": "Dos etapas (días pico)",
    "weekend_interactions": "Interacciones fin de semana",
    "winsorize_plus_huber": "Winsorize P99 + Huber",
    "gbr_huber_loss": "GBR pérdida Huber",
    "weekend_calibration": "Calibración fin de semana",
    "baseline_minimal_gbr": "Baseline minimal",
}

ENSEMBLE_LABELS = {
    "weighted_average": "Ensemble — promedio ponderado (CV)",
    "median": "Ensemble — mediana (CV)",
    "segment_routing": "Ensemble — competencia por segmento (CV)",
}


def _read_json(path):
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _test_mae_by_name():
    scores = {}
    for log_path in (REFINEMENT_LOG_PATH, ENSEMBLE_LOG_PATH):
        log = _read_json(log_path)
        for exp in log.get("experiments", []):
            name = exp.get("name")
            tm = exp.get("test_metrics") or {}
            if name and "mae" in tm:
                scores[name] = float(tm["mae"])
        best = log.get("best") or {}
        if best.get("name") and best.get("test_metrics", {}).get("mae"):
            scores[best["name"]] = float(best["test_metrics"]["mae"])
    return scores


def _artifact_label(artifact):
    if artifact.get("experiment"):
        return artifact["experiment"]
    if artifact.get("model_type") == "ensemble":
        return artifact.get("experiment", f"ensemble_{artifact.get('strategy', '')}")
    return artifact.get("model_type", "modelo")


def _load_ensemble_bundle():
    if not os.path.exists(ENSEMBLE_MEMBERS_PATH):
        return None, {}
    bundle = joblib.load(ENSEMBLE_MEMBERS_PATH)
    log = _read_json(ENSEMBLE_LOG_PATH)
    return bundle, log


def list_models():
    """Lista modelos disponibles con metadatos para el frontend."""
    models = []
    test_mae = _test_mae_by_name()

    if os.path.exists(MODEL_PATH):
        try:
            art = joblib.load(MODEL_PATH)
            exp = art.get("experiment", _artifact_label(art))
            mae = test_mae.get(exp.replace("ensemble_", ""), test_mae.get(exp))
            label = MEMBER_LABELS.get(exp, exp)
            if art.get("model_type") == "ensemble":
                label = ENSEMBLE_LABELS.get(art.get("strategy", ""), label)
            desc = f"Modelo en producción ({art.get('model_type', 'unknown')})"
            if mae:
                desc += f" · MAE test ≈ {mae:.0f}"
            models.append({
                "id": "default",
                "label": f"{label} (producción)",
                "description": desc,
                "model_type": art.get("model_type"),
                "experiment": exp,
                "is_default": True,
                "test_mae": mae,
            })
        except Exception:
            models.append({
                "id": "default",
                "label": "Modelo principal",
                "description": "outputs/model_cnt.pkl",
                "is_default": True,
            })

    if os.path.exists(LEGACY_MODEL_PATH):
        models.append({
            "id": "linear_legacy",
            "label": "Lineal (legado)",
            "description": "Modelo lineal original",
            "model_type": "linear",
            "is_default": False,
        })

    bundle, log = _load_ensemble_bundle()
    if bundle and bundle.get("members"):
        weights = log.get("oof_weights") or bundle.get("oof_weights") or {}
        routing = log.get("oof_segment_routing") or bundle.get("oof_routing") or {}

        for member in bundle["members"]:
            name = member["name"]
            mae = test_mae.get(name)
            label = MEMBER_LABELS.get(name, name)
            desc = "Candidato del refinamiento"
            if mae:
                desc += f" · MAE test ≈ {mae:.0f}"
            models.append({
                "id": f"member:{name}",
                "label": label,
                "description": desc,
                "model_type": member["artifact"].get("model_type"),
                "experiment": name,
                "is_default": False,
                "test_mae": mae,
            })

        if len(bundle["members"]) >= 2:
            for strategy, slabel in ENSEMBLE_LABELS.items():
                models.append({
                    "id": f"ensemble:{strategy}",
                    "label": slabel,
                    "description": "Combinación de candidatos; pesos/routing aprendidos en CV (train)",
                    "model_type": "ensemble",
                    "experiment": f"ensemble_{strategy}",
                    "is_default": False,
                })

    if not models:
        models.append({
            "id": "default",
            "label": "Sin modelo entrenado",
            "description": "Ejecuta refine_model.py o train_model.py",
            "is_default": True,
        })

    return {"models": models, "default_id": "default"}


def resolve_artifact(model_id=None):
    """Carga el artifact según id del catálogo."""
    model_id = (model_id or "default").strip()

    if model_id == "default":
        path = MODEL_PATH if os.path.exists(MODEL_PATH) else LEGACY_MODEL_PATH
        if not os.path.exists(path):
            raise FileNotFoundError("No hay modelo entrenado en outputs/")
        artifact = joblib.load(path)
        meta = {
            "id": "default",
            "label": _artifact_label(artifact),
            "experiment": artifact.get("experiment"),
        }
        return artifact, meta

    if model_id == "linear_legacy":
        if not os.path.exists(LEGACY_MODEL_PATH):
            raise FileNotFoundError("Modelo lineal legado no encontrado")
        artifact = joblib.load(LEGACY_MODEL_PATH)
        return artifact, {"id": model_id, "label": "Lineal (legado)", "experiment": "linear_legacy"}

    if model_id.startswith("member:"):
        name = model_id.split(":", 1)[1]
        bundle, _ = _load_ensemble_bundle()
        if not bundle:
            raise FileNotFoundError("Ejecuta ensemble_model.py para generar candidatos")
        for member in bundle.get("members", []):
            if member["name"] == name:
                label = MEMBER_LABELS.get(name, name)
                return member["artifact"], {
                    "id": model_id,
                    "label": label,
                    "experiment": name,
                }
        raise ValueError(f"Miembro no encontrado: {name}")

    if model_id.startswith("ensemble:"):
        strategy = model_id.split(":", 1)[1]
        if strategy not in ENSEMBLE_LABELS:
            raise ValueError(f"Estrategia ensemble desconocida: {strategy}")

        bundle, log = _load_ensemble_bundle()
        if not bundle or len(bundle.get("members", [])) < 2:
            raise FileNotFoundError("Ejecuta ensemble_model.py para generar el ensemble")

        weights = log.get("oof_weights") or bundle.get("oof_weights")
        routing = log.get("oof_segment_routing") or bundle.get("oof_routing")
        members = bundle["members"]
        primary_name = members[0]["name"]
        if log.get("selected_by_cv", "").startswith("individual:"):
            primary_name = log["selected_by_cv"].split(":", 1)[1]

        artifact = build_ensemble_artifact(
            strategy,
            members,
            weights=weights if strategy == "weighted_average" else None,
            routing=routing if strategy == "segment_routing" else None,
            primary_name=primary_name,
        )
        return artifact, {
            "id": model_id,
            "label": ENSEMBLE_LABELS[strategy],
            "experiment": artifact.get("experiment"),
        }

    raise ValueError(f"model_id desconocido: {model_id}")
