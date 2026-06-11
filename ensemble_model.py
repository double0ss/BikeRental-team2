"""
Ensemble de los mejores candidatos del refinamiento.
Aprende pesos y estrategia solo con CV temporal en train; el test externo solo reporta.
Guarda model_cnt.pkl solo si el ensemble supera al mejor individual en score CV.
"""
import json
import os
import time

import joblib
import numpy as np
from sklearn.model_selection import TimeSeriesSplit

from evaluate_model import composite_from_predictions, evaluate_artifact
from model_inference import predict_cnt
from preprocessing import TARGET, extract_train_stats, load_and_clean
from refine_model import EXPERIMENTS
from train_model import save_candidate_artifact, train_candidate

DATA_PATH = "data/BikeRentalDaily_train.csv"
OUTPUT_DIR = "outputs"
LOG_PATH = os.path.join(OUTPUT_DIR, "ensemble_log.json")
MODEL_PATH = os.path.join(OUTPUT_DIR, "model_cnt.pkl")

ENSEMBLE_MEMBER_NAMES = [
    "full_heat_index",
    "two_stage_peak",
    "weekend_interactions",
    "winsorize_plus_huber",
]

STRATEGIES = ("weighted_average", "median", "segment_routing")


def get_member_experiments():
    by_name = {e["name"]: e for e in EXPERIMENTS}
    return [by_name[n] for n in ENSEMBLE_MEMBER_NAMES if n in by_name]


def lean_artifact(artifact):
    """Quita objetos pesados antes de persistir miembros del ensemble."""
    skip = {"df", "X", "X_raw", "y_log", "y_raw"}
    lean = {k: v for k, v in artifact.items() if k not in skip}
    if lean.get("model_type") == "dual":
        segments = {}
        for key, seg in lean.get("segments", {}).items():
            segments[key] = {k: v for k, v in seg.items() if k not in skip}
        lean["segments"] = segments
    return lean


def build_oof_predictions(df, experiments, n_splits=4):
    """Predicciones out-of-fold por miembro (solo train, sin fuga al test)."""
    tscv = TimeSeriesSplit(n_splits=n_splits)
    names = [e["name"] for e in experiments]
    oof = {name: np.zeros(len(df), dtype=float) for name in names}

    for fold, (train_idx, val_idx) in enumerate(tscv.split(df), 1):
        fold_train = df.iloc[train_idx].copy().reset_index(drop=True)
        fold_val = df.iloc[val_idx].copy().reset_index(drop=True)
        fold_stats = extract_train_stats(fold_train)
        print(f"  OOF fold {fold}/{n_splits}...")
        for exp in experiments:
            artifact = train_candidate(exp, df=fold_train, train_stats=fold_stats)
            oof[exp["name"]][val_idx] = predict_cnt(fold_val, artifact)

    return oof


def inverse_mae_weights(oof_preds, y_true):
    weights = {}
    for name, preds in oof_preds.items():
        mae = float(np.mean(np.abs(y_true - preds)))
        weights[name] = 1.0 / (mae + 1e-6)
    total = sum(weights.values())
    return {k: v / total for k, v in weights.items()}


def apply_weighted_average(oof_preds, weights, names):
    out = np.zeros(len(next(iter(oof_preds.values()))), dtype=float)
    for name in names:
        out += weights[name] * oof_preds[name]
    return out


def apply_median(oof_preds, names):
    stack = np.column_stack([oof_preds[n] for n in names])
    return np.median(stack, axis=1)


def learn_segment_routing(oof_preds, y_true, df, names):
    """Elige el mejor miembro por segmento según MAE OOF en train."""

    def best_for_mask(mask):
        scores = {}
        for name in names:
            if mask.sum() < 2:
                continue
            scores[name] = float(np.mean(np.abs(y_true[mask] - oof_preds[name][mask])))
        return min(scores, key=scores.get) if scores else names[0]

    wd0 = df["workingday"].values == 0
    wd1 = df["workingday"].values == 1
    yr1 = df["yr"].values == 1

    routing = {
        "workingday_0": best_for_mask(wd0),
        "workingday_1": best_for_mask(wd1),
        "yr_1": best_for_mask(yr1),
    }
    return routing


def apply_segment_routing(oof_preds, df, routing, names):
    preds = np.zeros(len(df), dtype=float)
    for i in range(len(df)):
        row = df.iloc[i]
        if row["workingday"] == 0:
            member = routing["workingday_0"]
        elif row["yr"] == 1:
            member = routing["yr_1"]
        else:
            member = routing["workingday_1"]
        preds[i] = oof_preds[member][i]
    return preds


def evaluate_strategy_on_oof(strategy, oof_preds, df, y_true, names, weights=None, routing=None):
    if strategy == "weighted_average":
        pred = apply_weighted_average(oof_preds, weights, names)
    elif strategy == "median":
        pred = apply_median(oof_preds, names)
    elif strategy == "segment_routing":
        pred = apply_segment_routing(oof_preds, df, routing, names)
    else:
        raise ValueError(f"Estrategia desconocida: {strategy}")

    score_info, global_metrics, segments = composite_from_predictions(df, y_true, pred)
    return {
        "strategy": strategy,
        "cv_composite_score": score_info["score"],
        "cv_metrics": global_metrics,
        "composite_detail": score_info,
        "segments": segments,
        "weights": weights,
        "segment_routing": routing,
    }


def build_ensemble_artifact(strategy, members, weights=None, routing=None, primary_name=None):
    """Construye artifact de producción con miembros entrenados en todo el train."""
    lean_members = [{"name": m["name"], "artifact": lean_artifact(m["artifact"])} for m in members]
    primary = next(m for m in lean_members if m["name"] == primary_name)
    primary_art = primary["artifact"]

    return {
        "model_type": "ensemble",
        "strategy": strategy,
        "experiment": f"ensemble_{strategy}",
        "hypothesis": "Combinación de top candidatos con selección por CV en train",
        "members": lean_members,
        "member_names": [m["name"] for m in lean_members],
        "weights": weights or {},
        "segment_routing": routing or {},
        "primary_member": primary_name,
        "target": TARGET,
        "use_log_transform": True,
        "selected_features": primary_art.get("selected_features", []),
        "scale_columns": primary_art.get("scale_columns", []),
        "scaler_mean": primary_art.get("scaler_mean", []),
        "scaler_scale": primary_art.get("scaler_scale", []),
        "dummy_columns": primary_art.get("dummy_columns", []),
        "base_features": primary_art.get("base_features", []),
        "casual_ratio": primary_art.get("casual_ratio", 0.25),
        "registered_ratio": primary_art.get("registered_ratio", 0.75),
        "base_year": 2011,
        "train_stats": primary_art.get("train_stats"),
        "normalized": True,
        "model": primary_art.get("model"),
    }


def run_ensemble():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    experiments = get_member_experiments()
    names = [e["name"] for e in experiments]

    print("=== Ensemble: miembros ===")
    for n in names:
        print(f"  - {n}")

    df = load_and_clean(DATA_PATH)
    y_true = df[TARGET].astype(float).values

    print("\n=== OOF temporal (solo train) ===")
    t0 = time.time()
    oof_preds = build_oof_predictions(df, experiments)
    print(f"OOF completado en {time.time() - t0:.1f}s")

    weights = inverse_mae_weights(oof_preds, y_true)
    routing = learn_segment_routing(oof_preds, y_true, df, names)

    candidates = []

    for name in names:
        score_info, global_m, segments = composite_from_predictions(df, y_true, oof_preds[name])
        candidates.append({
            "strategy": f"individual:{name}",
            "cv_composite_score": score_info["score"],
            "cv_metrics": global_m,
            "composite_detail": score_info,
        })
        print(
            f"  individual:{name:22} CV score={score_info['score']:.1f} "
            f"MAE={global_m['mae']:.1f} finde={score_info['mae_weekend']:.1f}"
        )

    for strategy in STRATEGIES:
        result = evaluate_strategy_on_oof(
            strategy,
            oof_preds,
            df,
            y_true,
            names,
            weights=weights if strategy == "weighted_average" else None,
            routing=routing if strategy == "segment_routing" else None,
        )
        candidates.append(result)
        print(
            f"  {strategy:22} CV score={result['cv_composite_score']:.1f} "
            f"MAE={result['cv_metrics']['mae']:.1f}"
        )

    best_cv = min(candidates, key=lambda c: c["cv_composite_score"])
    best_individual = min(
        [c for c in candidates if c["strategy"].startswith("individual:")],
        key=lambda c: c["cv_composite_score"],
    )

    print(f"\nMejor individual (CV): {best_individual['strategy']} "
          f"(score={best_individual['cv_composite_score']:.1f})")
    print(f"Mejor estrategia (CV): {best_cv['strategy']} "
          f"(score={best_cv['cv_composite_score']:.1f})")

    use_ensemble = not best_cv["strategy"].startswith("individual:")
    chosen_strategy = best_cv["strategy"]
    if chosen_strategy.startswith("individual:"):
        chosen_member = chosen_strategy.split(":", 1)[1]
    else:
        chosen_member = best_individual["strategy"].split(":", 1)[1]

    print("\n=== Entrenamiento final en todo el train ===")
    train_stats = extract_train_stats(df)
    members = []
    for exp in experiments:
        artifact = train_candidate(exp, df=df, train_stats=train_stats)
        members.append({"name": exp["name"], "artifact": artifact})
        print(f"  entrenado: {exp['name']}")

    log = {
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "members": names,
        "oof_weights": weights,
        "oof_segment_routing": routing,
        "cv_candidates": [
            {
                "strategy": c["strategy"],
                "cv_composite_score": c["cv_composite_score"],
                "cv_mae": c["cv_metrics"]["mae"],
                "cv_r2": c["cv_metrics"]["r2"],
                "composite_detail": c.get("composite_detail"),
            }
            for c in candidates
        ],
        "selected_by_cv": best_cv["strategy"],
        "deployed_as_ensemble": use_ensemble,
    }

    members_path = os.path.join(OUTPUT_DIR, "ensemble_members.pkl")
    joblib.dump({"members": members, "oof_weights": weights, "oof_routing": routing}, members_path)

    if use_ensemble:
        strategy_name = best_cv["strategy"]
        artifact = build_ensemble_artifact(
            strategy_name,
            members,
            weights=weights if strategy_name == "weighted_average" else None,
            routing=routing if strategy_name == "segment_routing" else None,
            primary_name=chosen_member,
        )
        artifact["cv_metrics"] = best_cv["cv_metrics"]
        save_candidate_artifact(artifact)
        joblib.dump(artifact, MODEL_PATH)
        deployed = artifact
        log["deployed_model"] = f"ensemble:{strategy_name}"
        print(f"\nDesplegado: ensemble ({strategy_name})")
    else:
        deployed = None
        if os.path.exists(MODEL_PATH):
            deployed = joblib.load(MODEL_PATH)
            log["deployed_model"] = deployed.get("experiment", deployed.get("model_type"))
            print(
                f"\nEnsemble no superó en CV (mejor: {best_cv['strategy']}, "
                f"score={best_cv['cv_composite_score']:.1f}). "
                f"model_cnt.pkl sin cambios ({log['deployed_model']})."
            )
        else:
            primary = next(m for m in members if m["name"] == chosen_member)
            artifact = primary["artifact"]
            save_candidate_artifact(artifact)
            joblib.dump(artifact, MODEL_PATH)
            deployed = artifact
            log["deployed_model"] = chosen_member
            print(f"\nDesplegado: individual ({chosen_member}) — sin modelo previo")

    test_report = evaluate_artifact(deployed, save_outputs=True)
    log["test_metrics"] = {
        "strategy_deployed": deployed.get("experiment"),
        "model_type": deployed.get("model_type"),
        "global": test_report["global"],
        "composite_score": test_report["composite_score"],
    }
    log["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")

    with open(LOG_PATH, "w", encoding="utf-8") as f:
        json.dump(log, f, indent=2, default=str)

    tm = test_report["global"]
    cs = test_report["composite_score"]
    print("\n=== Test externo (solo reporte) ===")
    print(
        f"MAE={tm['mae']:.1f} | R2={tm['r2']:.4f} | "
        f"finde={cs['mae_weekend']:.1f} | score={cs['score']:.1f}"
    )
    print(f"Log: {LOG_PATH}")
    return log


if __name__ == "__main__":
    run_ensemble()
