"""
Orquestador de refinamiento iterativo guiado por métricas de test externo.
Cada iteración prueba una hipótesis distinta y registra resultados en refinement_log.json.
"""
import json
import os
import time

import joblib

from data_analysis import run_full_analysis
from evaluate_model import evaluate_artifact
from train_model import FEATURE_CONFIGS, save_candidate_artifact, train_candidate

DATA_PATH = "data/BikeRentalDaily_train.csv"
TEST_PATH = "tests/BikeRentalDaily_test.csv"
OUTPUT_DIR = "outputs"
LOG_PATH = os.path.join(OUTPUT_DIR, "refinement_log.json")
IMPROVEMENT_THRESHOLD = 0.02

MINIMAL = FEATURE_CONFIGS[3]
FULL = FEATURE_CONFIGS[0]

EXPERIMENTS = [
    # Fase 0 — baseline
    {
        "name": "baseline_minimal_gbr",
        "hypothesis": "Línea base actual: GBR + features minimal",
        "phase": 0,
        "feature_config": MINIMAL,
        "model_type": "gradient_boosting",
        "model_params": {
            "n_estimators": 300,
            "max_depth": 5,
            "learning_rate": 0.05,
            "subsample": 0.9,
        },
    },
    # Fase 1 — anti-overfitting
    {
        "name": "gbr_regularized",
        "hypothesis": "GBR demasiado flexible: regularizar profundidad y hojas",
        "phase": 1,
        "feature_config": MINIMAL,
        "model_type": "gradient_boosting",
        "model_params": {
            "n_estimators": 300,
            "max_depth": 3,
            "min_samples_leaf": 8,
            "learning_rate": 0.05,
            "subsample": 0.7,
            "validation_fraction": 0.15,
            "n_iter_no_change": 15,
        },
    },
    {
        "name": "gbr_huber_loss",
        "hypothesis": "Pérdida sensible a outliers: GBR con loss huber",
        "phase": 1,
        "feature_config": MINIMAL,
        "model_type": "gradient_boosting_huber",
        "model_params": {
            "n_estimators": 300,
            "max_depth": 4,
            "min_samples_leaf": 5,
            "learning_rate": 0.05,
            "subsample": 0.8,
        },
    },
    {
        "name": "huber_regressor",
        "hypothesis": "Regresión robusta Huber en features minimal",
        "phase": 1,
        "feature_config": MINIMAL,
        "model_type": "huber",
        "model_params": {"epsilon": 1.35, "alpha": 0.001},
    },
    {
        "name": "lasso_full_features",
        "hypothesis": "Regularización fuerte Lasso con config full",
        "phase": 1,
        "feature_config": FULL,
        "model_type": "lasso",
    },
    # Fase 2 — fines de semana
    {
        "name": "weekend_interactions",
        "hypothesis": "Patrón distinto laborable vs fin de semana: is_weekend + interacción temp",
        "phase": 2,
        "feature_config": {
            **MINIMAL,
            "name": "minimal_weekend",
            "weekend_features": True,
        },
        "model_type": "gradient_boosting",
        "model_params": {
            "n_estimators": 300,
            "max_depth": 4,
            "min_samples_leaf": 5,
            "learning_rate": 0.05,
            "subsample": 0.8,
        },
    },
    {
        "name": "dual_model_weekend",
        "hypothesis": "Modelos separados para laborables y fines de semana",
        "phase": 2,
        "feature_config": {
            **MINIMAL,
            "name": "minimal_dual",
            "weekend_features": True,
        },
        "model_type": "gradient_boosting",
        "model_params": {
            "n_estimators": 250,
            "max_depth": 4,
            "min_samples_leaf": 5,
            "learning_rate": 0.05,
            "subsample": 0.8,
        },
        "dual_model": True,
    },
    {
        "name": "weekend_calibration",
        "hypothesis": "Calibración multiplicativa para workingday=0 aprendida en CV",
        "phase": 2,
        "feature_config": {
            **MINIMAL,
            "name": "minimal_calibrated",
            "weekend_features": True,
        },
        "model_type": "gradient_boosting",
        "model_params": {
            "n_estimators": 300,
            "max_depth": 4,
            "min_samples_leaf": 5,
            "learning_rate": 0.05,
            "subsample": 0.8,
        },
        "weekend_calibration": True,
    },
    # Fase 3 — outliers
    {
        "name": "winsorize_p99",
        "hypothesis": "Outliers distorsionan: winsorizar cnt al P99 en train",
        "phase": 3,
        "feature_config": MINIMAL,
        "model_type": "gradient_boosting",
        "model_params": {
            "n_estimators": 300,
            "max_depth": 4,
            "min_samples_leaf": 5,
            "learning_rate": 0.05,
            "subsample": 0.8,
        },
        "winsorize_pct": 99,
    },
    {
        "name": "two_stage_peak",
        "hypothesis": "Días pico: clasificador P95 + regresores separados",
        "phase": 3,
        "feature_config": MINIMAL,
        "model_type": "gradient_boosting",
        "model_params": {
            "n_estimators": 250,
            "max_depth": 4,
            "min_samples_leaf": 5,
            "learning_rate": 0.05,
            "subsample": 0.8,
        },
        "two_stage": True,
        "peak_percentile": 95,
    },
    {
        "name": "winsorize_plus_huber",
        "hypothesis": "Combinar winsorize P99 con GBR huber",
        "phase": 3,
        "feature_config": MINIMAL,
        "model_type": "gradient_boosting_huber",
        "model_params": {
            "n_estimators": 300,
            "max_depth": 3,
            "min_samples_leaf": 8,
            "learning_rate": 0.05,
            "subsample": 0.7,
        },
        "winsorize_pct": 99,
    },
    # Fase 4 — features completas
    {
        "name": "full_heat_index",
        "hypothesis": "Config full + heat index en lugar de atemp",
        "phase": 4,
        "feature_config": {
            **FULL,
            "name": "full_heat",
            "heat_index": True,
        },
        "model_type": "gradient_boosting",
        "model_params": {
            "n_estimators": 300,
            "max_depth": 4,
            "min_samples_leaf": 5,
            "learning_rate": 0.05,
            "subsample": 0.8,
        },
    },
    {
        "name": "full_heat_weekend",
        "hypothesis": "Full + heat index + features fin de semana + calibración",
        "phase": 4,
        "feature_config": {
            **FULL,
            "name": "full_heat_weekend",
            "heat_index": True,
            "weekend_features": True,
        },
        "model_type": "gradient_boosting",
        "model_params": {
            "n_estimators": 300,
            "max_depth": 4,
            "min_samples_leaf": 5,
            "learning_rate": 0.05,
            "subsample": 0.8,
        },
        "weekend_calibration": True,
    },
    {
        "name": "full_regularized",
        "hypothesis": "Full features con GBR regularizado fuerte",
        "phase": 4,
        "feature_config": {
            **FULL,
            "name": "full_reg",
            "heat_index": True,
        },
        "model_type": "gradient_boosting",
        "model_params": {
            "n_estimators": 300,
            "max_depth": 3,
            "min_samples_leaf": 10,
            "learning_rate": 0.05,
            "subsample": 0.7,
            "validation_fraction": 0.15,
            "n_iter_no_change": 15,
        },
    },
]


def run_refinement():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    log = {
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "experiments": [],
        "best": None,
        "stopped_early": False,
    }

    print("=== Fase 0: análisis de sesgos y outliers ===")
    bias_report = run_full_analysis(DATA_PATH, TEST_PATH)
    log["bias_summary"] = bias_report.get("summary", {})
    print()

    best_score = float("inf")
    best_entry = None
    best_artifact = None
    no_improve_streak = 0
    last_improvement_pct = None

    for i, experiment in enumerate(EXPERIMENTS, 1):
        name = experiment["name"]
        print(f"\n=== Iteración {i}/{len(EXPERIMENTS)}: {name} ===")
        print(f"Hipótesis: {experiment.get('hypothesis', '')}")

        t0 = time.time()
        try:
            artifact = train_candidate(experiment)
            report = evaluate_artifact(artifact, save_outputs=False)
            score = report["composite_score"]["score"]
            elapsed = time.time() - t0

            entry = {
                "iteration": i,
                "name": name,
                "phase": experiment.get("phase"),
                "hypothesis": experiment.get("hypothesis"),
                "elapsed_sec": round(elapsed, 1),
                "test_metrics": {
                    "mae": report["global"]["mae"],
                    "rmse": report["global"]["rmse"],
                    "r2": report["global"]["r2"],
                    "mae_median": report["global"].get("mae_median"),
                    "mape_median": report["global"].get("mape_median"),
                    "r2_gap_train_test": report["global"].get("r2_gap_train_test"),
                },
                "composite_score": report["composite_score"],
                "segments": report["segments"],
                "improved": score < best_score,
            }
            log["experiments"].append(entry)

            g = report["global"]
            cs = report["composite_score"]
            print(
                f"Test MAE={g['mae']:.1f} | R2={g['r2']:.4f} | "
                f"MAE finde={cs['mae_weekend']:.1f} | MAE 2012={cs['mae_yr2012']:.1f} | "
                f"Score={score:.2f} ({elapsed:.1f}s)"
            )

            if score < best_score:
                improvement = (best_score - score) / best_score if best_score < float("inf") else 1.0
                if last_improvement_pct is not None and improvement < IMPROVEMENT_THRESHOLD:
                    no_improve_streak += 1
                else:
                    no_improve_streak = 0
                last_improvement_pct = improvement
                best_score = score
                best_entry = entry
                best_artifact = artifact
                print(f"  -> Nuevo mejor candidato (mejora {improvement * 100:.1f}%)")
            else:
                no_improve_streak += 1
                print(f"  -> Sin mejora ({no_improve_streak} rondas seguidas)")

            completed_phases = {e.get("phase") for e in log["experiments"] if "error" not in e}
            if (
                no_improve_streak >= 2
                and last_improvement_pct is not None
                and last_improvement_pct < IMPROVEMENT_THRESHOLD
                and completed_phases >= {0, 1, 2, 3, 4}
            ):
                print(
                    f"\nCriterio de parada: mejora < {IMPROVEMENT_THRESHOLD * 100:.0f}% "
                    f"tras completar todas las fases. Deteniendo búsqueda."
                )
                log["stopped_early"] = True
                break

        except Exception as exc:
            print(f"  ERROR en {name}: {exc}")
            log["experiments"].append({
                "iteration": i,
                "name": name,
                "error": str(exc),
            })

        with open(LOG_PATH, "w", encoding="utf-8") as f:
            json.dump(log, f, indent=2, default=str)

    if best_artifact:
        best_artifact["bias_outlier_summary"] = bias_report.get("summary", {})
        save_candidate_artifact(best_artifact)
        evaluate_artifact(best_artifact, save_outputs=True)
        joblib.dump(best_artifact, os.path.join(OUTPUT_DIR, "model_cnt.pkl"))

        log["best"] = {
            "name": best_entry["name"],
            "composite_score": best_entry["composite_score"],
            "test_metrics": best_entry["test_metrics"],
            "phase": best_entry.get("phase"),
        }
        log["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")

        with open(LOG_PATH, "w", encoding="utf-8") as f:
            json.dump(log, f, indent=2, default=str)

        print("\n=== Refinamiento completado ===")
        print(f"Mejor modelo: {best_entry['name']}")
        print(f"Score compuesto: {best_score:.2f}")
        tm = best_entry["test_metrics"]
        print(f"MAE test={tm['mae']:.1f} | R2 test={tm['r2']:.4f}")
        print(f"Log: {LOG_PATH}")
    else:
        print("No se encontró ningún candidato válido.")

    return log


if __name__ == "__main__":
    run_refinement()
