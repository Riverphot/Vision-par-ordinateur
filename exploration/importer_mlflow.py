import mlflow
import json
from pathlib import Path

# =============================================
# CONFIGURATION
# =============================================

RACINE_PROJET = Path(__file__).resolve().parents[1]

DOSSIER_RUN = (
    RACINE_PROJET / "evaluation" / "resultats"
    / "pytorch_fp32_benchmark_v1_center_crop_20261120_120305_271278"
)

# Connexion au serveur MLflow
mlflow.set_tracking_uri("http://127.0.0.1:5000")

# Création ou sélection de l'expérience
mlflow.set_experiment("Evaluation predictive")

# =============================================
# CHARGEMENT DES RESULTATS EXISTANTS
# =============================================

with open(DOSSIER_RUN / "configuration.json", encoding="utf-8") as f:
    configuration = json.load(f)

with open(DOSSIER_RUN / "metrics.json", encoding="utf-8") as f:
    metriques = json.load(f)

# =============================================
# ENREGISTREMENT DANS MLFLOW
# =============================================

with mlflow.start_run(run_name="PyTorch FP32 - Center Crop"):

    # Paramètres de l'expérimentation
    mlflow.log_params({
        "modele": configuration["modele"]["nom"],
        "version_modele": configuration["modele"]["version"],
        "precision_numerique": configuration["modele"]["precision"],
        "dataset": configuration["dataset"]["nom"],
        "pretraitement": configuration["parametres"]["pretraitement"],
        "device": configuration["parametres"]["device"],
        "top_k": configuration["parametres"]["top_k"],
        "modele_sha256": configuration["modele"]["sha256"],
        "labels_sha256": configuration["dataset"]["labels_sha256"]
    })

    # Métriques prédictives
    mlflow.log_metrics(metriques)

    # Fichiers de résultats
    mlflow.log_artifacts(str(DOSSIER_RUN))

    # Informations complémentaires
    mlflow.set_tags({
        "type_experimentation": "baseline",
        "run_source": configuration["run_id"],
        "date_evaluation": configuration["date"],
        "description": "Evaluation initiale du modele PyTorch"
    })

    print("Run MLflow enregistré :", mlflow.active_run().info.run_id)

print("Import terminé.")