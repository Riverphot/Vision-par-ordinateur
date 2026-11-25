
"""
Suivi des expérimentations prédictives et techniques dans MLflow.

Chaque exécution de l'orchestrateur génère un Run contenant :
    - La configuration du modèle
    - Les paramètres de l'expérimentation
    - Les métriques prédictives et techniques
    - Le dataset utilisé
    - Les artefacts CSV et JSON

L'association au Model Registry est facultative.
"""

import json
from pathlib import Path

import mlflow
import pandas as pd

from mlflow.entities import LoggedModelInput


# =============================================
# CONFIGURATION MLFLOW
# =============================================

TRACKING_URI = "http://127.0.0.1:5000"

NOM_EXPERIENCE = "Evaluation predictive"

RACINE_PROJET = Path(__file__).resolve().parents[1]


# =============================================
# MODELES ENREGISTRES DANS LE REGISTRY
# =============================================

# Seuls les modèles dont le Model ID est connu
# sont référencés ici.
#
# Les autres modèles restent évaluables et suivis
# normalement dans MLflow.

MODELES_MLFLOW = {
    "pytorch_fp32": {
        "model_id": "m-602a55cecd5945cc9d6866d62ec649c0",
        "registry": "xray_bodypart_classifier",
        "version": 1
    }
}


# =============================================
# IDENTIFICATION DES METRIQUES
# =============================================

METRIQUES_PREDICTIVES = {
    "nb_images",
    "nb_classes",
    "accuracy_top1",
    "accuracy_top5",
    "precision_macro",
    "rappel_macro",
    "f1_macro"
}


# =============================================
# IDENTIFICATION DU DATASET
# =============================================

def charger_dataset_mlflow(nom_dataset):

    fichier_labels = (
        RACINE_PROJET
        / "datasets"
        / nom_dataset
        / "labels.csv"
    )

    # Chargement des annotations
    labels = pd.read_csv(fichier_labels)

    # Création de l'objet Dataset MLflow
    return mlflow.data.from_pandas(
        labels,
        source=str(fichier_labels),
        name=nom_dataset,
        targets="label"
    )


# =============================================
# ENREGISTREMENT D'UNE EXPERIMENTATION
# =============================================

def enregistrer_evaluation(
    dossier_run,
    nom_experience=NOM_EXPERIENCE
):

    dossier_run = Path(dossier_run)

    # -----------------------------------------
    # CONNEXION MLFLOW
    # -----------------------------------------

    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(nom_experience)

    # -----------------------------------------
    # LECTURE DES RESULTATS
    # -----------------------------------------

    with open(
        dossier_run / "configuration.json",
        encoding="utf-8"
    ) as fichier:

        configuration = json.load(fichier)

    with open(
        dossier_run / "metrics.json",
        encoding="utf-8"
    ) as fichier:

        metriques = json.load(fichier)

    # -----------------------------------------
    # TYPE D'EXPERIMENTATION
    # -----------------------------------------

    evaluations = configuration["evaluations"]

    predictive = evaluations["predictive"]
    technique = evaluations["technique"]

    if predictive and technique:
        type_experimentation = "complete"

    elif predictive:
        type_experimentation = "predictive"

    else:
        type_experimentation = "technique"

    # -----------------------------------------
    # IDENTIFICATION DU MODELE
    # -----------------------------------------

    version_modele = configuration["modele"]["version"]

    # Recherche facultative dans le Registry
    modele_mlflow = MODELES_MLFLOW.get(version_modele)

    model_id = (
        modele_mlflow["model_id"]
        if modele_mlflow is not None
        else None
    )

    nom_run = (
        f"{version_modele} - "
        f"{configuration['parametres']['pretraitement']} - "
        f"{configuration['parametres']['device']}"
    )

    # =========================================
    # OUVERTURE DU RUN
    # =========================================

    with mlflow.start_run(run_name=nom_run) as run:

        # -------------------------------------
        # DATASET TRACKING
        # -------------------------------------

        dataset = None

        if predictive:

            dataset = charger_dataset_mlflow(
                configuration["dataset"]["nom"]
            )

            mlflow.log_input(
                dataset,
                context="evaluation"
            )

        # -------------------------------------
        # PARAMETRES COMMUNS
        # -------------------------------------

        parametres = {
            "modele": configuration["modele"]["nom"],
            "version_modele": version_modele,
            "moteur": configuration["modele"]["moteur"],
            "precision_numerique": configuration["modele"]["precision"],
            "fichier_modele": version_modele,
            "modele_sha256": configuration["modele"]["sha256"],
            "dataset": configuration["dataset"]["nom"],
            "pretraitement": configuration["parametres"]["pretraitement"],
            "device": configuration["parametres"]["device"],
            "evaluation_predictive": predictive,
            "evaluation_technique": technique,
            "pytorch_threads": configuration["environnement"]["pytorch_threads"]
        }

        # -------------------------------------
        # PARAMETRES PREDICTIFS
        # -------------------------------------

        if predictive:

            parametres.update({
                "top_k": configuration["parametres"]["top_k"],
                "labels_sha256": configuration["dataset"]["labels_sha256"]
            })

        # -------------------------------------
        # PARAMETRES TECHNIQUES
        # -------------------------------------

        if technique:

            protocole = configuration["protocole_technique"]

            parametres.update({
                "nb_warmup": protocole["nb_warmup"],
                "nb_mesures_latence": protocole["nb_mesures_latence"],
                "nb_mesures_batch": protocole["nb_mesures_batch"],
                "tailles_batch": str(
                    protocole["tailles_batch"]
                ),
                "image_test": protocole["image_test"],
                "perimetre_technique": protocole["perimetre"]
            })

        # -------------------------------------
        # ASSOCIATION FACULTATIVE AU REGISTRY
        # -------------------------------------

        if modele_mlflow is not None:

            parametres.update({
                "modele_registry": modele_mlflow["registry"],
                "modele_registry_version": modele_mlflow["version"]
            })

            # Déclaration du modèle comme entrée du Run
            mlflow.MlflowClient().log_inputs(
                run_id=run.info.run_id,
                models=[
                    LoggedModelInput(model_id=model_id)
                ]
            )

        # Enregistrement des paramètres
        mlflow.log_params(parametres)

        # -------------------------------------
        # SEPARATION DES METRIQUES
        # -------------------------------------

        metriques_predictives = {
            nom: valeur
            for nom, valeur in metriques.items()
            if nom in METRIQUES_PREDICTIVES
        }

        metriques_techniques = {
            nom: valeur
            for nom, valeur in metriques.items()
            if nom not in METRIQUES_PREDICTIVES
        }

        # -------------------------------------
        # METRIQUES PREDICTIVES
        # -------------------------------------

        if metriques_predictives:

            if model_id is not None:

                # Association au modèle et au dataset
                mlflow.log_metrics(
                    metriques_predictives,
                    model_id=model_id,
                    dataset=dataset
                )

            else:

                # Tracking classique sans Model Registry
                mlflow.log_metrics(
                    metriques_predictives
                )

        # -------------------------------------
        # METRIQUES TECHNIQUES
        # -------------------------------------

        if metriques_techniques:

            if model_id is not None:

                # Association au modèle uniquement
                mlflow.log_metrics(
                    metriques_techniques,
                    model_id=model_id
                )

            else:

                # Tracking classique
                mlflow.log_metrics(
                    metriques_techniques
                )

        # -------------------------------------
        # ARTEFACTS
        # -------------------------------------

        mlflow.log_artifacts(
            str(dossier_run)
        )

        # -------------------------------------
        # TAGS
        # -------------------------------------

        mlflow.set_tags({
            "type_experimentation": type_experimentation,
            "run_source": configuration["run_id"],
            "date_evaluation": configuration["date"],
            "version_modele": version_modele
        })

        run_id = run.info.run_id

    # =========================================
    # BILAN DU TRACKING
    # =========================================

    print("\n--- Tracking MLflow terminé ---")
    print(f"Modèle : {version_modele}")
    print(f"Type : {type_experimentation}")

    print(
        "Model Registry : "
        + (
            modele_mlflow["registry"]
            if modele_mlflow is not None
            else "Non associé"
        )
    )

    print(f"Run ID : {run_id}")

    return run_id
