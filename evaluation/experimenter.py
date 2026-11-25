
"""
Orchestration des expérimentations du modèle.

Ce script assure :
    - Le chargement de la configuration commune
    - La sélection du moteur d'inférence PyTorch ou ONNX
    - Le pilotage des évaluations prédictives et techniques
    - La centralisation des métriques
    - L'export des résultats
    - Le tracking MLflow
"""

import sys
import json
import platform
import hashlib

from datetime import datetime
from PIL import Image

import torch
import torchvision
import timm
import sklearn
import onnxruntime as ort

import configuration as cfg

from evaluer_modele import evaluer_modele
from evaluer_performances import evaluer_performances
from moteur_inference import MoteurInference

sys.path.insert(0, str(cfg.RACINE_PROJET))

from inference import preparer_image


# =============================================
# VALIDATION DE LA CONFIGURATION
# =============================================

def valider_configuration():

    # Vérification des évaluations activées
    if not (
        cfg.EVALUATION_PREDICTIVE
        or cfg.EVALUATION_TECHNIQUE
    ):
        raise ValueError("Aucune évaluation activée.")

    # Vérification de la version sélectionnée
    if cfg.VERSION_MODELE not in cfg.MODELES:
        raise ValueError(
            f"Version inconnue : {cfg.VERSION_MODELE}"
        )

    # Exécution sur CPU uniquement à ce stade
    if cfg.DEVICE != "cpu":
        raise ValueError(
            "Seule l'exécution CPU est actuellement configurée."
        )

    # Vérification du fichier du modèle
    if not cfg.FICHIER_MODELE.is_file():
        raise FileNotFoundError(cfg.FICHIER_MODELE)

    # Vérification des ressources prédictives
    if cfg.EVALUATION_PREDICTIVE:

        if not cfg.FICHIER_LABELS.is_file():
            raise FileNotFoundError(cfg.FICHIER_LABELS)

        if cfg.TOP_K != 5:
            raise ValueError(
                "Le protocole prédictif nécessite TOP_K = 5."
            )

    # Vérification de l'image de référence technique
    if cfg.EVALUATION_TECHNIQUE:

        fichier_image = (
            cfg.DOSSIER_IMAGES
            / cfg.IMAGE_TEST_TECHNIQUE
        )

        if not fichier_image.is_file():
            raise FileNotFoundError(fichier_image)


# =============================================
# IDENTIFICATION DES FICHIERS
# =============================================

def calculer_sha256(chemin):

    empreinte = hashlib.sha256()

    with open(chemin, "rb") as fichier:

        for bloc in iter(
            lambda: fichier.read(65536),
            b""
        ):
            empreinte.update(bloc)

    return empreinte.hexdigest()


# =============================================
# CONFIGURATION DE L'EXPERIMENTATION
# =============================================

def construire_configuration(
    run_id,
    date_run,
    moteur,
    resultats_predictifs
):

    # Métriques prédictives disponibles
    metriques = (
        resultats_predictifs["metrics"]
        if resultats_predictifs is not None
        else {}
    )

    # Nombre de paramètres accessible pour PyTorch
    # Pour ONNX, ce champ reste non renseigné
    nb_parametres = (
        sum(
            p.numel()
            for p in moteur.modele.parameters()
        )
        if moteur.modele is not None
        else None
    )

    return {
        "run_id": run_id,
        "date": date_run.isoformat(),

        "modele": {
            "nom": cfg.NOM_MODELE,
            "source": "emon5122/xray-bodypart-classifier",
            "version": cfg.VERSION_MODELE,
            "moteur": cfg.MOTEUR_INFERENCE,
            "precision": cfg.PRECISION_NUMERIQUE,
            "sha256": calculer_sha256(
                cfg.FICHIER_MODELE
            ),
            "nb_parametres": nb_parametres
        },

        "dataset": {
            "nom": cfg.NOM_DATASET,
            "nb_images": metriques.get("nb_images"),
            "nb_classes": metriques.get("nb_classes"),
            "labels_sha256": (
                calculer_sha256(cfg.FICHIER_LABELS)
                if cfg.EVALUATION_PREDICTIVE
                else None
            )
        },

        "parametres": {
            "pretraitement": cfg.PRETRAITEMENT,
            "taille_entree": [
                cfg.TAILLE_IMAGE,
                cfg.TAILLE_IMAGE
            ],
            "top_k": cfg.TOP_K,
            "device": cfg.DEVICE
        },

        "evaluations": {
            "predictive": cfg.EVALUATION_PREDICTIVE,
            "technique": cfg.EVALUATION_TECHNIQUE
        },

        "protocole_technique": {
            "nb_warmup": cfg.NB_WARMUP,
            "nb_mesures_latence": cfg.NB_MESURES_LATENCE,
            "nb_mesures_batch": cfg.NB_MESURES_BATCH,
            "tailles_batch": list(cfg.TAILLES_BATCH),
            "image_test": cfg.IMAGE_TEST_TECHNIQUE,
            "perimetre": "inference_reseau_uniquement"
        },

        "environnement": {
            "systeme": platform.platform(),
            "processeur": platform.processor(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torchvision": torchvision.__version__,
            "timm": timm.__version__,
            "onnxruntime": ort.__version__,
            "scikit_learn": sklearn.__version__,
            "pytorch_threads": torch.get_num_threads(),
            "cuda_disponible": torch.cuda.is_available()
        }
    }


# =============================================
# EXPORT DES RESULTATS
# =============================================

def exporter_resultats(
    dossier_run,
    configuration,
    resultats_predictifs,
    resultats_techniques
):

    # Création d'un dossier unique pour le Run
    dossier_run.mkdir(
        parents=True,
        exist_ok=False
    )

    metriques = {}

    # -----------------------------------------
    # RESULTATS PREDICTIFS
    # -----------------------------------------

    if resultats_predictifs is not None:

        metriques.update(
            resultats_predictifs["metrics"]
        )

        resultats_predictifs["predictions"].to_csv(
            dossier_run / "predictions.csv",
            index=False
        )

        resultats_predictifs["metrics_par_classe"].to_csv(
            dossier_run / "metrics_par_classe.csv",
            index=False
        )

        resultats_predictifs["matrice_confusion"].to_csv(
            dossier_run / "matrice_confusion.csv"
        )

    # -----------------------------------------
    # RESULTATS TECHNIQUES
    # -----------------------------------------

    if resultats_techniques is not None:

        metriques.update(
            resultats_techniques["metrics"]
        )

        resultats_techniques["latences"].to_csv(
            dossier_run / "latences.csv",
            index=False
        )

        resultats_techniques["batching"].to_csv(
            dossier_run / "batching.csv",
            index=False
        )

    # -----------------------------------------
    # EXPORTS COMMUNS
    # -----------------------------------------

    fichiers_json = {
        "configuration.json": configuration,
        "metrics.json": metriques
    }

    for nom, contenu in fichiers_json.items():

        with open(
            dossier_run / nom,
            "w",
            encoding="utf-8"
        ) as fichier:

            json.dump(
                contenu,
                fichier,
                indent=4,
                ensure_ascii=False
            )

    print("\n--- Export terminé ---")
    print(
        f"Résultats enregistrés dans : {dossier_run}"
    )

    return metriques


# =============================================
# ORCHESTRATION DE L'EXPERIMENTATION
# =============================================

def experimenter():

    # Validation des paramètres
    valider_configuration()

    # -----------------------------------------
    # IDENTIFICATION DU RUN
    # -----------------------------------------

    date_run = datetime.now()

    horodatage = date_run.strftime(
        "%Y%m%d_%H%M%S_%f"
    )

    run_id = (
        f"{cfg.VERSION_MODELE}_"
        f"{cfg.NOM_DATASET}_"
        f"{cfg.PRETRAITEMENT}_"
        f"{horodatage}"
    )

    dossier_run = cfg.DOSSIER_RESULTATS / run_id

    # -----------------------------------------
    # AFFICHAGE DE LA CONFIGURATION
    # -----------------------------------------

    print("\n--- Configuration de l'expérimentation ---")
    print(f"Modèle : {cfg.NOM_MODELE}")
    print(f"Version : {cfg.VERSION_MODELE}")
    print(f"Moteur : {cfg.MOTEUR_INFERENCE}")
    print(f"Précision : {cfg.PRECISION_NUMERIQUE}")
    print(f"Device : {cfg.DEVICE}")
    print(f"Fichier : {cfg.FICHIER_MODELE.name}")
    print(f"Dataset : {cfg.NOM_DATASET}")
    print(f"Prétraitement : {cfg.PRETRAITEMENT}")

    print(
        f"Evaluation prédictive : "
        f"{cfg.EVALUATION_PREDICTIVE}"
    )

    print(
        f"Evaluation technique : "
        f"{cfg.EVALUATION_TECHNIQUE}"
    )

    print(f"MLflow : {cfg.MLFLOW_ACTIF}")

    # -----------------------------------------
    # CHARGEMENT DU MOTEUR D'INFERENCE
    # -----------------------------------------

    moteur = MoteurInference(
        cfg.VERSION_MODELE
    )

    resultats_predictifs = None
    resultats_techniques = None

    # -----------------------------------------
    # EVALUATION TECHNIQUE
    # -----------------------------------------

    if cfg.EVALUATION_TECHNIQUE:

        fichier_image = (
            cfg.DOSSIER_IMAGES
            / cfg.IMAGE_TEST_TECHNIQUE
        )

        # Préparation d'une image de référence
        # Le prétraitement est exclu des mesures
        with Image.open(fichier_image) as image:

            tensor = preparer_image(
                image,
                pretraitement=cfg.PRETRAITEMENT
            ).to(cfg.DEVICE)

        # Mesures de latence et de débit
        resultats_techniques = evaluer_performances(
            executer=moteur.executer,
            tensor=tensor,
            fichier_modele=cfg.FICHIER_MODELE,
            device=cfg.DEVICE,
            nb_warmup=cfg.NB_WARMUP,
            nb_mesures_latence=cfg.NB_MESURES_LATENCE,
            nb_mesures_batch=cfg.NB_MESURES_BATCH,
            tailles_batch=cfg.TAILLES_BATCH
        )

    # -----------------------------------------
    # EVALUATION PREDICTIVE
    # -----------------------------------------

    if cfg.EVALUATION_PREDICTIVE:

        resultats_predictifs = evaluer_modele(
            predire=moteur.predire,
            fichier_labels=cfg.FICHIER_LABELS,
            dossier_images=cfg.DOSSIER_IMAGES,
            top_k=cfg.TOP_K,
            pretraitement=cfg.PRETRAITEMENT
        )

    # -----------------------------------------
    # CONFIGURATION ET EXPORT
    # -----------------------------------------

    configuration = construire_configuration(
        run_id=run_id,
        date_run=date_run,
        moteur=moteur,
        resultats_predictifs=resultats_predictifs
    )

    metriques = exporter_resultats(
        dossier_run=dossier_run,
        configuration=configuration,
        resultats_predictifs=resultats_predictifs,
        resultats_techniques=resultats_techniques
    )

    print("\n--- Métriques globales ---")
    print(
        json.dumps(
            metriques,
            indent=4,
            ensure_ascii=False
        )
    )

    # -----------------------------------------
    # TRACKING MLFLOW
    # -----------------------------------------

    if cfg.MLFLOW_ACTIF:

        from tracking_mlflow import enregistrer_evaluation

        enregistrer_evaluation(dossier_run)

    return dossier_run


# =============================================
# EXECUTION PRINCIPALE
# =============================================

if __name__ == "__main__":

    experimenter()
