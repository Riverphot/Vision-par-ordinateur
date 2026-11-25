"""
Evaluation des performances prédictives du modèle.

Ce module assure :
    - Le chargement du dataset
    - L'exécution des prédictions
    - Le calcul des métriques globales
    - Le calcul des métriques par classe
    - La construction de la matrice de confusion

Les exports et le tracking MLflow sont gérés par
le script d'orchestration experimenter.py.
"""

import sys
import json

from PIL import Image

import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support
)

from configuration import (
    RACINE_PROJET,
    FICHIER_LABELS,
    DOSSIER_IMAGES,
    NOM_MODELE,
    VERSION_MODELE,
    NOM_DATASET,
    TOP_K,
    PRETRAITEMENT
)


# =============================================
# CHARGEMENT DU DATASET
# =============================================

def charger_dataset(fichier_labels):

    labels = pd.read_csv(fichier_labels)

    print("\n--- Dataset d'évaluation ---")
    print(f"Nombre d'images : {len(labels)}")
    print(f"Nombre de classes : {labels['label'].nunique()}")

    print("\n--- Distribution des classes ---")
    print(labels["label"].value_counts())

    return labels


# =============================================
# EXECUTION DES PREDICTIONS
# =============================================

def executer_predictions(
    labels,
    predire,
    dossier_images,
    top_k,
    pretraitement
):

    resultats = []

    print("\n--- Exécution des prédictions ---")

    for index, ligne in enumerate(
        labels.itertuples(index=False),
        start=1
    ):

        chemin_image = dossier_images / ligne.filename

        # Chargement et classification de l'image
        with Image.open(chemin_image) as image:

            predictions = predire(
                image,
                top_k=top_k,
                pretraitement=pretraitement
            )

        # Extraction des classes prédites
        classes_predites = [
            p["classe"] for p in predictions
        ]

        # Enregistrement du résultat individuel
        resultats.append({
            "filename": ligne.filename,
            "label_reel": ligne.label,
            "label_predit": predictions[0]["classe"],
            "score_top1": predictions[0]["score"],
            "top5_predictions": json.dumps(predictions),
            "top1_correct": ligne.label == classes_predites[0],
            "top5_correct": ligne.label in classes_predites
        })

        # Affichage de la progression
        if index % 20 == 0 or index == len(labels):

            print(
                f"Images traitées : {index}/{len(labels)}"
            )

    return pd.DataFrame(resultats)


# =============================================
# CALCUL DES METRIQUES PREDICTIVES
# =============================================

def calculer_metriques(resultats):

    y_true = resultats["label_reel"]
    y_pred = resultats["label_predit"]

    # Classes représentées dans le dataset
    classes = sorted(y_true.unique())

    # Métriques globales
    accuracy_top1 = accuracy_score(y_true, y_pred)
    accuracy_top5 = resultats["top5_correct"].mean()

    # Métriques par classe
    precision, rappel, f1, support = (
        precision_recall_fscore_support(
            y_true,
            y_pred,
            labels=classes,
            average=None,
            zero_division=0
        )
    )

    metriques_classes = pd.DataFrame({
        "classe": classes,
        "precision": precision,
        "rappel": rappel,
        "f1_score": f1,
        "support": support
    })

    # Synthèse globale
    metriques_globales = {
        "nb_images": len(resultats),
        "nb_classes": len(classes),
        "accuracy_top1": float(accuracy_top1),
        "accuracy_top5": float(accuracy_top5),
        "precision_macro": float(precision.mean()),
        "rappel_macro": float(rappel.mean()),
        "f1_macro": float(f1.mean())
    }

    # Construction de la matrice de confusion
    matrice = pd.crosstab(
        y_true,
        y_pred,
        rownames=["Classe réelle"],
        colnames=["Classe prédite"]
    )

    # Conservation des classes prédites hors benchmark
    matrice = matrice.reindex(
        index=classes,
        columns=sorted(set(classes) | set(y_pred)),
        fill_value=0
    )

    return metriques_globales, metriques_classes, matrice


# =============================================
# EVALUATION PREDICTIVE COMPLETE
# =============================================

def evaluer_modele(
    predire,
    fichier_labels=FICHIER_LABELS,
    dossier_images=DOSSIER_IMAGES,
    top_k=TOP_K,
    pretraitement=PRETRAITEMENT
):

    """Exécute le protocole complet d'évaluation prédictive."""

    print("\n--- Evaluation prédictive ---")

    # Le protocole actuel repose sur cinq prédictions
    if top_k != 5:
        raise ValueError(
            "Le protocole prédictif nécessite TOP_K = 5."
        )

    # Chargement du dataset
    labels = charger_dataset(fichier_labels)

    if labels.empty:
        raise ValueError("Le dataset est vide.")

    # Exécution des prédictions
    predictions = executer_predictions(
        labels=labels,
        predire=predire,
        dossier_images=dossier_images,
        top_k=top_k,
        pretraitement=pretraitement
    )

    # Calcul des métriques
    metriques, metriques_classes, matrice = (
        calculer_metriques(predictions)
    )

    # Retour des résultats à l'orchestrateur
    return {
        "metrics": metriques,
        "predictions": predictions,
        "metrics_par_classe": metriques_classes,
        "matrice_confusion": matrice
    }


# =============================================
# EXECUTION AUTONOME DU SCRIPT
# =============================================

if __name__ == "__main__":

    # Import du moteur d'inférence actuel
    sys.path.insert(0, str(RACINE_PROJET))

    from inference import predict

    print("\n--- Configuration de l'évaluation ---")
    print(f"Modèle : {NOM_MODELE}")
    print(f"Version : {VERSION_MODELE}")
    print(f"Dataset : {NOM_DATASET}")
    print(f"Prétraitement : {PRETRAITEMENT}")

    # Evaluation prédictive
    resultats = evaluer_modele(
        predire=predict
    )

    # Affichage des résultats
    print("\n--- Métriques globales ---")
    print(json.dumps(resultats["metrics"], indent=4))

    print("\n--- Métriques par classe ---")
    print(
        resultats["metrics_par_classe"]
        .round(4)
        .to_string(index=False)
    )

    print("\n--- Matrice de confusion ---")
    print(resultats["matrice_confusion"])