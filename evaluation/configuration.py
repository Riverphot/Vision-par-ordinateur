"""
Configuration commune des expérimentations prédictives et techniques.
"""

from pathlib import Path

# =============================================
# CHEMINS DU PROJET
# =============================================

RACINE_PROJET = Path(__file__).resolve().parents[1]

DOSSIER_RESULTATS = RACINE_PROJET / "evaluation" / "resultats"


# =============================================
# CONFIGURATION DU MODELE
# =============================================

NOM_MODELE = "convnext_tiny"

# Version à évaluer
VERSION_MODELE = "onnx_auteur"

MODELES = {
    "pytorch_fp32": {
        "moteur": "pytorch",
        "precision": "FP32",
        "fichier": RACINE_PROJET / "models/xray/best.pt"
    },
    "onnx_fp32": {
        "moteur": "onnx",
        "precision": "FP32",
        "fichier": RACINE_PROJET / "models/xray/convnext_tiny_fp32.onnx"
    },
    "onnx_fp16": {
        "moteur": "onnx",
        "precision": "FP16",
        "fichier": RACINE_PROJET / "models/xray/convnext_tiny_fp16.onnx"
    },
    "onnx_auteur": {
        "moteur": "onnx",
        "precision": "FP32",
        "fichier": RACINE_PROJET / "models/xray/auteur/model.onnx"
    }
}

CONFIG_MODELE = MODELES[VERSION_MODELE]

MOTEUR_INFERENCE = CONFIG_MODELE["moteur"]
PRECISION_NUMERIQUE = CONFIG_MODELE["precision"]
FICHIER_MODELE = CONFIG_MODELE["fichier"]

DEVICE = "cpu"


# =============================================
# CONFIGURATION DES DONNEES
# =============================================

NOM_DATASET = "benchmark_v1"

DOSSIER_DATASET = RACINE_PROJET / "datasets" / NOM_DATASET

DOSSIER_IMAGES = DOSSIER_DATASET / "images"
FICHIER_LABELS = DOSSIER_DATASET / "labels.csv"

PRETRAITEMENT = "resize"
TAILLE_IMAGE = 224


# =============================================
# EVALUATION PREDICTIVE
# =============================================

EVALUATION_PREDICTIVE = True

TOP_K = 5


# =============================================
# EVALUATION TECHNIQUE
# =============================================

EVALUATION_TECHNIQUE = True

NB_WARMUP = 5
NB_MESURES_LATENCE = 100
NB_MESURES_BATCH = 20

TAILLES_BATCH = (1, 2, 4, 8, 16)

IMAGE_TEST_TECHNIQUE = "10339.png"


# =============================================
# TRACKING MLFLOW
# =============================================

MLFLOW_ACTIF = True