
"""
Moteur d'inférence commun pour les expérimentations.

Versions disponibles :
    pytorch_fp32
    onnx_fp32
    onnx_fp16
    onnx_auteur

Les modèles ONNX sont exécutés avec ONNX Runtime sur CPU.
"""

import sys
from pathlib import Path

import numpy as np
import torch
import onnxruntime as ort

RACINE_PROJET = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RACINE_PROJET))

from inference import (
    model as model_pytorch,
    preparer_image,
    classes,
    traductions
)


# =============================================
# CONFIGURATION DES MODELES
# =============================================

MODELES = {
    "pytorch_fp32": {
        "moteur": "pytorch",
        "fichier": RACINE_PROJET / "models/xray/best.pt",
        "sortie": "logits"
    },
    "onnx_fp32": {
        "moteur": "onnx",
        "fichier": RACINE_PROJET / "models/xray/convnext_tiny_fp32.onnx",
        "sortie": "logits"
    },
    "onnx_fp16": {
        "moteur": "onnx",
        "fichier": RACINE_PROJET / "models/xray/convnext_tiny_fp16.onnx",
        "sortie": "logits"
    },
    "onnx_auteur": {
        "moteur": "onnx",
        "fichier": RACINE_PROJET / "models/xray/auteur/model.onnx",
        "sortie": "probabilites"
    }
}


# =============================================
# MOTEUR D'INFERENCE
# =============================================

class MoteurInference:

    def __init__(self, version):

        if version not in MODELES:
            raise ValueError(f"Version inconnue : {version}")

        self.version = version
        self.configuration = MODELES[version]
        self.fichier = self.configuration["fichier"]

        if not self.fichier.exists():
            raise FileNotFoundError(self.fichier)

        if self.configuration["moteur"] == "pytorch":

            self.modele = model_pytorch.cpu()
            self.modele.eval()
            self.session = None

        else:

            self.modele = None

            self.session = ort.InferenceSession(
                str(self.fichier),
                providers=["CPUExecutionProvider"]
            )

            self.nom_entree = self.session.get_inputs()[0].name
            self.nom_sortie = self.session.get_outputs()[0].name

        print(f"Modèle chargé : {version}")

    # -----------------------------------------
    # EXECUTION BRUTE DU RESEAU
    # -----------------------------------------

    def executer(self, tensor):

        if self.modele is not None:

            with torch.inference_mode():
                return self.modele(tensor)

        # ONNX Runtime utilise des tableaux NumPy
        donnees = tensor.detach().cpu().numpy()

        return self.session.run(
            [self.nom_sortie],
            {self.nom_entree: donnees}
        )[0]

    # -----------------------------------------
    # CLASSIFICATION D'UNE IMAGE
    # -----------------------------------------

    def predire(self, image, top_k=5, pretraitement="center_crop"):

        tensor = preparer_image(
            image,
            pretraitement=pretraitement
        )

        sorties = self.executer(tensor)

        if isinstance(sorties, torch.Tensor):
            sorties = sorties.detach().cpu().numpy()

        scores = sorties[0]

        # Nos modèles produisent des logits.
        # Celui de l'auteur contient déjà le softmax.
        if self.configuration["sortie"] == "logits":

            # Softmax numérique stable
            scores = scores - np.max(scores)
            exponentielles = np.exp(scores)
            scores = exponentielles / exponentielles.sum()

        # Extraction des meilleures prédictions
        indices = np.argsort(scores)[::-1][:top_k]

        return [
            {
                "classe": classes[int(indice)],
                "libelle": traductions.get(
                    classes[int(indice)],
                    classes[int(indice)]
                ),
                "score": round(float(scores[indice]), 4)
            }
            for indice in indices
        ]
