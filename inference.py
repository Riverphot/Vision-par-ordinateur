import json
from pathlib import Path

import torch
import timm
from PIL import Image, ImageOps
from torchvision import transforms


# =============================================
# CONFIGURATION
# =============================================

MODEL_DIR = Path(__file__).resolve().parent / "models" / "xray"

TAILLE_IMAGE = 224
PRETRAITEMENT_DEFAUT = "center_crop"


# =============================================
# CHARGEMENT DES CLASSES
# =============================================

with open(MODEL_DIR / "classes.txt", encoding="utf-8") as f:
    classes = [ligne.strip() for ligne in f]

with open(MODEL_DIR / "classes_fr.json", encoding="utf-8") as f:
    traductions = json.load(f)


# =============================================
# PRETRAITEMENT DES IMAGES
# =============================================

def resize_padding(image):
    """
    Redimensionne l'image en conservant ses proportions,
    puis complète avec des bandes noires.
    """

    return ImageOps.pad(
        image,
        (TAILLE_IMAGE, TAILLE_IMAGE),
        method=Image.Resampling.BILINEAR,
        color=(0, 0, 0),
        centering=(0.5, 0.5)
    )


# Méthodes de redimensionnement disponibles
METHODES_PRETRAITEMENT = {

    # Méthode de référence du modèle
    "center_crop": transforms.Compose([
        transforms.Resize(TAILLE_IMAGE),
        transforms.CenterCrop(TAILLE_IMAGE)
    ]),

    # Redimensionnement direct
    "resize": transforms.Resize(
        (TAILLE_IMAGE, TAILLE_IMAGE)
    ),

    # Redimensionnement proportionnel avec padding
    "padding": transforms.Lambda(resize_padding)
}


# Transformations communes aux trois méthodes
NORMALISATION = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


def preparer_image(
    image: Image.Image,
    pretraitement: str = PRETRAITEMENT_DEFAUT
):
    """Prépare une image selon la méthode sélectionnée."""

    if pretraitement not in METHODES_PRETRAITEMENT:
        raise ValueError(
            f"Prétraitement inconnu : {pretraitement}"
        )

    image = image.convert("RGB")

    # Redimensionnement selon la méthode choisie
    image = METHODES_PRETRAITEMENT[pretraitement](image)

    # Conversion en tenseur et normalisation
    tensor = NORMALISATION(image)

    # Ajout de la dimension batch
    return tensor.unsqueeze(0)


# =============================================
# INITIALISATION DU MODELE
# =============================================

model = timm.create_model(
    "convnext_tiny",
    pretrained=False,
    num_classes=len(classes)
)

weights = torch.load(
    MODEL_DIR / "best.pt",
    map_location="cpu",
    weights_only=True
)

model.load_state_dict(weights)
model.eval()


# =============================================
# INFERENCE
# =============================================

def predict(
    image: Image.Image,
    top_k: int = 5,
    pretraitement: str = PRETRAITEMENT_DEFAUT
):
    """Classe une radiographie et retourne les meilleures prédictions."""

    # Préparation de l'image
    tensor = preparer_image(image, pretraitement)

    # Inférence
    with torch.inference_mode():
        sorties = model(tensor)
        probabilites = torch.softmax(sorties, dim=1)

    # Sélection des meilleures prédictions
    scores, indices = torch.topk(
        probabilites,
        k=top_k,
        dim=1
    )

    resultats = []

    for indice, score in zip(indices[0], scores[0]):

        classe = classes[indice.item()]

        resultats.append({
            "classe": classe,
            "libelle": traductions.get(classe, classe),
            "score": round(score.item(), 4)
        })

    return resultats