from PIL import Image
from torchvision import transforms
import torch
import timm
import json
import argparse

# Image de test
parser = argparse.ArgumentParser(description="Classification anatomique d'une radiographie")

parser.add_argument(
    "image",
    help="Chemin de l'image à analyser"
)

args = parser.parse_args()
image_path = args.image

# Ouverture de l'image
image = Image.open(image_path).convert("RGB")

print("Dimensions originales :", image.size)

# Prétraitement attendu par le modèle
transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Transformation de l'image en tenseur
tensor = transform(image)

# Ajout de la dimension batch
tensor = tensor.unsqueeze(0)

print("Dimensions du tenseur :", tensor.shape)

# Chargement du modèle
model = timm.create_model(
    "convnext_tiny",
    pretrained=False,
    num_classes=33
)

weights = torch.load(
    "models/xray/best.pt",
    map_location="cpu",
    weights_only=True
)

model.load_state_dict(weights)
model.eval()

# Chargement des catégories
with open("models/xray/classes.txt", encoding="utf-8") as f:
    classes = [ligne.strip() for ligne in f]

# Chargement des traductions françaises
with open("models/xray/classes_fr.json", encoding="utf-8") as f:
    traductions = json.load(f)

# Prédiction
with torch.inference_mode():
    sorties = model(tensor)
    probabilites = torch.softmax(sorties, dim=1)

# Récupération des 5 meilleures prédictions
scores, indices = torch.topk(probabilites, k=5, dim=1)

print("\nRésultats de classification :")

for indice, score in zip(indices[0], scores[0]):
    classe = classes[indice.item()]
    libelle = traductions.get(classe, classe)

    print(f"{libelle} : {score.item() * 100:.2f} %")