# Chargement du modèle PyTorch

## Architecture et poids du modèle

Le modèle sélectionné, `emon5122/xray-bodypart-classifier`, repose sur **ConvNeXt-Tiny**, une architecture de réseau neuronal conçue pour la classification d'images.

Son développement s'appuie sur l'apprentissage par transfert (*Transfer Learning*) :

1. **Architecture ConvNeXt-Tiny** : structure du réseau neuronal.
2. **Pré-entraînement sur ImageNet** : (lot d'images) apprentissage de caractéristiques visuelles générales.
3. **Fine-tuning** : adaptation du modèle à la reconnaissance de 33 régions anatomiques à partir de radiographies.
4. **Export des poids** : sauvegarde des paramètres entraînés dans le fichier `best.pt`.

Le dépôt Hugging Face fournit deux versions du modèle :

- **`best.pt` (PyTorch)** : poids entraînés permettant de reconstruire le modèle avec son architecture.
- **`model.onnx` (ONNX)** : version exportée dans un format facilitant notamment l'exécution et le déploiement.

La version PyTorch a été retenue comme **référence initiale**, afin de réaliser ensuite les optimisations et de mesurer leurs effets sur les performances.

## Reconstruction du modèle

L'exécution du modèle repose sur trois bibliothèques :

- **PyTorch** : framework de deep learning permettant les calculs et l'exécution du réseau neuronal.
- **Torchvision** : outils de traitement d'images, pour préparer les images uploadée.
- **timm** : bibliothèque fournissant notamment l'architecture ConvNeXt-Tiny utilisée lors de l'entraînement.

Le fichier `best.pt` contient uniquement les poids entraînés (*state_dict*). L'architecture doit donc être reconstruite avant de les charger.

```python
import torch
import timm

# Construction de l'architecture
model = timm.create_model(
    "convnext_tiny",
    pretrained=False,
    num_classes=33
)

# Chargement des poids entraînés
weights = torch.load(
    "models/xray/best.pt",
    map_location="cpu",
    weights_only=True
)

model.load_state_dict(weights)
model.eval()
```

Le paramètre `pretrained=False` évite de charger les poids génériques d'ImageNet, puisque les poids spécialisés du modèle radiographique sont ensuite chargés depuis `best.pt`.

Le modèle est exécuté sur **CPU**, afin de disposer d'une configuration de référence pour les futures comparaisons de performances.

## Prétraitement des images

Avant l'inférence, chaque image doit être transformée conformément à la configuration du modèle :

- Conversion en RGB.
- Redimensionnement du plus petit côté à 224 pixels.
- Recadrage central en 224 × 224 pixels.
- Conversion en tenseur et normalisation des valeurs des pixels.
- Ajout d'une dimension *batch*.

Le tenseur obtenu possède les dimensions **`[1, 3, 224, 224]`** : une image, trois canaux de couleur, une hauteur et une largeur de 224 pixels.

## Première inférence

Une première classification a été réalisée sur une radiographie du thorax au format JPEG, afin de vérifier le fonctionnement du modèle PyTorch.

Le script `test_modele.py` réalise les opérations suivantes :

1. Chargement et prétraitement de l'image avec Torchvision.
2. Reconstruction de l'architecture ConvNeXt-Tiny avec `timm`.
3. Chargement des poids entraînés depuis `best.pt`.
4. Exécution de l'inférence sur CPU.
5. Conversion des sorties du réseau en scores avec `softmax`.
6. Extraction des cinq meilleures prédictions et traduction des catégories en français.

### Exécution de la prédiction

Après préparation de l'image sous forme de tenseur, la prédiction est réalisée avec le code suivant :

```python
# Passage du modèle en mode évaluation
model.eval()

# Exécution de l'inférence
with torch.inference_mode():
    sorties = model(tensor)
    probabilites = torch.softmax(sorties, dim=1)

# Sélection des 5 catégories les plus probables
scores, indices = torch.topk(probabilites, k=5, dim=1)

# Affichage des résultats
for indice, score in zip(indices[0], scores[0]):
    classe = classes[indice.item()]
    libelle = traductions.get(classe, classe)

    print(f"{libelle} : {score.item() * 100:.2f} %")
```

La fonction `softmax` transforme les valeurs produites par le réseau en scores compris entre 0 et 1. La fonction `topk` permet de sélectionner les cinq catégories ayant les scores les plus élevés.

Le script est exécuté depuis le terminal :

```powershell
python test_modele.py
```

### Résultats obtenus

**Image testée :** `images/images/thorax_normal.jpg`

```text
Dimensions originales : (1929, 2207)
Dimensions du tenseur : torch.Size([1, 3, 224, 224])
```

| Catégorie prédite | Score |
|---|---:|
| Thorax | **81,02 %** |
| Épaule | 2,62 % |
| Bras | 1,06 % |
| Membre supérieur | 0,93 % |
| Genou | 0,88 % |

Le modèle a correctement identifié la région anatomique représentée, avec un score nettement supérieur aux autres catégories.

**Bilan :** cette première inférence valide le chargement du modèle, le prétraitement de l'image et la production de prédictions. Les scores représentent les valeurs de confiance calculées par le modèle et ne garantissent pas l'exactitude de la classification.

Ce premier objectif est de valider l'intégration du modèle choisi dans notre api et de tester le déploiement. Ici nous ne remettons pas en cause les scores de confiance données par le modèle sélectionnés.
Il sera cependant intéressant de pousser les tests sur le modèle avant de valider ces scores ulterieurement.
