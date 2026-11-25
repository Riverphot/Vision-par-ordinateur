# Intégration du modèle dans l'API

Après validation du modèle en local avec `test_modele.py`, la logique d'inférence a été isolée dans un module `inference.py`, afin de pouvoir être réutilisée par l'API FastAPI.

Cette organisation permet de séparer les responsabilités :

- **`main.py`** : réception des requêtes HTTP, lecture et vérification des images, gestion des erreurs et restitution des réponses JSON.
- **`inference.py`** : chargement du modèle, prétraitement des images et exécution de la classification.
- **`test_modele.py`** : script de test indépendant permettant de vérifier les prédictions du modèle.

Le modèle est chargé une seule fois lors de l'import du module `inference.py`, puis réutilisé pour les requêtes suivantes dans le même processus Python. Cette approche évite de recharger les poids du réseau à chaque prédiction.

## Création du module d'inférence

Le fichier `inference.py` centralise l'utilisation du modèle ConvNeXt-Tiny et expose une fonction `predict()`.

Cette fonction reçoit une image Pillow, applique le prétraitement attendu par le modèle et retourne les cinq meilleures prédictions.

```python
def predict(image: Image.Image, top_k: int = 5):

    image = image.convert("RGB")
    tensor = transform(image).unsqueeze(0)

    with torch.inference_mode():
        sorties = model(tensor)
        probabilites = torch.softmax(sorties, dim=1)

    scores, indices = torch.topk(
        probabilites, k=top_k, dim=1
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
```

Les classes sont associées à leurs libellés français à partir du fichier `classes_fr.json`, sans modifier les sorties du modèle.

Le module a été testé indépendamment de FastAPI :

```powershell
python -c "from PIL import Image; from inference import predict; print(predict(Image.open('images/images/thorax_normal.jpg')))"
```

Le test a confirmé que les prédictions étaient identiques à celles obtenues avec `test_modele.py`.

## Connexion avec FastAPI

L'endpoint `POST /upload` a ensuite été modifié pour appeler le module d'inférence.

Le traitement suit les étapes suivantes :

1. Réception de l'image envoyée par le client.
2. Vérification du format déclaré (JPEG ou PNG).
3. Lecture de l'image en mémoire avec Pillow.
4. Transmission de l'image à la fonction `predict()`.
5. Restitution des prédictions au format JSON.

Extrait du fichier `main.py` :

```python
from io import BytesIO

from fastapi import FastAPI, File, UploadFile, HTTPException
from PIL import Image, UnidentifiedImageError

from inference import predict

app = FastAPI(
    title="API de classification de radiographies",
    version="1.0.0"
)


@app.get("/")
def accueil():
    return {"message": "API de vision par ordinateur opérationnelle"}


@app.post("/upload")
async def upload_image(file: UploadFile = File(...)):

    if file.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(
            status_code=415,
            detail="Format non pris en charge. Utiliser JPEG ou PNG."
        )

    try:
        contenu = await file.read()
        image = Image.open(BytesIO(contenu))
        image.load()

    except (UnidentifiedImageError, OSError, ValueError):
        raise HTTPException(
            status_code=400,
            detail="Image invalide ou illisible."
        )

    finally:
        await file.close()

    resultats = predict(image)

    return {
        "filename": file.filename,
        "predictions": resultats
    }
```


## Tests de l'API

L'API est démarrée avec la commande :

```powershell
uvicorn main:app --reload
```

Les tests sont réalisés depuis la documentation interactive Swagger :

http://127.0.0.1:8000/docs

Une radiographie du thorax a été transmise à l'endpoint `POST /upload`.

**Extrait de la réponse JSON :**

```json
{
  "filename": "thorax_normal.jpg",
  "predictions": [
    {
      "classe": "CHEST",
      "libelle": "Thorax",
      "score": 0.8102
    },
    {
      "classe": "SHOULDER",
      "libelle": "Épaule",
      "score": 0.0262
    }
  ]
}
```

La réponse complète contient les cinq catégories les plus probables.

**Résultat :** la classification obtenue via l'API est identique à celle du script de test initial. La chaîne de traitement JPEG/PNG, depuis la réception de l'image jusqu'à la prédiction JSON, est fonctionnelle.


