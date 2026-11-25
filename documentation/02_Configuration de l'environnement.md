# Configuration de l'environnement et premier tests

## Préparation de l'environnement 

Le projet utilise **Conda** pour gérer un environnement Python isolé.

Cette approche permet :
- D'éviter les conflits de dépendances entre projets.
- De maîtriser la version de Python utilisée.
- De faciliter la reproductibilité de l'environnement.

**Création de l'environnement :**

```bash
conda create -n vision-api python=3.11 -y
```

**Activation :**

```bash
conda activate vision-api
```

## Création de l'API

L'objectif est de mettre en place une première API REST avec **FastAPI**, capable de recevoir une image via une requête HTTP POST et de retourner une réponse JSON. 
Cette première version constitue le socle de l'application. 

### Installation des dépendances


```bash
pip install fastapi uvicorn python-multipart
```

- **FastAPI** : framework Python utilisé pour développer l'API REST.
- **Uvicorn** : serveur ASGI permettant d'exécuter l'application.
- **python-multipart** : bibliothèque permettant de recevoir des fichiers via des formulaires HTTP.

### Développement de l'API

Création du fichier `main.py` :

```python
from fastapi import FastAPI, File, UploadFile

# Initialisation de l'application
app = FastAPI(title="API de classification de radiographies")

# Endpoint de vérification
@app.get("/")
def accueil():
    return {"message": "API de vision par ordinateur opérationnelle"}


# Endpoint de réception des images
@app.post("/upload")
async def upload_image(file: UploadFile = File(...)):

    return {
        "filename": file.filename,
        "content_type": file.content_type,
        "message": "Fichier reçu avec succès"
    }
```

Deux endpoints sont disponibles :

- **`GET /`** : vérifier que l'API fonctionne.
- **`POST /upload`** : recevoir un fichier et retourner ses informations (nom et type MIME déclaré).

### Exécution et tests

Démarrage du serveur :

```bash
uvicorn main:app --reload
```

L'API est accessible aux adresses suivantes :

- API : http://127.0.0.1:8000
- Documentation Swagger : http://127.0.0.1:8000/docs


