# Conteneurisation de l'application avec Docker

La conteneurisation permet de disposer d'un environnement d'exécution reproductible, indépendant de l'environnement Conda utilisé pour le développement initial.

Deux modes d'utilisation ont été mis en place :

- **Image autonome** : exécution de l'application à partir d'une image contenant le code et les poids du modèle.
- **Environnement de développement** : exécution dans Docker avec synchronisation du code local et rechargement automatique.

## Environnement Docker

L'image Linux utilise Python 3.11 et installe les dépendances nécessaires à l'application, sans embarquer l'environnement Conda du poste Windows.

Les dépendances Python ont été séparées en deux fichiers afin de distinguer l'exécution de l'application des outils d'expérimentation.

### `requirements.txt`

Ce fichier contient les bibliothèques nécessaires à l'application :

- FastAPI, Uvicorn et python-multipart pour l'API.
- Gradio et Requests pour l'interface utilisateur.
- PyTorch, Torchvision et timm pour l'inférence.
- Pillow, NumPy et pydicom pour le traitement des images.

Les versions utilisées sont issues de l'environnement Conda `vision-api`.

Les versions CPU de PyTorch et Torchvision sont explicitement sélectionnées afin de ne pas installer les dépendances CUDA dans l'image.

### `requirements-dev.txt`

Ce fichier complète les dépendances applicatives avec les outils d'expérimentation :

- Scikit-learn, Pandas et Matplotlib pour l'évaluation.
- Jupyter et ipykernel pour les notebooks.
- psutil pour l'analyse des ressources.
- MLflow pour le suivi des expérimentations.

```text
-r requirements.txt
```

L'image Docker actuelle installe uniquement `requirements.txt`. Les outils d'expérimentation restent disponibles dans l'environnement Conda local.

Cette séparation limite les dépendances inutiles dans l'image applicative.

## Construction de l'image Docker

Un fichier `Dockerfile` a été créé à la racine du projet.

Le Dockerfile réalise les opérations suivantes :

1. Préparation d'un environnement Python 3.11 sous Linux.
2. Installation des dépendances applicatives.
3. Copie des modules Python nécessaires au fonctionnement de l'API et de Gradio.
4. Intégration des poids et fichiers de configuration du modèle.
5. Configuration du démarrage de l'application sur le port 8000.

L'installation des dépendances précède la copie du code applicatif afin de bénéficier du cache Docker lors des reconstructions.

Les poids du modèle sont directement intégrés à l'image. Aucun téléchargement du modèle n'est nécessaire au démarrage du conteneur.

### Fichier `.dockerignore`

Le fichier `.dockerignore` permet d'exclure du contexte de construction les ressources inutiles à l'application.

Les principaux éléments exclus sont :

| Élément | Justification |
|---|---|
| `.git/` | Historique du dépôt |
| `.venv/`, `venv/` | Environnements Python locaux |
| `__pycache__/`, `*.pyc` | Fichiers temporaires Python |
| `notebooks/`, `exploration/` | Travaux exploratoires |
| `evaluation/`, `datasets/` | Outils et données d'évaluation |
| `mlflow_tracking/` | Base et artefacts MLflow |
| `docs/`, `*.md` | Documentation |
| `requirements-dev.txt` | Dépendances non nécessaires à l'application |
| `.env`, `.env.*` | Fichiers de configuration potentiellement sensibles |

L'exclusion des datasets et des artefacts MLflow évite notamment d'augmenter inutilement la taille du contexte de construction.

## Image construite et validation

Depuis la racine du projet :

```powershell
docker build -t vision-api:1.0 .
```

La commande crée une image nommée `vision-api`, avec le tag `1.0`.

Vérification :

```powershell
docker images
```

L'image a été construite avec succès.

| Élément | Résultat |
|---|---|
| Image | `vision-api:1.0` |
| Image ID | `76b7b2348244` |
| Disk usage affiché par Docker | 2,15 Go |
| Content size affiché par Docker | 526 Mo |

La taille de l'image est notamment liée aux bibliothèques d'inférence, en particulier PyTorch et ses dépendances.

### Démarrage du conteneur

```powershell
docker run --rm --name vision-api -p 8000:8000 vision-api:1.0
```

Le conteneur démarre un serveur Uvicorn qui expose l'application FastAPI.



## Environnement de développement avec Docker Compose

L'image autonome contient une copie du code source. Toute modification nécessite donc une nouvelle construction de l'image pour être intégrée.

Afin de faciliter le développement, un environnement Docker Compose a été configuré avec des volumes et le rechargement automatique du serveur.

### Fichier `compose.yaml`

```yaml
services:
  vision-api:
    build:
      context: .
    image: vision-api:dev
    container_name: vision-api-dev

    ports:
      - "8000:8000"

    volumes:
      # Synchronisation des fichiers Python
      - ./main.py:/app/main.py
      - ./inference.py:/app/inference.py
      - ./dicom_utils.py:/app/dicom_utils.py
      - ./gradio_ui.py:/app/gradio_ui.py

    command:
      - uvicorn
      - main:app
      - --host
      - 0.0.0.0
      - --port
      - "8000"
      - --reload
      - --reload-dir
      - /app
```

### Fonctionnement des volumes

Les fichiers Python du poste hôte sont montés dans le conteneur.

Une modification du code local est ainsi immédiatement visible dans l'environnement Docker.

L'option `--reload` d'Uvicorn permet de redémarrer automatiquement le serveur lorsque des changements sont détectés.

Les poids du modèle restent intégrés à l'image Docker et ne sont pas montés depuis le poste hôte.

### Commandes principales

Démarrage de l'environnement de développement :

```powershell
docker compose up --build
```

Démarrage sans reconstruction :

```powershell
docker compose up
```

Une reconstruction reste nécessaire lorsqu'une dépendance, le Dockerfile ou les fichiers intégrés directement dans l'image sont modifiés.

La conteneurisation concerne actuellement l'application d'inférence uniquement.

Les outils de diagnostic, d'évaluation et de suivi des expérimentations restent exécutés depuis l'environnement Conda `vision-api`.

Cette organisation permet de poursuivre le développement de l'application dans Docker tout en conservant les outils d'expérimentation existants.

