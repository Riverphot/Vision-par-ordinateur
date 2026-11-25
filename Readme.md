# Classification de radiographies — Vision par ordinateur

POC de classification anatomique de radiographies, basé sur un modèle **ConvNeXt-Tiny**. L'application propose une **API FastAPI** et une **interface Gradio** permettant d'importer des images **JPEG, PNG ou DICOM** et de consulter les cinq prédictions principales. Les fichiers DICOM peuvent également fournir des métadonnées.

## Exécution en local

**Prérequis :** Python 3.11 et Conda.

Depuis la racine du projet, dans PowerShell :

```powershell
conda create -n vision-api python=3.11 -y  # première installation uniquement
conda activate vision-api
pip install -r requirements.txt
```

Le modèle PyTorch doit être présent dans `models/xray/best.pt`.

Démarrer l'API :

```powershell
uvicorn main:app --reload
```

- **Documentation interactive (Swagger)** : http://127.0.0.1:8000/docs
- **Endpoint de classification** : `POST /upload`

Pour utiliser l'interface Gradio, la lancer dans un **second terminal** (API toujours active) :

```powershell
conda activate vision-api
python gradio_ui.py
```

Ouvrir l'URL locale indiquée par Gradio dans le terminal.

## Exécution avec Docker

**Prérequis :** Docker Desktop 

### À partir de l'image construite

```powershell
docker build -t vision-api:1.0 .
docker run --rm -p 8000:8000 vision-api:1.0
```

API : http://127.0.0.1:8000/docs


## MLflow — suivi des expérimentations

Le suivi des évaluations est indépendant du fonctionnement de l'API. Depuis la racine du projet, avec l'environnement `vision-api` activé :

```powershell
mlflow server --host 127.0.0.1 --port 5000 --backend-store-uri sqlite:///mlflow_tracking/mlflow.db --artifacts-destination ./mlflow_tracking/artifacts
```

Interface MLflow : http://127.0.0.1:5000

Dans un autre terminal, pour lancer une évaluation avec la configuration définie dans `evaluation/configuration.py` :

```powershell
python evaluation/experimenter.py
```

## Organisation du projet

```text
.
├── main.py                  # API FastAPI
├── gradio_ui.py             # Interface utilisateur
├── inference.py             # Chargement du modèle et inférence
├── dicom_utils.py           # Lecture et traitement des fichiers DICOM
├── documentation/           # Contient tout le descriptif et les étapes projet
├── models/                  # Poids des modèles et scripts d'export ONNX
│   └── xray/
├── datasets/                # Jeux de données (lot images irma)
├── evaluation/              # Évaluations, configuration et tracking MLflow
│   └── resultats/           # Résultats exportés (CSV / JSON)
├── exploration/             # Scripts de tests ponctuels
├── images/                  # Dossier d'image pour tester l'aplication
├── notebooks/               # Notebooks d'analyse et d'exploration
├── mlflow_tracking/         # Base SQLite et artefacts MLflow locaux
├── Dockerfile               # Construction de l'image de l'application
├── compose.yaml             # Configuration Docker Compose 
├── requirements.txt         # Dépendances de l'application
└── requirements-dev.txt     # Dépendances des expérimentations
```

Les travaux et résultats détaillés des expérimentations sont documentés séparément. L'API utilise actuellement le modèle **PyTorch FP32**.