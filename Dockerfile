# Image de base Python sous Linux
FROM python:3.11-slim

# Configuration de Python
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Répertoire de travail dans le conteneur
WORKDIR /app

# Installation des dépendances
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copie des fichiers de l'application
COPY main.py .
COPY inference.py .
COPY dicom_utils.py .
COPY gradio_ui.py .

# Copie du modèle et de ses fichiers de configuration
COPY models/ ./models/

# Port de l'application
EXPOSE 8000

# Démarrage de l'application FastAPI
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]