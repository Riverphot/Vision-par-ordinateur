
from pathlib import Path

import gradio as gr
import pydicom
import requests

from PIL import Image

from dicom_utils import dicom_to_image


API_URL = "http://127.0.0.1:8000/upload"

TYPES_FICHIERS = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".dcm": "application/dicom",
}


def analyser_image(chemin_fichier):
    """Envoie le fichier à FastAPI et présente les résultats."""

    if not chemin_fichier:
        raise gr.Error("Sélectionner une image avant de lancer l'analyse.")

    chemin = Path(chemin_fichier)
    extension = chemin.suffix.lower()

    if extension not in TYPES_FICHIERS:
        raise gr.Error("Format non pris en charge : JPEG, PNG ou DICOM.")

    # Envoi du fichier à notre API REST
    try:
        with open(chemin, "rb") as fichier:
            reponse = requests.post(
                API_URL,
                files={
                    "file": (
                        chemin.name,
                        fichier,
                        TYPES_FICHIERS[extension],
                    )
                },
                timeout=120,
            )

    except requests.RequestException as erreur:
        raise gr.Error(
            f"Impossible de contacter l'API FastAPI : {erreur}"
        )

    # Gestion des erreurs retournées par FastAPI
    if not reponse.ok:
        try:
            detail = reponse.json().get("detail", reponse.text)
        except ValueError:
            detail = reponse.text

        raise gr.Error(f"Erreur API ({reponse.status_code}) : {detail}")

    donnees = reponse.json()

    # Préparation de l'aperçu visuel
    try:
        if extension == ".dcm":
            ds = pydicom.dcmread(str(chemin))
            image = dicom_to_image(ds)
        else:
            with Image.open(chemin) as source:
                image = source.convert("RGB")

    except Exception as erreur:
        raise gr.Error(f"Impossible d'afficher l'image : {erreur}")

    # Scores renvoyés par le modèle
    predictions = donnees["predictions"]

    scores = {
        prediction["libelle"]: prediction["score"]
        for prediction in predictions
    }

    # Métadonnées DICOM
    metadonnees = donnees.get("metadata") or {}

    meilleure_prediction = predictions[0]

    message = (
        f"**Résultat : {meilleure_prediction['libelle']}** "
        f"— Score : {meilleure_prediction['score'] * 100:.2f} %"
    )

    return image, scores, metadonnees, message


def creer_interface():
    """Construit l'interface graphique Gradio."""

    with gr.Blocks(title="Classification de radiographies") as demo:

        gr.Markdown(
            """
            # Classification anatomique de radiographies

            **POC de vision par ordinateur — ConvNeXt-Tiny**

            Importer une radiographie JPEG, PNG ou DICOM pour
            identifier sa région anatomique.

            *Outil de démonstration, sans usage diagnostique.*
            """
        )

        with gr.Row():

            with gr.Column():
                fichier = gr.File(
                    label="Importer une radiographie",
                    file_types=[".jpg", ".jpeg", ".png", ".dcm"],
                    type="filepath",
                )

                bouton = gr.Button(
                    "Analyser l'image",
                    variant="primary",
                )

                apercu = gr.Image(
                    label="Image analysée",
                    interactive=False,
                    type="pil",
                    height=400,
                )

            with gr.Column():
                resultat = gr.Markdown(
                    "Importer une image puis lancer l'analyse."
                )

                scores = gr.Label(
                    label="Prédictions du modèle",
                    num_top_classes=5,
                )

                metadonnees = gr.JSON(
                    label="Métadonnées DICOM",
                )

        bouton.click(
            fn=analyser_image,
            inputs=[fichier],
            outputs=[apercu, scores, metadonnees, resultat],
        )

    return demo
