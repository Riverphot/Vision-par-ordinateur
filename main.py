
from io import BytesIO
from pathlib import Path

import pydicom

from fastapi import FastAPI, File, UploadFile, HTTPException
from PIL import Image, UnidentifiedImageError
from pydicom.errors import InvalidDicomError

from inference import predict
from dicom_utils import dicom_to_image, lire_metadonnees


app = FastAPI(
    title="API de classification de radiographies",
    description="POC d'industrialisation d'un modèle de vision par ordinateur",
    version="1.0.0"
)


@app.get("/")
def accueil():
    return {"message": "API de vision par ordinateur opérationnelle"}


@app.post("/upload")
async def upload_image(file: UploadFile = File(...)):

    # Identification du format
    extension = Path(file.filename or "").suffix.lower()

    est_image = file.content_type in ["image/jpeg", "image/png"]
    est_dicom = (
        extension == ".dcm"
        or file.content_type == "application/dicom"
    )

    if not est_image and not est_dicom:
        raise HTTPException(
            status_code=415,
            detail="Format non pris en charge. Utiliser JPEG, PNG ou DICOM."
        )

    metadonnees = None

    try:
        # Lecture du fichier en mémoire
        contenu = await file.read()

        if est_dicom:
            # Lecture du fichier DICOM
            ds = pydicom.dcmread(BytesIO(contenu))

            # Vérification du nombre de frames
            nombre_frames = int(ds.get("NumberOfFrames", 1))

            if nombre_frames != 1:
                raise HTTPException(
                    status_code=415,
                    detail="Les DICOM multiframe ne sont pas pris en charge."
                )

            # Vérification de la modalité
            modalite = str(ds.get("Modality", "")).upper()
            region = str(ds.get("BodyPartExamined", "")).upper()

            modalite_acceptee = modalite in ["DX", "CR"]

            # Cas OT limité aux exemples radiographiques du POC
            ot_accepte = (
                modalite == "OT"
                and region in ["SHOULDER", "KNEE"]
            )

            if not modalite_acceptee and not ot_accepte:
                raise HTTPException(
                    status_code=415,
                    detail=f"Modalité DICOM non prise en charge : {modalite}"
                )

            # Extraction des métadonnées utiles
            metadonnees = lire_metadonnees(ds)

            # Conversion DICOM vers image Pillow
            image = dicom_to_image(ds)

        else:
            # Lecture des images JPEG / PNG
            image = Image.open(BytesIO(contenu))
            image.load()

            if image.format not in ["JPEG", "PNG"]:
                raise ValueError("Format d'image invalide")

    except HTTPException:
        raise

    except (
        InvalidDicomError,
        UnidentifiedImageError,
        OSError,
        ValueError,
        TypeError,
        AttributeError,
        RuntimeError
    ):
        raise HTTPException(
            status_code=400,
            detail="Fichier invalide, illisible ou encodage non pris en charge."
        )

    finally:
        await file.close()

    # Classification avec le modèle PyTorch
    resultats = predict(image)

    return {
        "filename": file.filename,
        "metadata": metadonnees,
        "predictions": resultats
    }


# Intégration de l'interface Gradio
import gradio as gr
from gradio_ui import creer_interface

demo = creer_interface()

app = gr.mount_gradio_app(
    app,
    demo,
    path="/gradio",
    max_file_size="20mb",
    run_history=False,
)