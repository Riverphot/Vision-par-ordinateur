
from pathlib import Path
import pydicom

# Dossier contenant les fichiers DICOM
DOSSIER = Path("images/dicom")

for fichier in sorted(DOSSIER.glob("*.dcm")):
    print(f"\n{'=' * 55}")
    print(f"Fichier : {fichier.name}")

    try:
        ds = pydicom.dcmread(fichier)

        print("Modalité :", ds.get("Modality", "Non renseignée"))
        print("Région anatomique :", ds.get("BodyPartExamined", "Non renseignée"))
        print("Description :", ds.get("StudyDescription", "Non renseignée"))
        print("Dimensions :", ds.get("Rows", "?"), "x", ds.get("Columns", "?"))
        print("Photométrie :", ds.get("PhotometricInterpretation", "?"))
        print("Bits stockés :", ds.get("BitsStored", "?"))
        print("Rescale Slope :", ds.get("RescaleSlope", "Absent"))
        print("Rescale Intercept :", ds.get("RescaleIntercept", "Absent"))
        print("Window Center :", ds.get("WindowCenter", "Absent"))
        print("Window Width :", ds.get("WindowWidth", "Absent"))
        print("VOI LUT :", "Présente" if "VOILUTSequence" in ds else "Absente")
        print("Presentation LUT Shape :", ds.get("PresentationLUTShape", "Absent"))
        print("Nombre de frames :", ds.get("NumberOfFrames", 1))

        # Vérification du décodage
        pixels = ds.pixel_array

        print("Type numérique :", pixels.dtype)
        print("Forme du tableau :", pixels.shape)
        print("Valeurs min/max :", pixels.min(), "/", pixels.max())
        print("Décodage : OK")

    except Exception as e:
        print("Erreur :", type(e).__name__, str(e)[:200])

