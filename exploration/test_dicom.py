
import argparse
import pydicom
from pathlib import Path

from inference import predict
from dicom_utils import dicom_to_image, lire_metadonnees


# Récupération du fichier passé en argument
parser = argparse.ArgumentParser()
parser.add_argument("fichier", help="Chemin du fichier DICOM")
args = parser.parse_args()


# Lecture du fichier DICOM
ds = pydicom.dcmread(args.fichier)


# Extraction des métadonnées
metadonnees = lire_metadonnees(ds)

print("\nMétadonnées DICOM :")
print("Modalité :", metadonnees["modalite"])
print("Région anatomique :", metadonnees["region_anatomique"])
print("Description :", metadonnees["description"])

print("\nCaractéristiques de l'image :")
print("Dimensions :", ds.get("Rows", "?"), "x", ds.get("Columns", "?"))
print("Photométrie :", ds.get("PhotometricInterpretation", "?"))
print("Bits stockés :", ds.get("BitsStored", "?"))


# Conversion des pixels DICOM en image Pillow
image = dicom_to_image(ds)

# Sauvegarde de l'image convertie pour inspection visuelle
dossier_sortie = Path("images/convert")
dossier_sortie.mkdir(parents=True, exist_ok=True)

nom_fichier = Path(args.fichier).stem + ".png"
chemin_sortie = dossier_sortie / nom_fichier

image.save(chemin_sortie)

print("Image convertie enregistrée :", chemin_sortie)

print("\nImage convertie :")
print("Format :", image.mode)
print("Dimensions :", image.size)


# Classification anatomique avec PyTorch
resultats = predict(image)

print("\nRésultats de classification :")
for resultat in resultats:
    print(
        f"{resultat['libelle']} : "
        f"{resultat['score'] * 100:.2f} %"
    )
