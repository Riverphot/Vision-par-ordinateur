
import csv
import random
import shutil

from pathlib import Path
from collections import defaultdict

# =============================================
# PARAMETRES DU DATASET
# =============================================

DOSSIER_DATASETS = Path(__file__).resolve().parent
DOSSIER_SOURCE = DOSSIER_DATASETS / "irma_original"

SOURCE = "dev"
NOM_DATASET = "benchmark_v1"

IMAGES_PAR_CLASSE = 20
SEED = 42

fichier_csv = DOSSIER_SOURCE / f"{SOURCE}.csv"

print("Chemin du CSV :", fichier_csv)
print("Fichier présent :", fichier_csv.exists())


# Correspondance entre les annotations IRMA
# et les classes de notre modèle

MAPPING_CLASSES = {
    "chest": "CHEST",
    "hand": "HAND",
    "knee": "KNEE",
    "lumbar spine": "LUMBAR_SPINE",
    "cervical spine": "CERVICAL_SPINE",
    "foot": "FOOT",
    "ankle joint": "ANKLE",
    "shoulder": "SHOULDER",
    "thoracic spine": "DORSAL_SPINE",
    "pelvis": "PELVIS",
}


# =============================================
# LECTURE DES ANNOTATIONS
# =============================================

fichier_csv = DOSSIER_SOURCE / f"{SOURCE}.csv"

images_par_classe = defaultdict(list)

with open(fichier_csv, encoding="utf-8-sig") as fichier:

    lecteur = csv.reader(fichier)

    for ligne in lecteur:

        if not ligne:
            continue

        nom_image = ligne[0].strip()

        annotations = {
            valeur.strip().lower()
            for valeur in ligne[1:]
        }

        classes = {
            classe
            for annotation, classe in MAPPING_CLASSES.items()
            if annotation in annotations
        }

        # Conservation des correspondances uniques
        if len(classes) == 1:

            classe = classes.pop()
            images_par_classe[classe].append(nom_image)


# =============================================
# INDEXATION DES IMAGES SOURCES
# =============================================

images_sources = {
    image.name: image
    for image in (DOSSIER_SOURCE / SOURCE).rglob("*.png")
}


# =============================================
# SELECTION ALEATOIRE
# =============================================

rng = random.Random(SEED)

selection = []

print("\nComposition du dataset :\n")

for classe in MAPPING_CLASSES.values():

    disponibles = sorted({
        nom
        for nom in images_par_classe[classe]
        if nom in images_sources
    })

    if len(disponibles) < IMAGES_PAR_CLASSE:

        raise ValueError(
            f"{classe} : seulement {len(disponibles)} "
            f"images disponibles."
        )

    images_selectionnees = rng.sample(
        disponibles,
        IMAGES_PAR_CLASSE
    )

    for nom_image in images_selectionnees:
        selection.append((nom_image, classe))

    print(
        f"{classe} : "
        f"{len(images_selectionnees)} / "
        f"{len(disponibles)} images sélectionnées"
    )


# =============================================
# CREATION DU DATASET
# =============================================

dossier_sortie = DOSSIER_DATASETS / NOM_DATASET
dossier_images = dossier_sortie / "images"

if dossier_sortie.exists():
    raise FileExistsError(
        f"Le dataset {NOM_DATASET} existe déjà. "
        "Choisir un nouveau nom de version."
    )

dossier_images.mkdir(parents=True)


# Copie des images sélectionnées

for nom_image, classe in selection:

    shutil.copy2(
        images_sources[nom_image],
        dossier_images / nom_image
    )


# Création du fichier labels.csv

with open(
    dossier_sortie / "labels.csv",
    "w",
    newline="",
    encoding="utf-8"
) as fichier:

    writer = csv.writer(fichier)

    writer.writerow(["filename", "label"])
    writer.writerows(selection)


# =============================================
# DOCUMENTATION DU DATASET
# =============================================

documentation = f"""# {NOM_DATASET}

## Source

Dataset IRMA X-ray Dataset
Partition : {SOURCE}

## Paramètres de génération

- Seed : {SEED}
- Images par classe : {IMAGES_PAR_CLASSE}
- Nombre de classes : {len(MAPPING_CLASSES)}
- Nombre total d'images : {len(selection)}

## Correspondance des classes

| Annotation IRMA | Classe du modèle |
|---|---|
"""

for annotation, classe in MAPPING_CLASSES.items():
    documentation += f"| {annotation} | {classe} |\n"


(dossier_sortie / "README.md").write_text(
    documentation,
    encoding="utf-8"
)

print(f"\nDataset créé : {dossier_sortie}")
print(f"Nombre total d'images : {len(selection)}")
