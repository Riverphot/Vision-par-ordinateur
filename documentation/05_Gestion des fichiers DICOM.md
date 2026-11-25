# Gestion des fichiers DICOM

Après avoir validé la classification des images JPEG et PNG via FastAPI, le projet a été étendu à la prise en charge des images médicales au format DICOM. 
Cette étape nécessite un traitement spécifique.

L'objectif est de **convertir les images DICOM en images exploitables par le modèle ConvNeXt-Tiny**, tout en conservant les métadonnées utiles à l'analyse.

## Comprendre le format DICOM

**DICOM** (*Digital Imaging and Communications in Medicine*) est un standard international utilisé pour représenter, stocker et échanger des données d'imagerie médicale. Il est notamment utilisé pour les radiographies, scanners, IRM et échographies.

Un objet DICOM peut contenir :

- Les données numériques des pixels.
- Des métadonnées techniques décrivant l'image.
- Des informations sur l'examen et la région anatomique.
- Des informations administratives et éventuellement des données personnelles de santé.

Un fichier DICOM peut contenir une seule image ou plusieurs images, appelées *frames*.

Par exemple, les examens scanner et IRM peuvent comporter de nombreuses coupes anatomiques permettant d'explorer progressivement une région du corps. 


## Analyse des fichiers DICOM

Avant de développer la conversion, une phase exploratoire a été réalisée sur les fichiers disponibles. Cette approche permet d'identifier concrètement les différentes représentations des pixels et les métadonnées nécessaires à leur interprétation.



### Bibliothèque pydicom

Pour permettre la lecture et le traitement des fichiers DICOM en Python, la bibliothèque **`pydicom`** a été installée dans l'environnement `vision-api`.

```powershell
pip install pydicom
```

Cette bibliothèque permet notamment de :
- Lire les fichiers DICOM et accéder à leurs métadonnées.
- Extraire et décoder les données des pixels.
- Récupérer les images sous forme de tableaux NumPy pour effectuer les traitements nécessaires.


### Métadonnées examinées

Un script `analyse_dicom.py` a été créé pour parcourir les fichiers DICOM disponibles et examiner leurs caractéristiques techniques.

| Métadonnée | Description |
|---|---|
| `Modality` | Modalité d'imagerie : DX, CR, CT, MR, OT… |
| `BodyPartExamined` | Région anatomique déclarée |
| `StudyDescription` | Description de l'examen |
| `Rows`, `Columns` | Dimensions de l'image |
| `PhotometricInterpretation` | Interprétation des valeurs des pixels |
| `BitsStored` | Nombre de bits significatifs par pixel |
| `NumberOfFrames` | Nombre d'images contenues dans l'objet |
| `RescaleSlope`, `RescaleIntercept` | Paramètres de transformation des intensités |
| `WindowCenter`, `WindowWidth` | Paramètres de fenêtrage |
| `VOILUTSequence` | Éventuelle table de correspondance d'affichage |
| `PresentationLUTShape` | Indication complémentaire sur le rendu des intensités |

L'analyse des sept fichiers DICOM a permis d'identifier différentes caractéristiques techniques, notamment les types de pixels, leur profondeur et les paramètres de fenêtrage.

Les cinq radiographies exploitables dans notre POC se répartissent en trois cas :

| Type d'image | Fichiers concernés | Traitement nécessaire |
|---|---|---|
| Niveaux de gris 8 bits | Thorax TensorFlow | Conversion directe en RGB |
| Niveaux de gris 12/14 bits | Thorax COVID et LIDC | Application du fenêtrage DICOM et normalisation en 8 bits |
| Couleur RGB 8 bits | Épaule et genou | Conversion directe en image Pillow |

Les deux autres fichiers, correspondant à un scanner (CT) et une IRM (MR), sont exclus du périmètre du POC.

**Cette analyse a permis de définir les différentes transformations nécessaires pour convertir les images DICOM en un format compatible avec notre modèle de classification.**


## Représentation numérique et affichage des images DICOM

Contrairement aux images classiques, les images DICOM peuvent conserver une grande précision dans les valeurs des pixels et nécessiter des transformations spécifiques pour être affichées correctement.

Plusieurs caractéristiques interviennent dans cette représentation :

- **Photométrie (`PhotometricInterpretation`)** : définit comment interpréter les pixels. Une image peut être en couleur (`RGB`) ou en niveaux de gris (`MONOCHROME1` et `MONOCHROME2`). Ces deux derniers modes se distinguent par le sens de représentation des intensités, du noir vers le blanc ou inversement.
- **Profondeur de bits (`BitsStored`)** : détermine le nombre de niveaux d'intensité possibles. Une image 8 bits dispose de 256 niveaux, contre 4 096 pour une image 12 bits. Une profondeur plus importante permet de conserver davantage de nuances, sans modifier la résolution spatiale.
- **Transformation des intensités (`RescaleSlope`, `RescaleIntercept`)** : permet de convertir les valeurs numériques initiales des pixels selon les paramètres définis dans le fichier.
- **Fenêtrage (`WindowCenter`, `WindowWidth`)** : définit la plage d'intensités à représenter visuellement. Il permet d'ajuster la luminosité et le contraste en mettant en évidence certaines nuances de gris, sans modifier les pixels d'origine.

Les paramètres de fenêtrage sont facultatifs. Un fichier peut proposer une ou plusieurs fenêtres, ou n'en définir aucune. Dans ce dernier cas, une méthode de normalisation doit être choisie pour afficher l'image.

**Ces informations guident la conversion des images DICOM vers un format exploitable par notre modèle.** Le traitement consiste à interpréter les pixels selon leur photométrie, appliquer les transformations d'intensité et le fenêtrage lorsqu'ils sont disponibles, puis ramener les valeurs sur 8 bits pour produire une image RGB compatible avec notre chaîne de prétraitement.

Cette conversion constitue une étape importante : les choix d'affichage peuvent modifier le contraste de l'image et potentiellement influencer les prédictions du modèle.


## Développement du module de conversion DICOM

Un module `dicom_utils.py` a été créé pour isoler le traitement des fichiers DICOM de la logique d'inférence.

Il contient deux fonctions principales :

- `dicom_to_image()` : conversion des pixels DICOM en image Pillow.
- `lire_metadonnees()` : extraction des informations utiles à l'analyse.


Les métadonnées sont conservées séparément des prédictions.

Elles pourront notamment servir à comparer la région anatomique déclarée dans le DICOM avec celle reconnue par le modèle.



## Tests de conversion et de classification

Le script `test_dicom.py` a été adapté pour utiliser les fonctions de `dicom_utils.py`.

Son fonctionnement est le suivant :

1. Lecture d'un fichier DICOM avec `pydicom`.
2. Extraction des métadonnées.
3. Conversion des pixels en image Pillow.
4. Transmission de l'image à `predict()`.
5. Affichage des cinq meilleures prédictions.

Exemple d'exécution :

```powershell
python test_dicom.py images/dicom/DX_thorax_LIDC.dcm
```

Le module d'inférence reste inchangé : il reçoit une image Pillow, indépendamment de son format d'origine.

### Résultats obtenus

| Image | Profondeur | Classe prédite | Score |
|---|---|---|---:|
| Thorax TensorFlow | 8 bits | Thorax | 85,57 % |
| Thorax COVID SBU | 12 bits | Thorax | 86,25 % |
| Thorax LIDC | 14 bits | Thorax | 63,67 % |
| Épaule OrthoView | RGB 8 bits | Épaule | 91,14 % |
| Genou OrthoView | RGB 8 bits | Genou | 85,47 % |

**Résultat :** les cinq radiographies ont été correctement décodées, converties et classifiées par le modèle PyTorch.

Ces tests valident la chaîne de conversion sur les fichiers disponibles. Ils ne suffisent pas à garantir une compatibilité générale avec tous les fichiers DICOM ni à évaluer statistiquement les performances du modèle.

Pour faciliter la compréhension du traitement et contrôler le rendu obtenu, une sauvegarde locale en PNG a été ajoutée au script de test.

Les fichiers sont enregistrés dans le répertoire : `images/convert/`

Cela permet de comparer le rendu avec une visionneuse DICOM et de mieux appréhender les différents conepts liés aux métadonnées de ces fichiers.


## Périmètre du POC

Le modèle retenu est entraîné à reconnaître les régions anatomiques sur des **radiographies conventionnelles en deux dimensions**.

Le périmètre de cette première version est donc limité :

- Radiographies numériques de modalités `DX` et `CR`.
- Radiographies identifiées dans certains fichiers de modalité `OT`.
- Fichiers DICOM contenant une seule image (*single-frame*).
- Images convertibles en RGB pour être transmises au modèle.

Les scanners (`CT`) et IRM (`MR`) sont exclus : leurs images correspondent à d'autres modalités d'acquisition et nécessiteraient un traitement ainsi qu'un modèle adaptés.

Les fichiers multiframe sont également exclus pour l'instant afin de conserver un fonctionnement simple : **un fichier transmis correspond à une image à classifier**.


## Intégration des fichiers DICOM dans FastAPI

Après validation du module `dicom_utils.py` avec les scripts de test, l'endpoint `POST /upload` de `main.py` a été adapté pour prendre en charge les fichiers DICOM, en complément des formats JPEG et PNG déjà opérationnels.


L'API identifie le format du fichier puis applique le traitement correspondant :

1. **JPEG/PNG** : lecture de l'image en mémoire avec Pillow.
2. **DICOM** : lecture avec `pydicom`, vérification des caractéristiques et conversion en image Pillow avec `dicom_to_image()`.
3. **Classification** : transmission de l'image obtenue à `predict()`, qui applique le prétraitement attendu par ConvNeXt-Tiny.
4. **Restitution** : retour des cinq meilleures prédictions au format JSON, accompagnées des métadonnées pour les fichiers DICOM.

Plusieurs vérifications ont été ajoutées dans `main.py` afin de respecter le périmètre défini pour le POC. Comme le contrôle de type d'imagerie, et les modalités prise en charge ou non.

Pour les fichiers DICOM, la réponse contient désormais les métadonnées extraites, en complément des prédictions.

Exemple obtenu avec une radiographie du genou :

```json
{
  "filename": "OT_genou_AP_OrthoView.dcm",
  "metadata": {
    "modalite": "OT",
    "region_anatomique": "Knee",
    "description": null
  },
  "predictions": [
    {
      "classe": "KNEE",
      "libelle": "Genou",
      "score": 0.8547
    },
    {
      "classe": "ABDOMEN",
      "libelle": "Abdomen",
      "score": 0.0138
    },
    {
      "classe": "FOOT",
      "libelle": "Pied",
      "score": 0.0107
    },
    {
      "classe": "LOWER_EXTREMITY",
      "libelle": "Membre inférieur",
      "score": 0.0082
    },
    {
      "classe": "SPINE",
      "libelle": "Colonne vertébrale",
      "score": 0.0077
    }
  ]
}
```

