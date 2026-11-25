# Préparation du dataset de référence

Afin d'évaluer le modèle et de comparer les résultats des futures expérimentations (optimisation, quantification, changement de modèle), un dataset de référence a été constitué.

L'objectif est de disposer d'un ensemble d'images annotées, équilibré et reproductible, permettant de comparer les différentes versions du modèle dans les mêmes conditions.

## Acquisition des données

Le dataset retenu est **IRMA X-ray Dataset**, un ensemble de radiographies disposant d'annotations anatomiques et techniques.

Sources :

- [IRMA X-ray Dataset – Kaggle](https://www.kaggle.com/datasets/raddar/irma-xray-dataset)
- [ImageCLEF 2009 – Medical Image Annotation](https://www.imageclef.org/2009/medical)

Le dataset téléchargé est organisé en trois partitions :

| Partition | Composition | Utilisation envisagée |
|---|---|---|
| `train` | 9 000 images annotées | Éventuel entraînement ou fine-tuning |
| `dev` | 1 000 images annotées | Constitution du dataset de référence |
| `test` | 1 000 images sans annotations disponibles | Non exploité pour l'instant |

Les fichiers `train.csv` et `dev.csv` contiennent les annotations associées aux images, notamment la région anatomique, la projection et les caractéristiques d'acquisition.

Les données sources sont conservées dans leur organisation d'origine, sans modification.

## Préparation et harmonisation des annotations

Les annotations IRMA sont plus détaillées que les informations nécessaires à notre tâche de classification. Leur structure est également irrégulière, avec un nombre variable de champs.

Pour évaluer le modèle, seules deux informations sont nécessaires :

- Le nom du fichier image.
- La classe anatomique attendue.

Une correspondance entre les libellés IRMA et les classes du modèle ConvNeXt-Tiny a donc été définie.

| Annotation IRMA | Classe du modèle |
|---|---|
| chest | CHEST |
| hand | HAND |
| knee | KNEE |
| lumbar spine | LUMBAR_SPINE |
| cervical spine | CERVICAL_SPINE |
| foot | FOOT |
| ankle joint | ANKLE |
| shoulder | SHOULDER |
| thoracic spine | DORSAL_SPINE |
| pelvis | PELVIS |


## Automatisation de la génération des datasets

Un script `generer_dataset.py` a été développé afin d'automatiser la préparation des annotations et la constitution des datasets de travail.

Cette automatisation permet de générer plusieurs datasets à partir des mêmes données sources, en faisant varier les critères de sélection.

Les principales étapes du script sont :

1. Lecture du fichier CSV source.
2. Harmonisation des annotations avec les classes du modèle.
3. Exclusion des correspondances multiples ou non identifiées.
4. Vérification de la disponibilité des images.
5. Échantillonnage aléatoire d'un nombre défini d'images par classe, sans remise.
6. Copie des images sélectionnées dans un nouveau dossier.
7. Génération du fichier `labels.csv` et de la documentation du dataset.



### Paramétrage

Les paramètres de génération sont centralisés au début du script :

```python
SOURCE = "dev"
NOM_DATASET = "benchmark_v1"

IMAGES_PAR_CLASSE = 20
SEED = 42
```

| Paramètre | Rôle |
|---|---|
| `SOURCE` | Partition du dataset original à exploiter |
| `NOM_DATASET` | Nom et version du dataset généré |
| `IMAGES_PAR_CLASSE` | Nombre d'images à sélectionner par classe |
| `SEED` | Graine utilisée pour la sélection aléatoire |
| `MAPPING_CLASSES` | Correspondance entre les annotations sources et les classes retenues |

La graine aléatoire (*seed*) permet de reproduire la même sélection d'images à partir des mêmes données, paramètres et conditions d'exécution.

Le nombre d'images par classe est paramétrable afin de construire des datasets équilibrés et d'éviter qu'une région anatomique surreprésentée influence excessivement les métriques globales.

## Organisation des données

```text
datasets/
├── irma_original/
│   ├── train/
│   ├── dev/
│   ├── test/
│   ├── train.csv
│   └── dev.csv
│
├── generer_dataset.py
│
└── benchmark_v1/
    ├── images/
    ├── labels.csv
    └── README.md
```

Chaque dataset généré est indépendant et contient ses propres images, annotations et informations de génération.

Le script empêche l'écrasement d'un dataset existant afin de préserver les versions précédentes.

## Génération du premier dataset de référence

Le script est exécuté depuis le dossier `datasets` :

```powershell
python generer_dataset.py
```

Le premier dataset, nommé `benchmark_v1`, a été généré avec succès.

| Caractéristique | Valeur |
|---|---|
| Source | IRMA |
| Partition | dev |
| Nombre de classes | 10 |
| Nombre d'images par classe | 20 |
| Nombre total d'images | 200 |
| Seed | 42 |
| Distribution | Équilibrée |

Ce dataset sera utilisé comme référence commune pour les futures évaluations du modèle.

## Traçabilité et reproductibilité

La génération automatisée des datasets s'inscrit dans une démarche MLOps et répond à plusieurs objectifs :

- **Paramétrage** : générer différents datasets sans modifier la logique du script.
- **Reproductibilité** : reproduire une sélection grâce aux paramètres de génération et à la graine aléatoire.
- **Traçabilité** : conserver les informations de génération dans un fichier `README.md` produit automatiquement.
- **Versionnement** : identifier et conserver chaque dataset indépendamment.
- **Comparabilité** : utiliser exactement les mêmes images et annotations pour évaluer les différentes versions du modèle.

Lors des futures expérimentations, le dataset utilisé pourra être référencé dans MLflow afin d'associer chaque résultat d'évaluation à un jeu de données identifié.