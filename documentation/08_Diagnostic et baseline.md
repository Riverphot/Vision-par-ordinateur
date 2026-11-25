# Diagnostic et baseline du modèle

Avant d'engager les optimisations du modèle, une évaluation de référence (*baseline*) est nécessaire pour mesurer leurs effets sur deux dimensions complémentaires :

- **Qualité prédictive** : capacité du modèle à identifier correctement les régions anatomiques.
- **Performance technique** : rapidité d'exécution, débit et taille du modèle.

Deux notebooks exploratoires permettent d'étudier ces dimensions, de comprendre le comportement initial du modèle et de définir les protocoles d'évaluation. Ces protocoles sont ensuite automatisés afin de comparer les différentes configurations du modèle.

## Configuration initiale du modèle

| Élément | Configuration |
|---|---|
| Tâche | Classification anatomique multiclasses de radiographies |
| Modèle | [emon5122/xray-bodypart-classifier](https://huggingface.co/emon5122/xray-bodypart-classifier) |
| Architecture | ConvNeXt-Tiny |
| Poids | `models/xray/best.pt` |
| Moteur d'inférence | PyTorch FP32 |
| Matériel | CPU local |
| Environnement | Conda `vision-api`, Python 3.11, PyTorch, Torchvision, timm |
| Dataset | `benchmark_v1`, issu de la partition `dev` d'IRMA |
| Composition | 200 images, 10 classes, 20 images par classe |
| Classes de sortie | 33, sans restriction aux classes du benchmark |
| Entrée | RGB, tenseur `[1, 3, 224, 224]` |
| Normalisation | Mean = [0.485, 0.456, 0.406] ; std = [0.229, 0.224, 0.225] |
| Prétraitement initial | Resize proportionnel puis Center Crop |

## 1. Diagnostic prédictif

### Objectif et rôle du notebook

Le notebook `notebooks/01_evaluation_predictive.ipynb` restitue la démarche exploratoire menée sur le modèle dans sa configuration initiale.

Son objectif est de comprendre les premiers résultats, de repérer les difficultés du modèle et de justifier les métriques et les expérimentations à retenir avant optimisation.


#### Déroulement de l'investigation

| Étape | Analyse | Objectif |
|---|---|---|
| 1 | Exploration du dataset | Comprendre les données et leurs particularités |
| 2 | Premières métriques globales | Établir la référence prédictive |
| 3 | Performances par classe | Repérer les régions difficiles à reconnaître |
| 4 | Matrice de confusion et erreurs Top-1 | Identifier les confusions récurrentes |
| 5 | Examen visuel des erreurs | Questionner la proximité anatomique |
| 6 | Progression Top-K et contenu du Top-5 | Comprendre l'intérêt des prédictions supplémentaires |
| 7 | Regroupement exploratoire du rachis | Estimer l'effet d'une taxonomie moins détaillée |
| 8 | Justification des métriques | Retenir des indicateurs adaptés à la tâche |
| 9 | Visualisation des prétraitements | Identifier une éventuelle perte d'informations |
| 10 | Synthèse des hypothèses | Orienter les expérimentations |

### Résultats du diagnostic prédictif

La première évaluation du modèle dans sa configuration initiale (PyTorch FP32, Center Crop) donne les résultats suivants :

| Métrique | Baseline | Performance annoncée par l'auteur* |
|---|---:|---:|
| Accuracy Top-1 | 41,50 % | 70,40 % |
| Accuracy Top-5 | 83,00 % | 94,00 % |
| Précision macro | 63,24 % | — |
| Rappel macro | 41,50 % | — |
| F1-score macro | 46,81 % | — |

*Performances annoncées à confirmer sur la fiche source du modèle.

Les performances sont inférieures aux valeurs annoncées. Cette comparaison reste indicative, les jeux de données étant différents.

L'analyse exploratoire réalisée dans le notebook met en évidence plusieurs éléments :

- Un écart important entre le Top-1 et le Top-5 : le modèle propose régulièrement la bonne classe sans la sélectionner en première position.
- Des performances hétérogènes selon les régions anatomiques.
- Des confusions récurrentes entre des régions anatomiquement proches.
- Des associations fréquentes dans le Top-5, soulevant la question de la granularité des 33 classes du modèle.
- Une perte potentielle d'informations anatomiques liée au recadrage central des images.

### Choix des métriques prédictives

Le choix des métriques dépend de la tâche à réaliser, mais également de son contexte d'utilisation.

Dans le domaine de l'imagerie médicale, une radiographie peut représenter plusieurs régions anatomiques simultanément. Certaines catégories peuvent donc se chevaucher ou présenter des caractéristiques visuelles proches.

Une classification différente de l'annotation attendue n'est ainsi pas nécessairement incohérente sur le plan anatomique.

L'analyse exploratoire confirme l'intérêt de conserver conjointement l'Accuracy Top-1 et l'Accuracy Top-5, selon deux usages distincts : la catégorisation automatique et l'assistance à la classification.

| Métrique | Justification |
|---|---|
| Accuracy Top-1 | Évaluer la capacité à identifier précisément la région anatomique attendue. |
| Accuracy Top-5 | Évaluer la capacité à proposer la bonne région parmi plusieurs catégories possibles. |
| Précision macro | Évaluer la fiabilité moyenne des classifications par région. |
| Rappel macro | Évaluer la capacité à reconnaître les différentes régions anatomiques. |
| F1-score macro | Disposer d'un indicateur synthétique équilibrant précision et rappel. |
| Matrice de confusion | Identifier et caractériser les erreurs de classification. |

Les moyennes macro permettent d'accorder la même importance à chaque classe évaluée.

L'évaluation conserve une correspondance stricte entre la prédiction et l'annotation afin de garantir la comparabilité des résultats.

### Hypothèses et orientations des expérimentations

| Axe d'investigation | Hypothèse | Expérimentation envisagée |
|---|---|---|
| Prétraitement | Le recadrage central peut supprimer des repères anatomiques utiles. | Comparer Center Crop, Resize et Padding. |
| Granularité anatomique | Certaines catégories sont trop proches ou se chevauchent. | Évaluer l'effet d'un regroupement en familles anatomiques. |
| Adaptation du modèle | La taxonomie et les données initiales pourraient être insuffisamment adaptées au contexte d'utilisation. | Envisager un fine-tuning sur une classification simplifiée. |

## 2. Diagnostic des performances techniques

### Objectif et rôle du notebook

Le notebook `notebooks/02_diagnostic_performances.ipynb` permet d'explorer les performances techniques du modèle avant optimisation.

Contrairement au notebook prédictif, il exécute directement le modèle afin d'expérimenter différentes méthodes de mesure et de définir un protocole reproductible.

#### Déroulement de l'investigation

| Étape | Analyse | Objectif |
|---|---|---|
| 1 | Configuration matérielle et logicielle | Identifier les conditions d'exécution |
| 2 | Nombre de paramètres et taille du modèle | Évaluer les besoins de stockage et l'intérêt d'une réduction de précision |
| 3 | Premières inférences | Observer le phénomène de warm-up |
| 4 | Distribution des latences | Étudier la rapidité et la régularité du modèle |
| 5 | Débit et batching | Identifier l'influence de la taille des lots |
| 6 | Consommation mémoire | Explorer les ressources nécessaires et les limites de leur mesure |
| 7 | Synthèse | Définir le protocole technique de référence |

### Résultats du diagnostic technique

Les premières mesures ont été réalisées sur le modèle PyTorch FP32, exécuté sur CPU avec quatre threads PyTorch.

| Indicateur | Résultat exploratoire |
|---|---:|
| Nombre de paramètres | 27 845 505 |
| Taille du fichier de poids | 111,44 Mo |
| Taille mémoire des paramètres | 111,38 Mo |
| Première inférence | 410,73 ms |
| Latence moyenne après warm-up | 85,13 ms |
| Latence P50 | 82,19 ms |
| Latence P95 | 116,07 ms |
| Latence P99 | 160,44 ms |
| Débit Batch 1 | 11,08 images/s |
| Débit Batch 8 | 11,93 images/s |
| Débit Batch 16 | 11,95 images/s |

L'analyse exploratoire met en évidence plusieurs éléments :

- Le phénomène de warm-up entraîne une première inférence sensiblement plus lente que les suivantes.
- La latence varie entre les exécutions, ce qui justifie l'utilisation des percentiles en complément de la moyenne.
- Le traitement par lots apporte un gain de débit limité sur le CPU utilisé.
- La précision FP32 représente quatre octets par paramètre. Une réduction de cette précision pourrait diminuer la taille du modèle.
- Les mesures ponctuelles de mémoire du processus Python sont difficiles à interpréter en raison de la gestion et de la réutilisation des ressources mémoire.

### Choix des métriques techniques

| Métrique | Justification |
|---|---|
| Taille du modèle (Mo) | Évaluer les contraintes de stockage et les gains liés à la quantification. |
| Latence moyenne | Mesurer le temps moyen nécessaire à une inférence. |
| Latence P50 | Caractériser le temps d'inférence médian. |
| Latence P95 et P99 | Identifier les temps d'exécution les plus élevés. |
| Écart-type | Évaluer la variabilité des temps d'exécution. |
| Latence par lot | Mesurer le temps nécessaire au traitement simultané de plusieurs images. |
| Débit (images/s) | Évaluer la capacité de traitement massif du modèle. |

Les mesures de mémoire RSS ne sont pas retenues dans le protocole automatisé, leur fiabilité étant insuffisante avec la méthode expérimentée.

### Protocole de mesure retenu

| Paramètre | Configuration |
|---|---|
| Image de référence | `10339.png` |
| Warm-up | 5 inférences avant chaque série |
| Mesure de latence | 100 inférences |
| Tailles de batch | 1, 2, 4, 8 et 16 |
| Mesures par taille de batch | 20 |
| Périmètre | Inférence du réseau neuronal uniquement |

L'évaluation technique repose sur une image prétraitée, répétée pour constituer les différents lots.

Le modèle recevant systématiquement un tenseur de dimensions identiques, il n'est pas nécessaire d'utiliser l'intégralité du dataset pour mesurer les performances du réseau neuronal.

La lecture des fichiers, le décodage, le prétraitement et le post-traitement sont exclus de ces mesures. 

### Hypothèses et orientations des expérimentations

| Axe d'investigation | Hypothèse | Expérimentation envisagée |
|---|---|---|
| Moteur d'inférence | Un moteur optimisé pour l'inférence pourrait réduire la latence. | Comparer PyTorch et ONNX Runtime. |
| Précision numérique | Une précision réduite pourrait diminuer la taille du modèle et accélérer les calculs. | Expérimenter la quantification INT8. |
| Matériel | Le GPU pourrait améliorer les performances d'exécution. | Comparer CPU et GPU. |
| Batching | Le traitement par lots pourrait améliorer l'utilisation des ressources. | Comparer les débits selon le moteur et le matériel. |

## 3. Automatisation des évaluations

Les protocoles définis lors des diagnostics prédictif et technique sont intégrés dans des scripts Python afin de permettre leur exécution automatique et reproductible.

L'objectif est de mesurer conjointement la qualité prédictive et les performances techniques d'une même configuration du modèle.

### Environnement

Les bibliothèques nécessaires sont installées dans l'environnement Conda `vision-api` :

```powershell
pip install scikit-learn matplotlib pandas jupyter ipykernel psutil
```

Le kernel Jupyter utilise ce même environnement :

```powershell
python -m ipykernel install --user --name vision-api --display-name "Python (vision-api)"
```

### Organisation des scripts

```text
evaluation/
├── configuration.py
├── experimenter.py
├── evaluer_modele.py
├── evaluer_performances.py
├── tracking_mlflow.py
└── resultats/
```

| Fichier | Responsabilité |
|---|---|
| `configuration.py` | Centralisation des paramètres de l'expérimentation et des protocoles d'évaluation |
| `experimenter.py` | Chargement du modèle, orchestration des évaluations, exports et déclenchement du tracking |
| `evaluer_modele.py` | Classification du benchmark et calcul des métriques prédictives |
| `evaluer_performances.py` | Mesure de la latence, du débit et de la taille du modèle |


Les modules d'évaluation sont indépendants du stockage des résultats. Ils reçoivent les fonctions d'inférence en paramètres, afin de faciliter l'intégration de nouveaux moteurs.

### Configuration des expérimentations

Le fichier `configuration.py` permet de modifier les paramètres sans intervenir dans les scripts d'évaluation.

```python
# Configuration du modèle
NOM_MODELE = "convnext_tiny"
VERSION_MODELE = "pytorch_fp32"
MOTEUR_INFERENCE = "pytorch"
PRECISION_NUMERIQUE = "FP32"
DEVICE = "cpu"

# Configuration prédictive
NOM_DATASET = "benchmark_v1"
PRETRAITEMENT = "resize"
TOP_K = 5

# Configuration technique
NB_WARMUP = 5
NB_MESURES_LATENCE = 100
NB_MESURES_BATCH = 20
TAILLES_BATCH = (1, 2, 4, 8, 16)
IMAGE_TEST_TECHNIQUE = "10339.png"

# Pilotage
EVALUATION_PREDICTIVE = True
EVALUATION_TECHNIQUE = True
MLFLOW_ACTIF = True
```

Les deux évaluations peuvent être activées indépendamment ou conjointement.


### Fonctionnement général


Lors d'une expérimentation complète, l'évaluation technique est réalisée avant l'évaluation prédictive. Les deux évaluations utilisent le même modèle et la même configuration.

Exécution depuis la racine du projet :

```powershell
python evaluation/experimenter.py
```

### Exports et traçabilité locale

Chaque expérimentation est identifiée par la version du modèle, le dataset, le prétraitement et un horodatage.

Les résultats prédictifs et techniques sont regroupés dans un même dossier.

```text
evaluation/resultats/
└── pytorch_fp32_benchmark_v1_center_crop_20261120_190403_672891/
    ├── configuration.json
    ├── metrics.json
    ├── predictions.csv
    ├── metrics_par_classe.csv
    ├── matrice_confusion.csv
    ├── latences.csv
    └── batching.csv
```

| Fichier | Contenu |
|---|---|
| `configuration.json` | Configuration du modèle, paramètres, protocoles, environnement et empreintes SHA-256 |
| `metrics.json` | Ensemble des métriques prédictives et techniques |
| `predictions.csv` | Prédictions individuelles et Top-5 sérialisé en JSON |
| `metrics_par_classe.csv` | Précision, rappel, F1-score et effectifs par classe |
| `matrice_confusion.csv` | Croisement des classes réelles et prédites |
| `latences.csv` | Mesures individuelles des latences |
| `batching.csv` | Latence, temps par image et débit selon la taille du lot |

Les dossiers ne sont pas écrasés. Les exports permettent d'historiser les résultats, de les comparer et de les exploiter sans relancer les évaluations.


## 4. Validation du processus automatisé

Une première expérimentation complète a été réalisée sur la configuration initiale PyTorch FP32 avec Center Crop.


### Baseline globale automatisée

| Métrique | Résultat |
|---|---:|
| Accuracy Top-1 | 41,50 % |
| Accuracy Top-5 | 83,00 % |
| Précision macro | 63,24 % |
| Rappel macro | 41,50 % |
| F1-score macro | 46,81 % |
| Taille du modèle | 111,44 Mo |
| Latence moyenne | 81,52 ms |
| Latence P50 | 80,40 ms |
| Latence P95 | 99,68 ms |
| Latence P99 | 119,53 ms |
| Débit Batch 1 | 12,74 images/s |
| Débit Batch 2 | 14,38 images/s |
| Débit Batch 4 | 15,55 images/s |
| Débit Batch 8 | 15,37 images/s |
| Débit Batch 16 | 8,01 images/s |

Cette expérimentation constitue la première référence globale automatisée du projet.

