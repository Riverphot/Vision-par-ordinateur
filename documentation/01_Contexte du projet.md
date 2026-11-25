# Projet d’industrialisation d’une API de vision par ordinateur

Ce projet s'inscrit dans le module Industrialisation de l'IA dans le Cloud. Il vise à expérimenter l'intégration, l'optimisation, la sécurisation et le déploiement d'un modèle de vision par ordinateur au sein d'une API web.

L'objectif est de valider une architecture technique performante et évolutive, pouvant à terme accueillir des modèles d'analyse d'imagerie médicale plus avancés.

## Choix du sujet

**Classification anatomique de radiographies à l'aide d'un modèle de vision par ordinateur pré-entraîné.**

Le projet prend la forme d'un Proof of Concept (POC) volontairement limité à l'identification des régions anatomiques sur des radiographies, sans objectif de diagnostic médical.

Ce cas d'usage permet de se concentrer sur les enjeux techniques de l'industrialisation d'un modèle d'IA, avant d'envisager des fonctionnalités plus complexes, comme la détection et la localisation d'anomalies médicales.

Problématique : Comment concevoir, optimiser et déployer une API de classification d'images médicales, tout en maîtrisant les contraintes de performance, de sécurité et de consommation des ressources ?

### Périmètre du POC

- Classification de radiographies conventionnelles selon la région anatomique représentée (épaule, genou, main, thorax, etc.).
- Prise en charge des images JPEG, PNG et DICOM.
- Extraction des pixels et des métadonnées DICOM, avec comparaison indicative aux prédictions du modèle.
- Utilisation d'un modèle pré-entraîné disponible sur Hugging Face.
- Développement et déploiement d'une API Python.
- Évaluation des prédictions et des performances techniques à partir d'un jeu d'images de test autorisées, sans données personnelles identifiantes.


## Recherche et sélection du modèle

Avant le développement de l’API, une recherche de modèles pré-entraînés a été effectuée . Cette première étude permet de vérifier la faissabilité de notre projet.4

Après recherche sur **Hugging Face**, trois modèles ont été présélectionnés :

| Modèle | Caractéristiques 
|---|---
| [InfoBayAI — ResNet50 Xray Anatomy Classifier](https://huggingface.co/InfoBayAI/restnet50-Xray-Anatomy-Classifier) | 7 catégories anatomiques, PyTorch, performances annoncées élevées 
| [emon5122 — X-Ray Body-Part Classifier](https://huggingface.co/emon5122/xray-bodypart-classifier) | 33 catégories anatomiques, ConvNeXt-Tiny, fichiers PyTorch et ONNX 
| [RadiologyVisionNet-Clinical](https://huggingface.co/toolevalxm/RadiologyVisionNet-Clinical) | Analyse radiologique plus large, incluant des anomalies cliniques 

Le modèle **`emon5122/xray-bodypart-classifier`** a été sélectionné pour son adéquation avec la classification anatomique détaillée et la disponibilité de sa version PyTorch, permettant d’expérimenter l’intégration, l’optimisation et la comparaison des performances.


## Constitution d'un jeu d'images de test

Un jeu de **16 fichiers** a été constitué pour vérifier la réception, le prétraitement et la classification des radiographies par l'API. Il permet également de tester l'extraction des images et des métadonnées DICOM, ainsi que le traitement des formats non pris en charge ou hors périmètre.

Le détail de chaque image et de leur traçabilité est enregistré dans un fichier `inventaire(1).csv` : liste détaillée des fichiers, de leur format, de leur région ou modalité et de leur provenance.

### Composition

| Ensemble | Nombre | Contenu | Usage principal |
|---|---:|---|---|
| JPEG | 7 | Thorax, main, genoux, pied, crâne et deux poignets | Classification anatomique |
| PNG | 2 | Copies converties des images de thorax et de main | Cohérence des prédictions entre formats |
| DICOM radiographiques | 5 | Thorax (3), genou (1), épaule (1) ; modalités CR, DX ou OT | Décodage DICOM, métadonnées et classification |
| DICOM hors périmètre | 2 | Scanner CT et IRM MR de démonstration | Gestion des modalités non radiographiques |
| **Total** | **16** | | |
