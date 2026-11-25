# Création d'une interface de test avec Gradio

Après validation de l'API FastAPI avec Swagger, une interface graphique a été développée pour faciliter les tests et la visualisation des résultats du modèle.

L'objectif est de disposer d'un outil de démonstration permettant d'importer une radiographie, de visualiser l'image analysée et de consulter les prédictions et métadonnées associées.

## Choix de Gradio

**Gradio** est une bibliothèque Python permettant de créer rapidement des interfaces web interactives, notamment pour les applications d'intelligence artificielle.


Installation dans l'environnement `vision-api` :

```powershell
pip install gradio requests
```

La bibliothèque `requests` permet à l'interface de communiquer avec l'API REST via HTTP.

## Architecture 

Un module `gradio_ui.py` a été ajouté au projet afin de séparer l'interface graphique de la logique métier.

L'organisation repose sur les composants suivants :

| Module | Responsabilité |
|---|---|
| `gradio_ui.py` | Interface utilisateur, import des fichiers et présentation des résultats |
| `main.py` | API REST, validation des fichiers et orchestration |
| `dicom_utils.py` | Lecture et conversion des images DICOM |
| `inference.py` | Prétraitement du modèle et classification anatomique |

L'interface Gradio appelle l'endpoint `POST /upload` de FastAPI pour obtenir les prédictions. Elle ne communique pas directement avec le modèle.


## Fonctionnalités de l'interface

L'interface permet de :

- Importer une image JPEG, PNG ou DICOM.
- Afficher l'image transmise, après conversion si nécessaire.
- Déclencher la classification depuis un bouton.
- Visualiser les cinq meilleures prédictions et leurs scores.
- Consulter les métadonnées extraites des fichiers DICOM.
- Afficher les erreurs retournées par l'API lorsque le fichier n'est pas pris en charge.


