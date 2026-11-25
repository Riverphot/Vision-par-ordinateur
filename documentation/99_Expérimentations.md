# Expérimentation sur les prétraitements

Le prétraitement initial redimensionne le plus petit côté de l'image à 224 pixels, puis extrait un carré central de 224 × 224 pixels. Sur une radiographie allongée, cette opération peut supprimer des repères anatomiques situés en périphérie, tels que les côtes ou le bassin.

Trois méthodes ont été comparées afin d'isoler l'influence de cette transformation, sans modifier les poids du modèle.

| Paramètre | Méthode | Avantage | Limite |
|---|---|---|---|
| `center_crop` | Resize proportionnel + recadrage central | Proportions conservées | Perte d'une partie de l'image |
| `resize` | Resize direct en 224 × 224 | Image entière conservée | Déformation des proportions |
| `padding` | Resize proportionnel + bandes noires centrées | Image entière et proportions conservées | Surface utile réduite, ajout de bordures |

Le padding est réalisé avec `ImageOps.pad()` de Pillow, avec interpolation bilinéaire, fond noir et centrage. Les trois méthodes utilisent ensuite la même conversion RGB, la même normalisation et les mêmes dimensions d'entrée.

### Adaptation du code

Le module `inference.py` sépare désormais les transformations géométriques de la conversion en tenseur et de la normalisation. La fonction `preparer_image()` sélectionne la méthode ; `predict()` conserve la même logique d'inférence.

```python
predictions = predict(
    image,
    top_k=TOP_K,
    pretraitement=PRETRAITEMENT
)
```

Le Center Crop reste le comportement par défaut de l'API. Une nouvelle évaluation avec ce prétraitement a reproduit exactement les métriques initiales, validant la non-régression des métriques après réorganisation du code.

### Résultats comparatifs

Les trois évaluations ont été réalisées sur les mêmes 200 radiographies le 20 novembre 2026 et enregistrées automatiquement dans MLflow.

| Métrique | Center Crop | Resize | Padding |
|---|---:|---:|---:|
| Accuracy Top-1 | 41,50 % | **57,50 %** | 40,00 % |
| Accuracy Top-5 | 83,00 % | **85,00 %** | 84,50 % |
| Précision macro | 63,24 % | **81,49 %** | 81,13 % |
| Rappel macro | 41,50 % | **57,50 %** | 40,00 % |
| F1-score macro | 46,81 % | **65,01 %** | 49,45 % |

Le Resize direct améliore l'Accuracy Top-1 de 16 points de pourcentage (32 classifications correctes supplémentaires), le Top-5 de 2 points et le F1-score macro de 18,20 points par rapport au Center Crop.

| Confusion observée | Center Crop | Resize |
|---|---:|---:|
| Rachis dorsal → rachis lombaire | 11 | 6 |
| Rachis lombaire → bassin | 9 | 1 |
| Rachis dorsal → bassin | 5 | 0 |

La conservation du contexte anatomique pourrait expliquer une partie de ces gains. Le Padding, malgré ses avantages géométriques, n'améliore pas le Top-1. La réduction de la surface utile ou les bordures ajoutées pourraient perturber le modèle. Ces explications restent des hypothèses, non des causes démontrées.

**Décision provisoire :** retenir le Resize direct comme candidat pour la suite des optimisations. Le Center Crop demeure la référence initiale et le comportement actuel de l'API tant qu'une décision finale n'a pas été appliquée au code. Le benchmark ayant servi à sélectionner le prétraitement, un jeu indépendant serait nécessaire pour confirmer le gain.