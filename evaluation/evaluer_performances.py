"""
Evaluation des performances techniques du modèle.

Mesures :
    - Taille du modèle
    - Latence d'inférence
    - Percentiles P50, P95 et P99
    - Débit selon la taille des lots

Les mesures excluent la lecture et le prétraitement des images.
"""

import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch


# =============================================
# MESURE DE LA TAILLE DU MODELE
# =============================================

def mesurer_taille_modele(fichier_modele):
    """Mesure la taille du fichier du modèle en Mo."""

    return Path(fichier_modele).stat().st_size / 1e6


# =============================================
# MESURE DES LATENCES
# =============================================

def synchroniser(device):
    """Attend la fin des calculs lors d'une exécution CUDA."""

    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


def mesurer_latences(
    executer,
    tensor,
    device,
    nb_warmup=5,
    nb_mesures=100
):
    """Mesure les durées d'exécution en millisecondes."""

    latences = []

    with torch.inference_mode():

        # Warm-up
        for _ in range(nb_warmup):
            executer(tensor)

        synchroniser(device)

        # Mesures
        for _ in range(nb_mesures):

            synchroniser(device)
            debut = time.perf_counter()

            executer(tensor)

            synchroniser(device)
            fin = time.perf_counter()

            latences.append((fin - debut) * 1000)

    return np.array(latences)


# =============================================
# CALCUL DES INDICATEURS DE LATENCE
# =============================================

def calculer_metriques_latence(latences):
    """Calcule les indicateurs statistiques."""

    return {
        "latence_moyenne_ms": float(np.mean(latences)),
        "latence_min_ms": float(np.min(latences)),
        "latence_max_ms": float(np.max(latences)),
        "latence_ecart_type_ms": float(np.std(latences)),
        "latence_p50_ms": float(np.percentile(latences, 50)),
        "latence_p95_ms": float(np.percentile(latences, 95)),
        "latence_p99_ms": float(np.percentile(latences, 99))
    }


# =============================================
# MESURE DU BATCHING
# =============================================

def mesurer_batching(
    executer,
    tensor,
    device,
    tailles_batch=(1, 2, 4, 8, 16),
    nb_warmup=5,
    nb_mesures=20
):
    """Compare les performances selon la taille des lots."""

    resultats = []

    for taille in tailles_batch:

        print(f"Evaluation batch {taille}...")

        batch = tensor.repeat(taille, 1, 1, 1)

        latences = mesurer_latences(
            executer=executer,
            tensor=batch,
            device=device,
            nb_warmup=nb_warmup,
            nb_mesures=nb_mesures
        )

        moyenne = float(np.mean(latences))

        resultats.append({
            "batch_size": taille,
            "latence_lot_ms": moyenne,
            "temps_par_image_ms": moyenne / taille,
            "debit_images_s": taille * 1000 / moyenne
        })

        del batch

    return pd.DataFrame(resultats)


# =============================================
# EVALUATION TECHNIQUE COMPLETE
# =============================================

def evaluer_performances(
    executer,
    tensor,
    fichier_modele,
    device="cpu",
    nb_warmup=5,
    nb_mesures_latence=100,
    nb_mesures_batch=20,
    tailles_batch=(1, 2, 4, 8, 16)
):
    """Exécute le protocole complet d'évaluation technique."""

    print("\n--- Evaluation technique ---")

    # Taille du modèle
    metriques = {
        "taille_modele_mo": mesurer_taille_modele(
            fichier_modele
        )
    }

    # Latence
    print("Mesure de la latence...")

    latences = mesurer_latences(
        executer=executer,
        tensor=tensor,
        device=device,
        nb_warmup=nb_warmup,
        nb_mesures=nb_mesures_latence
    )

    metriques.update(
        calculer_metriques_latence(latences)
    )

    # Batching
    batching = mesurer_batching(
        executer=executer,
        tensor=tensor,
        device=device,
        tailles_batch=tailles_batch,
        nb_warmup=nb_warmup,
        nb_mesures=nb_mesures_batch
    )

    # Ajout des métriques de batching
    for _, ligne in batching.iterrows():

        taille = int(ligne["batch_size"])

        metriques[f"debit_batch_{taille}"] = float(
            ligne["debit_images_s"]
        )

        metriques[f"latence_batch_{taille}_ms"] = float(
            ligne["latence_lot_ms"]
        )

    return {
        "metrics": metriques,
        "latences": pd.DataFrame({
            "latence_ms": latences
        }),
        "batching": batching
    }

# =============================================
# EXECUTION AUTONOME DU SCRIPT
# =============================================

if __name__ == "__main__":

    import sys
    from PIL import Image

    # Racine du projet
    RACINE_PROJET = Path(__file__).resolve().parents[1]

    sys.path.insert(0, str(RACINE_PROJET))

    from inference import model, preparer_image

    # Configuration du test
    DEVICE = "cpu"
    PRETRAITEMENT = "center_crop"

    FICHIER_MODELE = (
        RACINE_PROJET / "models" / "xray" / "best.pt"
    )

    DOSSIER_IMAGES = (
        RACINE_PROJET / "datasets" / "benchmark_v1" / "images"
    )

    # Chargement du modèle
    model = model.to(DEVICE)
    model.eval()

    # Préparation d'une image de référence
    fichier_image = sorted(DOSSIER_IMAGES.glob("*.png"))[0]

    with Image.open(fichier_image) as image:
        tensor = preparer_image(
            image,
            pretraitement=PRETRAITEMENT
        ).to(DEVICE)

    print("Modèle : PyTorch FP32")
    print("Device :", DEVICE)
    print("Image :", fichier_image.name)

    # Evaluation technique
    resultats = evaluer_performances(
        executer=model,
        tensor=tensor,
        fichier_modele=FICHIER_MODELE,
        device=DEVICE
    )

    # Affichage des résultats
    print("\n--- Métriques techniques ---\n")

    for nom, valeur in resultats["metrics"].items():
        print(f"{nom:<35} : {valeur:.2f}")

    print("\n--- Résultats du batching ---\n")

    print(
        resultats["batching"]
        .round(2)
        .to_string(index=False)
    )