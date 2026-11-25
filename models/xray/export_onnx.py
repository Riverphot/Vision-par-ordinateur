
"""
Export et conversion du modèle ConvNeXt-Tiny vers ONNX.

Formats disponibles :
    fp32 : export PyTorch vers ONNX FP32
    fp16 : conversion ONNX FP32 vers ONNX FP16

La quantification INT8 sera ajoutée avec une calibration dédiée.
"""

import sys
import argparse
from pathlib import Path

import torch
import onnx
import onnxruntime as ort
import numpy as np

# =============================================
# CHEMINS
# =============================================

RACINE_PROJET = Path(__file__).resolve().parents[2]
DOSSIER_MODELE = Path(__file__).resolve().parent

sys.path.insert(0, str(RACINE_PROJET))

FICHIER_FP32 = DOSSIER_MODELE / "convnext_tiny_export.onnx"
FICHIER_FP16 = DOSSIER_MODELE / "convnext_tiny_fp16.onnx"


# =============================================
# EXPORT PYTORCH VERS ONNX FP32
# =============================================

def exporter_fp32():

    from inference import model

    model.cpu()
    model.eval()

    # Tenseur de référence utilisé pour l'export
    exemple = torch.randn(1, 3, 224, 224)

    print("\n--- Export PyTorch vers ONNX FP32 ---")

    with torch.inference_mode():
        torch.onnx.export(
            model,
            exemple,
            str(FICHIER_FP32),
            input_names=["images"],
            output_names=["logits"],
            opset_version=17,
            dynamic_axes={
                "images": {0: "batch_size"},
                "logits": {0: "batch_size"}
            },
            dynamo=False
        )

    verifier_structure(FICHIER_FP32)
    verifier_equivalence_fp32(model, exemple)

    afficher_taille(FICHIER_FP32)


# =============================================
# CONVERSION ONNX FP32 VERS FP16
# =============================================

def convertir_fp16():

    from onnxconverter_common import float16

    if not FICHIER_FP32.exists():
        raise FileNotFoundError(
            "Le modèle ONNX FP32 est absent. "
            "Exécuter d'abord --format fp32."
        )

    print("\n--- Conversion ONNX FP32 vers FP16 ---")

    modele = onnx.load(str(FICHIER_FP32))

    # Conservation des entrées et sorties en FP32
    modele_fp16 = float16.convert_float_to_float16(
        modele,
        keep_io_types=True
    )

    onnx.save(modele_fp16, str(FICHIER_FP16))

    verifier_structure(FICHIER_FP16)
    afficher_taille(FICHIER_FP16)


# =============================================
# VERIFICATION DE LA STRUCTURE ONNX
# =============================================

def verifier_structure(fichier):

    onnx.checker.check_model(str(fichier))

    modele = onnx.load(str(fichier))

    print("Structure ONNX : valide")
    print(f"Opset : {modele.opset_import[0].version}")
    print(f"Nombre de nœuds : {len(modele.graph.node)}")


# =============================================
# VERIFICATION NUMERIQUE FP32
# =============================================

def verifier_equivalence_fp32(model, exemple):

    # Référence PyTorch
    with torch.inference_mode():
        sortie_pytorch = model(exemple).numpy()

    # Exécution ONNX Runtime sur CPU
    session = ort.InferenceSession(
        str(FICHIER_FP32),
        providers=["CPUExecutionProvider"]
    )

    sortie_onnx = session.run(
        ["logits"],
        {"images": exemple.numpy()}
    )[0]

    ecart_max = np.max(
        np.abs(sortie_pytorch - sortie_onnx)
    )

    memes_classes = (
        sortie_pytorch.argmax(axis=1)
        == sortie_onnx.argmax(axis=1)
    ).all()

    print(f"Ecart absolu maximal : {ecart_max:.8f}")
    print(f"Même classe Top-1 : {memes_classes}")

    np.testing.assert_allclose(
        sortie_pytorch,
        sortie_onnx,
        rtol=1e-3,
        atol=1e-4
    )

    print("Comparaison numérique FP32 : validée")


# =============================================
# AFFICHAGE DE LA TAILLE
# =============================================

def afficher_taille(fichier):

    taille = fichier.stat().st_size / 1e6

    print(f"Fichier : {fichier.name}")
    print(f"Taille : {taille:.2f} Mo")


# =============================================
# POINT D'ENTREE
# =============================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Export et conversion ONNX"
    )

    parser.add_argument(
        "--format",
        choices=["fp32", "fp16"],
        required=True,
        help="Format du modèle à générer"
    )

    args = parser.parse_args()

    if args.format == "fp32":
        exporter_fp32()

    elif args.format == "fp16":
        convertir_fp16()
