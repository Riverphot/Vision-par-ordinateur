import sys
from pathlib import Path

import mlflow
import mlflow.pytorch
import torch


# =============================================
# CONFIGURATION
# =============================================

RACINE_PROJET = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(RACINE_PROJET))

from inference import model

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("Evaluation predictive")


# =============================================
# ENREGISTREMENT DU MODELE
# =============================================

with mlflow.start_run(
    run_name="Enregistrement ConvNeXt-Tiny FP32"
):

    model_info = mlflow.pytorch.log_model(
        pytorch_model=model,
        name="convnext_tiny_pytorch_fp32",
        input_example=torch.zeros(
            1, 3, 224, 224
        ).numpy(),
        registered_model_name="xray_bodypart_classifier"
    )

    print("\n--- Modèle enregistré ---")
    print(f"Model ID : {model_info.model_id}")
    print(f"Model URI : {model_info.model_uri}")
    print(f"Version : {model_info.registered_model_version}")