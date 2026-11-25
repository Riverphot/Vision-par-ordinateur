import mlflow

from mlflow import MlflowClient
from mlflow.entities import LoggedModelInput

mlflow.set_tracking_uri("http://127.0.0.1:5000")

client = MlflowClient()

MODEL_ID = "m-602a55cecd5945cc9d6866d62ec649c0"

runs = client.search_runs(
    experiment_ids=["1"],
    filter_string="params.version_modele = 'pytorch_fp32'"
)

for run in runs:

    # Vérification des associations existantes
    modeles_associes = {
        m.model_id
        for m in run.inputs.model_inputs
    }

    if MODEL_ID not in modeles_associes:

        client.log_inputs(
            run_id=run.info.run_id,
            models=[
                LoggedModelInput(model_id=MODEL_ID)
            ]
        )

        print(f"Modèle associé : {run.info.run_name}")

    else:
        print(f"Déjà associé : {run.info.run_name}")

print("\nAssociation terminée.")