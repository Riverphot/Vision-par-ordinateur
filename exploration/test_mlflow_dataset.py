import sys
from pathlib import Path
import mlflow

RACINE_PROJET = Path(__file__).resolve().parents[1]

sys.path.insert(
    0, str(RACINE_PROJET / "evaluation")
)

from tracking_mlflow import charger_dataset_mlflow

mlflow.set_tracking_uri("http://127.0.0.1:5000")

dataset = charger_dataset_mlflow("benchmark_v1")

runs = mlflow.search_runs(
    experiment_ids=["1"],
    filter_string="params.dataset = 'benchmark_v1'"
)

for run_id in runs["run_id"]:

    with mlflow.start_run(run_id=run_id):

        mlflow.log_input(
            dataset,
            context="evaluation"
        )

        print(f"Dataset associé au Run : {run_id}")

print("\nAssociation terminée.")