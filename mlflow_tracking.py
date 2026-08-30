import mlflow
import mlflow.sklearn
import joblib

from pathlib import Path


mlflow.set_tracking_uri(
    "sqlite:///mlflow.db"
)

mlflow.set_experiment(
    "Factory Predictive Maintenance"
)


def log_experiment(
    experiment_name,
    model,
    accuracy,
    roc_auc,
    recall,
    parameters
):

    with mlflow.start_run(
        run_name=experiment_name
    ):

        mlflow.log_params(parameters)

        mlflow.log_metric(
            "accuracy",
            accuracy
        )

        mlflow.log_metric(
            "roc_auc",
            roc_auc
        )

        mlflow.log_metric(
            "failure_recall",
            recall
        )

        mlflow.sklearn.log_model(
            model,
            "model"
        )

        print(
            f"MLflow run logged: {experiment_name}"
        )


if __name__ == "__main__":

    print("=" * 60)
    print("MLFLOW TRACKING READY")
    print("=" * 60)

    print("\nExperiment:")
    print("Factory Predictive Maintenance")

    print("\nTracking URI:")
    print("sqlite:///mlflow.db")

    print("\nNext step:")
    print("Run 3 model experiments.")
