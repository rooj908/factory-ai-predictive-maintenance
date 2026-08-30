import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, recall_score


DATA_PATH = "data/factory_data.csv"

df = pd.read_csv(DATA_PATH)

target = "failure_within_24h"

X = df.drop(columns=[target])
y = df[target]

categorical_columns = [
    "machine_type",
    "operating_mode"
]

numeric_columns = [
    "vibration_rms",
    "temperature_motor",
    "current_phase_avg",
    "pressure_level",
    "rpm",
    "hours_since_maintenance",
    "ambient_temp"
]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            "passthrough",
            numeric_columns
        ),
        (
            "cat",
            OneHotEncoder(
                handle_unknown="ignore"
            ),
            categorical_columns
        )
    ]
)

model = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "classifier",
            RandomForestClassifier(
                n_estimators=200,
                max_depth=12,
                random_state=42,
                class_weight="balanced"
            )
        )
    ]
)

mlflow.set_tracking_uri(
    "sqlite:///mlflow.db"
)

mlflow.set_experiment(
    "Factory Predictive Maintenance"
)

with mlflow.start_run(
    run_name="RandomForest_Baseline"
):

    model.fit(
        X_train,
        y_train
    )

    probabilities = model.predict_proba(
        X_test
    )[:, 1]

    predictions = (
        probabilities >= 0.5
    ).astype(int)

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    roc_auc = roc_auc_score(
        y_test,
        probabilities
    )

    recall = recall_score(
        y_test,
        predictions
    )

    mlflow.log_params({
        "model": "RandomForest",
        "n_estimators": 200,
        "max_depth": 12,
        "class_weight": "balanced"
    })

    mlflow.log_metrics({
        "accuracy": accuracy,
        "roc_auc": roc_auc,
        "failure_recall": recall
    })

    mlflow.sklearn.log_model(
        model,
        "random_forest_model"
    )

    print("=" * 60)
    print("MLFLOW EXPERIMENT COMPLETE")
    print("=" * 60)

    print("Accuracy:", round(accuracy, 4))
    print("ROC-AUC:", round(roc_auc, 4))
    print("Failure Recall:", round(recall, 4))

