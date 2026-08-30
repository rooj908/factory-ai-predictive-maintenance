import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

from dotenv import load_dotenv


# ============================================================
# CONFIG
# ============================================================

DATA_PATH = "data/factory_data.csv"
TARGET = "failure_within_24h"

load_dotenv()


# ============================================================
# LOAD DATA
# ============================================================

print("\nLoading dataset...")

df = pd.read_csv(DATA_PATH)

df["timestamp"] = pd.to_datetime(df["timestamp"])

df = df.sort_values("timestamp").reset_index(drop=True)

print("Dataset shape:", df.shape)


# ============================================================
# TIME-BASED TRAIN / TEST SPLIT
# ============================================================

split_index = int(len(df) * 0.80)

train = df.iloc[:split_index].copy()
test = df.iloc[split_index:].copy()

print("\nTrain:", train.shape)
print("Test :", test.shape)

print("\nTrain max timestamp:",
      train["timestamp"].max())

print("Test min timestamp:",
      test["timestamp"].min())


# ============================================================
# LEAKAGE CHECK
# ============================================================

drop_cols = [
    "timestamp",
    "machine_id",
    "failure_within_24h",
    "failure_type",
    "estimated_repair_cost",
    "rul_hours"
]

X_train = train.drop(columns=drop_cols)

y_train = train[TARGET]

X_test = test.drop(columns=drop_cols)

y_test = test[TARGET]

print("\nTarget in features:",
      TARGET in X_train.columns)


# ============================================================
# COLUMN TYPES
# ============================================================

categorical_cols = X_train.select_dtypes(
    include=["object"]
).columns.tolist()

numeric_cols = X_train.select_dtypes(
    include=np.number
).columns.tolist()

print("\nCategorical columns:")
print(categorical_cols)

print("\nNumeric columns:")
print(numeric_cols)


# ============================================================
# PREPROCESSOR
# ============================================================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "cat",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_cols
        ),
        (
            "num",
            "passthrough",
            numeric_cols
        )
    ]
)


# ============================================================
# RANDOM FOREST
# ============================================================

model = Pipeline([
    (
        "preprocessor",
        preprocessor
    ),
    (
        "classifier",
        RandomForestClassifier(
            n_estimators=150,
            random_state=42,
            class_weight="balanced",
            n_jobs=-1
        )
    )
])


print("\nTraining Random Forest...")

model.fit(X_train, y_train)


# ============================================================
# EVALUATION
# ============================================================

y_pred = model.predict(X_test)

y_prob = model.predict_proba(X_test)[:, 1]

print("\n" + "=" * 60)
print("RANDOM FOREST RESULTS")
print("=" * 60)

print(
    classification_report(
        y_test,
        y_pred,
        zero_division=0
    )
)

print(
    "ROC-AUC:",
    round(roc_auc_score(y_test, y_prob), 4)
)

print("\nConfusion Matrix:")

print(
    confusion_matrix(
        y_test,
        y_pred
    )
)


# ============================================================
# XAI FEATURE IMPORTANCE
# ============================================================

rf_model = model.named_steps["classifier"]

processed_column_names = (
    model
    .named_steps["preprocessor"]
    .get_feature_names_out()
)

feature_importance = pd.DataFrame({
    "Feature": processed_column_names,
    "Importance": rf_model.feature_importances_
})

feature_importance = (
    feature_importance
    .sort_values(
        "Importance",
        ascending=False
    )
)

print("\n" + "=" * 60)
print("TOP 15 IMPORTANT FEATURES")
print("=" * 60)

print(
    feature_importance.head(15).to_string(
        index=False
    )
)


# ============================================================
# SAVE XAI RESULTS
# ============================================================

os.makedirs("models", exist_ok=True)

feature_importance.to_csv(
    "models/feature_importance.csv",
    index=False
)


# ============================================================
# XAI GRAPH
# ============================================================

top_features = (
    feature_importance
    .head(10)
    .sort_values("Importance")
)

plt.figure(figsize=(10, 6))

plt.barh(
    top_features["Feature"],
    top_features["Importance"]
)

plt.xlabel("Feature Importance")

plt.ylabel("Feature")

plt.title(
    "Top 10 Features for Failure Prediction"
)

plt.tight_layout()

os.makedirs("reports", exist_ok=True)

plt.savefig(
    "reports/feature_importance.png",
    dpi=150
)

plt.show()


# ============================================================
# SAMPLE PREDICTION
# ============================================================

sample = X_test.iloc[0:1]

failure_probability = (
    model.predict_proba(sample)[0][1]
)

prediction = model.predict(sample)[0]

print("\n" + "=" * 60)
print("SAMPLE MACHINE PREDICTION")
print("=" * 60)

print(
    "Failure probability:",
    round(failure_probability, 3)
)

print(
    "Prediction:",
    "FAILURE RISK"
    if prediction == 1
    else "NORMAL"
)


# ============================================================
# SAVE MODEL
# ============================================================

import joblib

joblib.dump(
    model,
    "models/random_forest_model.pkl"
)

print("\nModel saved:")
print("models/random_forest_model.pkl")

print("\nStage I + Random Forest + XAI completed successfully.")
