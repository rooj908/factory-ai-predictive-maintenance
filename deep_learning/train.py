import os
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score
)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping


# ============================================================
# 1. Load Dataset
# ============================================================

DATA_PATH = "data/factory_data.csv"

df = pd.read_csv(DATA_PATH)

print("=" * 60)
print("FACTORY AI - DEEP LEARNING MODEL")
print("=" * 60)

print("\nDataset Shape:", df.shape)


# ============================================================
# 2. Define Features and Target
# ============================================================

target = "failure_within_24h"

features = [
    "vibration_rms",
    "temperature_motor",
    "current_phase_avg",
    "pressure_level",
    "rpm",
    "operating_mode",
    "hours_since_maintenance",
    "ambient_temp",
    "machine_type"
]

X = df[features].copy()
y = df[target].copy()


# ============================================================
# 3. Handle Missing Values
# ============================================================

numeric_features = [
    "vibration_rms",
    "temperature_motor",
    "current_phase_avg",
    "pressure_level",
    "rpm",
    "hours_since_maintenance",
    "ambient_temp"
]

categorical_features = [
    "operating_mode",
    "machine_type"
]

for col in numeric_features:
    X[col] = X[col].fillna(X[col].median())

for col in categorical_features:
    X[col] = X[col].fillna(X[col].mode()[0])


# ============================================================
# 4. Train/Test Split
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining Samples:", len(X_train))
print("Testing Samples:", len(X_test))


# ============================================================
# 5. Preprocessing
# ============================================================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            StandardScaler(),
            numeric_features
        ),
        (
            "categorical",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            ),
            categorical_features
        )
    ]
)


# Fit preprocessing only on training data
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

print("\nProcessed Training Shape:", X_train_processed.shape)
print("Processed Testing Shape:", X_test_processed.shape)


# Convert to float32 for TensorFlow
X_train_processed = np.asarray(X_train_processed).astype("float32")
X_test_processed = np.asarray(X_test_processed).astype("float32")

y_train = np.asarray(y_train).astype("float32")
y_test = np.asarray(y_test).astype("float32")


# ============================================================
# 6. Build Deep Learning Neural Network
# ============================================================

model = Sequential([
    Dense(
        64,
        activation="relu",
        input_shape=(X_train_processed.shape[1],)
    ),

    Dropout(0.30),

    Dense(
        32,
        activation="relu"
    ),

    Dropout(0.20),

    Dense(
        16,
        activation="relu"
    ),

    Dense(
        1,
        activation="sigmoid"
    )
])


# ============================================================
# 7. Compile Model
# ============================================================

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)


model.summary()


# ============================================================
# 8. Early Stopping
# ============================================================

early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=8,
    restore_best_weights=True
)


# ============================================================
# 9. Train Model
# ============================================================

print("\nStarting Deep Learning Training...\n")

history = model.fit(
    X_train_processed,
    y_train,
    validation_split=0.20,
    epochs=50,
    batch_size=32,
    callbacks=[early_stopping],
    verbose=1
)


# ============================================================
# 10. Evaluate Model
# ============================================================

loss, accuracy = model.evaluate(
    X_test_processed,
    y_test,
    verbose=0
)

probabilities = model.predict(
    X_test_processed,
    verbose=0
).ravel()

predictions = (probabilities >= 0.5).astype(int)

roc_auc = roc_auc_score(
    y_test,
    probabilities
)

print("\n" + "=" * 60)
print("DEEP LEARNING RESULTS")
print("=" * 60)

print(f"\nTest Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy:.4f}")
print(f"ROC-AUC: {roc_auc:.4f}")

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        predictions,
        digits=4
    )
)

print("\nConfusion Matrix:")
print(
    confusion_matrix(
        y_test,
        predictions
    )
)


# ============================================================
# 11. Save Model
# ============================================================

MODEL_DIR = "deep_learning/model"

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)

model_path = os.path.join(
    MODEL_DIR,
    "factory_failure_mlp.keras"
)

preprocessor_path = os.path.join(
    MODEL_DIR,
    "preprocessor.pkl"
)

model.save(model_path)

joblib.dump(
    preprocessor,
    preprocessor_path
)


# ============================================================
# 12. Save Training History
# ============================================================

history_df = pd.DataFrame(
    history.history
)

history_path = os.path.join(
    MODEL_DIR,
    "training_history.csv"
)

history_df.to_csv(
    history_path,
    index=False
)


print("\n" + "=" * 60)
print("MODEL SAVED SUCCESSFULLY")
print("=" * 60)

print("\nModel:")
print(model_path)

print("\nPreprocessor:")
print(preprocessor_path)

print("\nTraining History:")
print(history_path)