
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)
import joblib
import os

import pandas as pd

# ==========================
# Load Dataset
# ==========================
df = pd.read_csv("dataset/diabetes_prediction_dataset.csv")

print("Original Shape:", df.shape)

# ==========================
# Remove Duplicate Rows
# ==========================
df = df.drop_duplicates()

print("Shape After Removing Duplicates:", df.shape)

# ==========================
# One-Hot Encode Categorical Features
# ==========================
df = pd.get_dummies(
    df,
    columns=["gender", "smoking_history"],
    drop_first=True,
    dtype=int
)

# ==========================
# Features and Target
# ==========================
X = df.drop("diabetes", axis=1)
y = df["diabetes"]

# ==========================
# Train-Test Split
# ==========================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# ==========================
# Create Models Folder
# ==========================
os.makedirs("models", exist_ok=True)


# ==========================
# Create Base Random Forest Model
# ==========================
classifier = RandomForestClassifier(
    random_state=42
)

# ==========================
# Hyperparameter Search Space
# ==========================
param_grid = {
    "n_estimators": [100, 200, 300],
    "max_depth": [10, 20, 30, None],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "max_features": ["sqrt", "log2"]
}

# ==========================
# Randomized Search
# ==========================
random_search = RandomizedSearchCV(
    estimator=classifier,
    param_distributions=param_grid,
    n_iter=20,
    cv=5,
    scoring="f1",
    random_state=42,
    n_jobs=-1,
    verbose=2
)

print("\nStarting Hyperparameter Tuning...")

random_search.fit(X_train, y_train)

print("\n✅ Hyperparameter Tuning Completed!")

# ==========================
# Best Parameters
# ==========================
print("\n========== BEST PARAMETERS ==========")
print(random_search.best_params_)

print("\nBest Cross Validation F1 Score:")
print(random_search.best_score_)

# ==========================
# Best Model
# ==========================
classifier = random_search.best_estimator_

# ==========================
# Make Predictions
# ==========================
y_pred = classifier.predict(X_test)

print("\nFirst 20 Predictions:")
print(y_pred[:20])

print("\nFirst 20 Actual Values:")
print(y_test.values[:20])

# ==========================
# Model Evaluation
# ==========================
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\n========== MODEL EVALUATION ==========")

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# ==========================
# Dataset Information
# ==========================
print("\n========== DATA SPLIT ==========")

print("Training Features Shape :", X_train.shape)
print("Testing Features Shape  :", X_test.shape)

print("Training Target Shape   :", y_train.shape)
print("Testing Target Shape    :", y_test.shape)

print("\nTraining Class Distribution:")
print(y_train.value_counts(normalize=True) * 100)

print("\nTesting Class Distribution:")
print(y_test.value_counts(normalize=True) * 100)

# ==========================
# Save Model
# ==========================
joblib.dump(classifier, "models/classification_model.pkl")

print("\n Classification model saved successfully!")

# ==========================
# Save Feature Names
# ==========================
joblib.dump(X.columns.tolist(), "models/feature_names.pkl")

print("✅ Feature names saved successfully!")