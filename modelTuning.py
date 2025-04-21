import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import numpy as np
import joblib

df = pd.read_json("applicants.json", orient="records")

X = df[["experienceYears", "technicalScore"]]
y = df["hiring"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

joblib.dump(scaler, "scaler.pkl")

paramGrid = [
    {"C": [0.1, 1, 10, 100, 1000], "kernel": ["linear"]},
    {"C": [0.1, 1, 10, 100, 1000], "gamma": [1, 0.1, 0.01, 0.001, 0.0001], "kernel": ["rbf"]},
    {"C": [0.1, 1, 10], "degree": [2, 3, 4], "gamma": ["scale", "auto"], "kernel": ["poly"]}
]

svc = SVC(probability=True, random_state=42)

gridSearch = GridSearchCV(estimator=svc, param_grid=paramGrid, cv=5, n_jobs=-1, verbose=2)
gridSearch.fit(X_train_scaled, y_train)
bestModel = gridSearch.best_estimator_
print("En iyi model parametreleri:\n", gridSearch.best_params_)
print(f"En iyi model skoru: {gridSearch.best_score_:.4f}")

y_pred = bestModel.predict(X_test_scaled)

accuracy = accuracy_score(y_test, y_pred)
confMatrix = confusion_matrix(y_test, y_pred)
classReport = classification_report(y_test, y_pred, target_names=["Kabul (0)", "Red (1)"])

print(f"Accuracy: {accuracy:.4f}")
print("Confusion Matrix:")
print(confMatrix)
print("Classification Report:")
print(classReport)

joblib.dump(bestModel, "bestModel.pkl")