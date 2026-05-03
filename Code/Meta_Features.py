import numpy as np
import pandas as pd
import optuna
import json
import sys
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris, load_digits
from sklearn.ensemble import GradientBoostingRegressor

# -------------------------------
# DISABLE OPTUNA LOGS
# -------------------------------
optuna.logging.set_verbosity(optuna.logging.WARNING)

# -------------------------------
# FILE PATHS (ABSOLUTE)
# -------------------------------
OUTPUT_PATH = r"D:\Project\ML\Results\Output\output.txt"
RESULTS_PATH = r"D:\Project\ML\Results\Output\results.json"

# -------------------------------
# REDIRECT OUTPUT TO FILE
# -------------------------------
log_file = open(OUTPUT_PATH, "w")
sys.stdout = log_file

# -------------------------------
# META-FEATURE EXTRACTION
# -------------------------------
def extract_meta_features(X, y):
    num_samples = X.shape[0]
    num_features = X.shape[1]
    num_classes = len(np.unique(y))

    unique, counts = np.unique(y, return_counts=True)
    imbalance_ratio = counts.max() / counts.min()
    imbalance_ratio = np.log1p(imbalance_ratio)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    mean = np.mean(X_scaled)
    std = np.std(X_scaled)

    return [num_samples, num_features, num_classes, imbalance_ratio, mean, std]


# -------------------------------
# LOAD DATASETS
# -------------------------------
iris = load_iris()
X_iris, y_iris = iris.data, iris.target

digits = load_digits()
X_digits, y_digits = digits.data, digits.target

df = pd.read_csv(r"D:\Project\ML\Datasets\WineQT.csv")
X_wine = df.drop("quality", axis=1).values
y_wine = df["quality"].values

datasets = {
    "iris": (X_iris, y_iris),
    "digits": (X_digits, y_digits),
    "wineqt": (X_wine, y_wine)
}

# -------------------------------
# EXTRACT META-FEATURES
# -------------------------------
meta_X = []

for name, (X, y) in datasets.items():
    meta = extract_meta_features(X, y)
    meta_X.append(meta)
    print(f"{name} meta-features:", meta)

meta_X = np.array(meta_X)

# -------------------------------
# OPTUNA OBJECTIVE
# -------------------------------
def objective(trial, X, y):

    n_estimators = trial.suggest_int("n_estimators", 50, 150)
    max_depth = trial.suggest_int("max_depth", 2, 10)

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42,
        n_jobs=-1
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    score = cross_val_score(model, X, y, cv=cv, scoring="accuracy").mean()

    return score


# -------------------------------
# HYPERPARAMETER OPTIMIZATION
# -------------------------------
results = []

for name, (X, y) in datasets.items():
    print(f"\nOptimizing for {name}...")

    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: objective(trial, X, y), n_trials=20)

    best_params = study.best_params
    best_score = study.best_value

    print("Best params:", best_params)
    print("Best accuracy:", best_score)

    results.append({
        "dataset": name,
        "best_params": best_params,
        "accuracy": best_score
    })

# -------------------------------
# CREATE META-TARGETS
# -------------------------------
meta_y = []

for r in results:
    params = r["best_params"]
    meta_y.append([params["n_estimators"], params["max_depth"]])

meta_y = np.array(meta_y)

print("\nMeta-features (X):")
print(meta_X)

print("\nMeta-targets (y):")
print(meta_y)

# -------------------------------
# META-LEARNING MODEL
# -------------------------------
scaler_meta = StandardScaler()
meta_X_scaled = scaler_meta.fit_transform(meta_X)

y_estimators = meta_y[:, 0]
y_depth = meta_y[:, 1]

model_estimators = GradientBoostingRegressor()
model_depth = GradientBoostingRegressor()

model_estimators.fit(meta_X_scaled, y_estimators)
model_depth.fit(meta_X_scaled, y_depth)

# -------------------------------
# TEST META-LEARNING
# -------------------------------
print("\n--- Testing Meta-Learning ---")

meta_test = extract_meta_features(X_iris, y_iris)
meta_test_scaled = scaler_meta.transform([meta_test])

pred_estimators = int(model_estimators.predict(meta_test_scaled)[0])
pred_depth = int(model_depth.predict(meta_test_scaled)[0])

print("\nPredicted hyperparameters:")
print("n_estimators:", pred_estimators)
print("max_depth:", pred_depth)

model = RandomForestClassifier(
    n_estimators=pred_estimators,
    max_depth=pred_depth,
    random_state=42
)

score = cross_val_score(model, X_iris, y_iris, cv=3).mean()

print("\nAccuracy using meta-learning:", score)

# -------------------------------
# SAVE RESULTS
# -------------------------------
final_results = {
    "iris": results[0],
    "digits": results[1],
    "wineqt": results[2],
    "meta_learning": {
        "n_estimators": pred_estimators,
        "max_depth": pred_depth,
        "accuracy": score
    }
}

with open(RESULTS_PATH, "w") as f:
    json.dump(final_results, f, indent=4)

print("\nResults saved to results.json")

# -------------------------------
# RESTORE TERMINAL OUTPUT
# -------------------------------
sys.stdout = sys.__stdout__
log_file.close()

print("Output complete.")
print("Check these files:")
print(OUTPUT_PATH)
print(RESULTS_PATH)