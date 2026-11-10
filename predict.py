
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

# =========================================
# 1. FILE PATHS
# =========================================

BASE_DIR = Path(r"C:\ddos_lab")

MODEL_PATH = BASE_DIR / "ddos_ids_models.joblib"
CSV_PATH = BASE_DIR / "flows" / "live_flows.csv"


# =========================================
# 2. LOAD TRAINED MODELS
# =========================================

print("Loading trained models...")

saved = joblib.load(MODEL_PATH)

dt_model = saved["decision_tree"]
lr_model = saved["logistic_regression"]

scaler = saved["scaler"]
scale_cols = list(saved["scale_cols"])

selected_features = list(saved["selected_features"])
feature_columns = list(saved["feature_columns"])


# =========================================
# 3. LOAD EXTRACTED NETWORK FLOWS
# =========================================

print("Reading captured network flows...")

if not CSV_PATH.exists():
    raise FileNotFoundError(
        f"Flow CSV not found: {CSV_PATH}"
    )

flows = pd.read_csv(CSV_PATH)

print(f"Number of extracted flows: {len(flows)}")


if flows.empty:
    print("No network flows available.")
    raise SystemExit


# =========================================
# 4. VALIDATE REQUIRED FEATURES
# =========================================

missing_columns = [
    col for col in feature_columns
    if col not in flows.columns
]

if missing_columns:
    raise ValueError(
        "CICFlowMeter output is missing these "
        f"training features:\n{missing_columns}"
    )

# Keep exactly the features used during training
X = flows[feature_columns].copy()

# Convert values to numeric
X = X.apply(pd.to_numeric, errors="coerce")

# Replace infinity with NaN
X = X.replace([np.inf, -np.inf], np.nan)

# Identify invalid flows
valid_mask = X.notna().all(axis=1)

X_valid = X.loc[valid_mask].copy()

print(f"Valid flows: {len(X_valid)}")
print(f"Invalid flows: {(~valid_mask).sum()}")

if X_valid.empty:
    print("No valid flows to classify.")
    raise SystemExit


# =========================================
# 5. APPLY THE TRAINING SCALER
# =========================================

X_scaled = X_valid.copy()

X_scaled[scale_cols] = scaler.transform(
    X_valid[scale_cols]
)


# =========================================
# 6. SELECT THE SAME 20 FEATURES
# =========================================

X_selected = X_scaled[selected_features]


# =========================================
# 7. PREDICT WITH BOTH MODELS
# =========================================

dt_predictions = dt_model.predict(X_selected)

lr_predictions = lr_model.predict(X_selected)

labels = {
    0: "BENIGN",
    1: "DDoS"
}


# =========================================
# 8. DISPLAY RESULTS
# =========================================

print("\n========== TRAFFIC CLASSIFICATION ==========\n")

results = []

for i, (dt_pred, lr_pred) in enumerate(
    zip(dt_predictions, lr_predictions)
):

    dt_label = labels[int(dt_pred)]
    lr_label = labels[int(lr_pred)]

    results.append({
        "Flow": int(X_valid.index[i]),
        "Decision Tree": dt_label,
        "Logistic Regression": lr_label
    })

    print(f"Flow {X_valid.index[i]}")

    print(f"Decision Tree: {dt_label}")
    print(f"Logistic Regression: {lr_label}")

    if dt_label == lr_label:
        print(f"Agreement: {dt_label}")
    else:
        print("Agreement: Models disagree")

    print("--------------------------------------------")


# =========================================
# 9. SAVE PREDICTIONS
# =========================================

results_df = pd.DataFrame(results)

OUTPUT_PATH = BASE_DIR / "predictions.csv"

results_df.to_csv(OUTPUT_PATH, index=False)

print("\n========== SUMMARY ==========")

print("\nDecision Tree:")
print(results_df["Decision Tree"].value_counts())

print("\nLogistic Regression:")
print(results_df["Logistic Regression"].value_counts())

print(f"\nPredictions saved to: {OUTPUT_PATH}")
