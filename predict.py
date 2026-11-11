
import pandas as pd
import numpy as np
import joblib

# File paths
MODEL_PATH = "C:/ddos_lab/ddos_ids_models.joblib"
CSV_PATH = "C:/ddos_lab/flows/live_flows.csv"
OUTPUT_PATH = "C:/ddos_lab/predictions.csv"

print("Loading trained models...")
bundle = joblib.load(MODEL_PATH)

dt_model = bundle["decision_tree"]
lr_model = bundle["logistic_regression"]
scaler = bundle["scaler"]

scale_cols = bundle["scale_cols"]
selected_features = bundle["selected_features"]

# CICFlowMeter names -> CIC-IDS2017 names
rename_columns = {
    "Tot Fwd Pkts": "Total Fwd Packets",
    "Tot Bwd Pkts": "Total Backward Packets",
    "TotLen Fwd Pkts": "Total Length of Fwd Packets",
    "TotLen Bwd Pkts": "Total Length of Bwd Packets",

    "Fwd Pkt Len Max": "Fwd Packet Length Max",
    "Fwd Pkt Len Min": "Fwd Packet Length Min",
    "Fwd Pkt Len Mean": "Fwd Packet Length Mean",
    "Fwd Pkt Len Std": "Fwd Packet Length Std",

    "Bwd Pkt Len Max": "Bwd Packet Length Max",
    "Bwd Pkt Len Min": "Bwd Packet Length Min",
    "Bwd Pkt Len Mean": "Bwd Packet Length Mean",
    "Bwd Pkt Len Std": "Bwd Packet Length Std",

    "Flow Byts/s": "Flow Bytes/s",
    "Flow Pkts/s": "Flow Packets/s",

    "Fwd IAT Tot": "Fwd IAT Total",
    "Bwd IAT Tot": "Bwd IAT Total",

    "Fwd Header Len": "Fwd Header Length",
    "Bwd Header Len": "Bwd Header Length",

    "Fwd Pkts/s": "Fwd Packets/s",
    "Bwd Pkts/s": "Bwd Packets/s",

    "Pkt Len Min": "Min Packet Length",
    "Pkt Len Max": "Max Packet Length",
    "Pkt Len Mean": "Packet Length Mean",
    "Pkt Len Std": "Packet Length Std",
    "Pkt Len Var": "Packet Length Variance",

    "FIN Flag Cnt": "FIN Flag Count",
    "SYN Flag Cnt": "SYN Flag Count",
    "RST Flag Cnt": "RST Flag Count",
    "PSH Flag Cnt": "PSH Flag Count",
    "ACK Flag Cnt": "ACK Flag Count",
    "URG Flag Cnt": "URG Flag Count",
    "ECE Flag Cnt": "ECE Flag Count",

    "Pkt Size Avg": "Average Packet Size",
    "Fwd Seg Size Avg": "Avg Fwd Segment Size",
    "Bwd Seg Size Avg": "Avg Bwd Segment Size",

    "Subflow Fwd Pkts": "Subflow Fwd Packets",
    "Subflow Fwd Byts": "Subflow Fwd Bytes",
    "Subflow Bwd Pkts": "Subflow Bwd Packets",
    "Subflow Bwd Byts": "Subflow Bwd Bytes",

    "Init Fwd Win Byts": "Init_Win_bytes_forward",
    "Init Bwd Win Byts": "Init_Win_bytes_backward",

    "Fwd Act Data Pkts": "act_data_pkt_fwd",
    "Fwd Seg Size Min": "min_seg_size_forward"
}

# Read captured network flows
print("\nReading captured network flows...")
df = pd.read_csv(CSV_PATH)

print("Number of extracted flows:", len(df))

# Rename CICFlowMeter columns
df.rename(columns=rename_columns, inplace=True)

# Check required scaling features
missing = [col for col in scale_cols if col not in df.columns]

if missing:
    print("\nMissing scaling features:")
    for col in missing:
        print("-", col)

    raise ValueError("Some required features are missing.")

# Extract numerical features
X = df[scale_cols].copy()

# Convert features to numeric
for col in scale_cols:
    X[col] = pd.to_numeric(X[col], errors="coerce")

# Replace infinity with NaN
X.replace([np.inf, -np.inf], np.nan, inplace=True)

# Display invalid values before replacement
print("\nInvalid values before replacement:")
print("NaN values:", X.isna().sum().sum())

# Replace all invalid values with zero
X.fillna(0, inplace=True)

print("\nInvalid values after replacement:")
print("NaN values:", X.isna().sum().sum())

# Keep all extracted flows
df_valid = df.copy()

print("\nTotal flows retained:", len(X))

# Apply the ORIGINAL training scaler
print("\nScaling features...")

X_scaled = scaler.transform(X)

X_scaled = pd.DataFrame(
    X_scaled,
    columns=scale_cols,
    index=X.index
)

# Select the same 20 features used during training
X_selected = X_scaled[selected_features]

print("\nMaking predictions...")

# Predict
dt_predictions = dt_model.predict(X_selected)
lr_predictions = lr_model.predict(X_selected)

# Convert encoded labels to readable labels
label_names = {
    0: "BENIGN",
    1: "DDoS"
}

# Build results
results = pd.DataFrame({
    "Flow ID": df_valid["Flow ID"].values,
    "Decision Tree": [
        label_names.get(int(x), str(x))
        for x in dt_predictions
    ],
    "Logistic Regression": [
        label_names.get(int(x), str(x))
        for x in lr_predictions
    ]
})

# Save predictions
results.to_csv(OUTPUT_PATH, index=False)

# Display results
print("\nPrediction results:")
print(results.to_string(index=False))

print("\nTotal flows processed:", len(results))

print("\nDecision Tree predictions:")
print(results["Decision Tree"].value_counts())

print("\nLogistic Regression predictions:")
print(results["Logistic Regression"].value_counts())

print("\nResults saved to:", OUTPUT_PATH)
