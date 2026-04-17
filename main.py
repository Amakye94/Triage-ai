import os
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib

# ----------------------------
# 1. LOAD SUBJECT DATA
# ----------------------------
def load_subject(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    with open(file_path, "rb") as f:
        data = pickle.load(f, encoding="latin1")

    wrist = data['signal']['wrist']

    eda = wrist['EDA']
    bvp = wrist['BVP']
    labels = data['label']

    min_len = min(len(eda), len(bvp), len(labels))

    eda = eda[:min_len].flatten()
    bvp = bvp[:min_len].flatten()
    labels = labels[:min_len].flatten()

    return pd.DataFrame({
        "eda": eda,
        "bvp": bvp,
        "label": labels
    })


# ----------------------------
# 2. LOAD DATA
# ----------------------------
print("Loading datasets...")

df_s2 = load_subject("S2/S2.pkl")
df_s3 = load_subject("S3/S3.pkl")
df_s4 = load_subject("S4/S4.pkl")

df = pd.concat([df_s2, df_s3, df_s4], ignore_index=True)

print("Combined shape:", df.shape)


# ----------------------------
# 3. MAP LABELS → TRIAGE
# ----------------------------
def map_label(x):
    if x == 1:
        return 0
    elif x == 2:
        return 1
    else:
        return 0

df["triage"] = df["label"].apply(map_label)
df = df.dropna()

print("\nTriage distribution:")
print(df["triage"].value_counts())


# ----------------------------
# 4. FEATURE ENGINEERING (ADVANCED)
# ----------------------------
window_size = 50
step_size = 10   # 🔥 sliding window

features = []

for i in range(0, len(df) - window_size, step_size):
    window = df.iloc[i:i+window_size]

    # EDA features
    eda_mean = window["eda"].mean()
    eda_std = window["eda"].std()
    eda_max = window["eda"].max()
    eda_min = window["eda"].min()

    # BVP features
    bvp_mean = window["bvp"].mean()
    bvp_std = window["bvp"].std()
    bvp_max = window["bvp"].max()
    bvp_min = window["bvp"].min()

    label = window["triage"].mode()[0]

    features.append([
        eda_mean, eda_std, eda_max, eda_min,
        bvp_mean, bvp_std, bvp_max, bvp_min,
        label
    ])

feature_df = pd.DataFrame(features, columns=[
    "eda_mean", "eda_std", "eda_max", "eda_min",
    "bvp_mean", "bvp_std", "bvp_max", "bvp_min",
    "label"
])

print("\nFeature dataset shape:", feature_df.shape)


# ----------------------------
# 5. TRAIN MODEL
# ----------------------------
print("\nTraining model...")

X = feature_df.drop("label", axis=1)
y = feature_df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(
    n_estimators=200,     # stronger model
    max_depth=10,
    random_state=42
)

model.fit(X_train, y_train)

preds = model.predict(X_test)

print("\nModel Performance:")
print(classification_report(y_test, preds))


# ----------------------------
# 6. SAVE MODEL
# ----------------------------
joblib.dump(model, "triage_model1.pkl")

print("\n✅ Model saved as triage_model.pkl")