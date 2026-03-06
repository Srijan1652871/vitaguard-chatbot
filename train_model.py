import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib

# ── Load Dataset ─────────────────────────────────────────────────────────────
df = pd.read_csv("diabetes_012_health_indicators_BRFSS2015.csv")

print(f"Dataset loaded: {len(df)} rows, {len(df.columns)} columns\n")


# ── Shared Training Function ──────────────────────────────────────────────────
def train_and_save(df, target_col, features, model_path, scaler_path, label):
    print(f"{'='*60}")
    print(f"Training: {label}")
    print(f"Target  : {target_col}")
    print(f"Features: {features}")
    print(f"{'='*60}")

    X = df[features]
    y = df[target_col]

    # Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    # Train
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\nAccuracy : {acc:.4f} ({acc*100:.2f}%)")
    print(f"\nClassification Report:\n{classification_report(y_test, y_pred)}")

    # Save
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    print(f"Saved → {model_path}, {scaler_path}\n")


# ── 1. Diabetes ───────────────────────────────────────────────────────────────
# Target: 0 = No diabetes, 1 = Prediabetes, 2 = Diabetes
# Uses all 16 original features (existing behaviour preserved)
train_and_save(
    df,
    target_col="Diabetes_012",
    features=[
        "HighBP", "HighChol", "BMI", "Smoker", "PhysActivity",
        "Fruits", "Veggies", "HvyAlcoholConsump", "GenHlth",
        "MentHlth", "PhysHlth", "DiffWalk", "Sex", "Age",
        "Education", "Income"
    ],
    model_path="model.pkl",
    scaler_path="scaler.pkl",
    label="Diabetes"
)


# ── 2. Heart Disease ──────────────────────────────────────────────────────────
# Target: 0 = No heart disease/attack, 1 = Heart disease/attack
# Dropped: Education, Income (weak predictors)
train_and_save(
    df,
    target_col="HeartDiseaseorAttack",
    features=[
        "HighBP", "HighChol", "BMI", "Smoker", "PhysActivity",
        "Fruits", "Veggies", "HvyAlcoholConsump", "GenHlth",
        "PhysHlth", "DiffWalk", "Sex", "Age"
    ],
    model_path="heart_model.pkl",
    scaler_path="heart_scaler.pkl",
    label="Heart Disease"
)


# ── 3. Hypertension (High BP) ─────────────────────────────────────────────────
# Target: 0 = No high BP, 1 = High BP
# Dropped: HighBP (this IS the target), Education, Income
train_and_save(
    df,
    target_col="HighBP",
    features=[
        "HighChol", "BMI", "Smoker", "PhysActivity",
        "Fruits", "Veggies", "HvyAlcoholConsump", "GenHlth",
        "MentHlth", "PhysHlth", "DiffWalk", "Sex", "Age"
    ],
    model_path="bp_model.pkl",
    scaler_path="bp_scaler.pkl",
    label="Hypertension (High BP)"
)


# ── 4. Stroke ─────────────────────────────────────────────────────────────────
# Target: 0 = No stroke, 1 = Stroke
# Dropped: DiffWalk (data leakage risk — often a post-stroke symptom)
# Dropped: Education, Income, MentHlth (weak predictors)
train_and_save(
    df,
    target_col="Stroke",
    features=[
        "HighBP", "HighChol", "BMI", "Smoker", "PhysActivity",
        "Fruits", "Veggies", "HvyAlcoholConsump", "GenHlth",
        "PhysHlth", "Sex", "Age"
    ],
    model_path="stroke_model.pkl",
    scaler_path="stroke_scaler.pkl",
    label="Stroke"
)


# ── 5. Obesity ────────────────────────────────────────────────────────────────
# Target: derived from BMI > 30 (no direct obesity column in dataset)
# Dropped: BMI (used to create the target — would cause data leakage)
# Dropped: Education, Income, Smoker (weak/paradoxical predictors)
df["Obese"] = (df["BMI"] > 30).astype(int)
train_and_save(
    df,
    target_col="Obese",
    features=[
        "HighBP", "HighChol", "PhysActivity", "Fruits", "Veggies",
        "HvyAlcoholConsump", "GenHlth", "MentHlth", "PhysHlth",
        "DiffWalk", "Sex", "Age"
    ],
    model_path="obesity_model.pkl",
    scaler_path="obesity_scaler.pkl",
    label="Obesity"
)


# ── Summary ───────────────────────────────────────────────────────────────────
print("=" * 60)
print("All 5 models trained and saved successfully.")
print()
print("Files created:")
print("  Diabetes    → diab_model.pkl,         diab_scaler.pkl")
print("  Heart       → heart_model.pkl,   heart_scaler.pkl")
print("  Hypertension→ bp_model.pkl,      bp_scaler.pkl")
print("  Stroke      → stroke_model.pkl,  stroke_scaler.pkl")
print("  Obesity     → obesity_model.pkl, obesity_scaler.pkl")
print("=" * 60)
