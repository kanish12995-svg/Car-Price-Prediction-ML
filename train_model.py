
import pandas as pd
import joblib

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error

BASE_DIR = Path(__file__).resolve().parent

# 1. Load dataset
df = pd.read_csv(BASE_DIR / "car_data.csv")

print("Dataset loaded successfully!")
print(df.head())

# 2. Remove identifier
df = df.drop(columns=["Car_ID"], errors="ignore")

# 3. Clean dataset
df["Car_Price"] = pd.to_numeric(
    df["Car_Price"], errors="coerce"
)

df["Model"] = pd.to_numeric(
    df["Model"], errors="coerce"
)

df = df.dropna()

df = df[df["Car_Price"] > 0]

# 4. Select features and target
features = [
    "Company_Name",
    "Brand",
    "Model",
    "Condition"
]

X = df[features]
y = df["Car_Price"]

# 5. Identify categorical columns
categorical_columns = [
    "Company_Name",
    "Brand",
    "Condition"
]

# 6. Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        (
            "encoder",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_columns
        )
    ],
    remainder="passthrough"
)

# 7. Random Forest model
model = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(
        n_estimators=200,
        max_depth=15,
        random_state=42
    ))
])

# 8. Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

# 9. Train model
print("Training Random Forest model...")

model.fit(X_train, y_train)

# 10. Prediction
y_pred = model.predict(X_test)

# 11. Evaluation
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print("\nModel Training Successful!")
print("R2 Score:", round(r2, 4))
print("R2 Percentage:", round(r2 * 100, 2), "%")
print("MAE:", round(mae, 2))

# 12. Save model
joblib.dump(
    {
        "model": model,
        "r2": r2,
        "mae": mae,
        "data": df,
        "features": features
    },
    BASE_DIR / "car_model.pkl"
)

print("\ncar_model.pkl created successfully!")


