# train_model.py
# ─────────────────────────────────────────────────────────────────────────────
# # Author: Neelabho Chakraborty & Aakriti Thakur 
# PURPOSE: Train the Boston Housing ML model and save it to disk.
# RUN THIS: Once only, from the terminal: python train_model.py
# OUTPUT: Creates model.pkl and scaler.pkl in the same folder as this script
# ─────────────────────────────────────────────────────────────────────────────

import numpy as np
import pandas as pd
import joblib
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Models
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

# Loading the dataset
from sklearn.datasets import fetch_openml

print("Loading dataset...")
boston = fetch_openml(name='boston', version=1, as_frame=True)
df = boston.frame  

# The target column in OpenML version is called 'MEDV'
X = df.drop(columns=['MEDV'])  # features (13 columns)
y = df['MEDV'].astype(float)   # target: median house value in $1000s

print(f"Dataset loaded. Shape: {X.shape}")
print(f"Features: {list(X.columns)}")
print(f"Target range: ${y.min():.1f}k – ${y.max():.1f}k")
print(f"Target mean: ${y.mean():.2f}k")

# Train/test split
# test_size=0.2   → 20% of data held back for evaluation
# random_state=42 → fixed seed so the split is the same every time 

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\nTraining set: {X_train.shape[0]} rows")
print(f"Test set:     {X_test.shape[0]} rows")

# Feature scaling
# StandardScaler transforms each feature to mean=0, std=1.
# CRITICAL: We fit the scaler on TRAINING data only.
# We then use that same fitted scaler to transform BOTH train and test data.
# Fitting on test data would cause data leakage (test data influencing training).

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)   # fit AND transform training data
X_test_scaled  = scaler.transform(X_test)         # ONLY transform test data (no re-fitting)


# Train all three models
print("\n" + "="*50)
print("TRAINING MODELS")
print("="*50)

models = {
    "Linear Regression":    LinearRegression(),
    "Random Forest":        RandomForestRegressor(n_estimators=100, random_state=42),
    "Gradient Boosting":    GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=4, random_state=42)
}

results = {}

for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train_scaled, y_train)
    predictions = model.predict(X_test_scaled)

    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2   = r2_score(y_test, predictions)

    results[name] = {"model": model, "rmse": rmse, "r2": r2}

    print(f"  RMSE:  {rmse:.4f}  (average error in $1000s)")
    print(f"  R²:    {r2:.4f}  (1.0 = perfect, 0.0 = useless)")

# Pick the best model (highest R²)
best_name = max(results, key=lambda n: results[n]["r2"])
best_model = results[best_name]["model"]

print("\n" + "="*50)
print(f"BEST MODEL: {best_name}")
print(f"  RMSE: {results[best_name]['rmse']:.4f}")
print(f"  R²:   {results[best_name]['r2']:.4f}")
print("="*50)

# Save model and scaler to disk
# These files will be loaded by app.py every time a prediction is requested.

joblib.dump(best_model, "model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("\nSaved model.pkl  ← the trained Gradient Boosting model")
print("Saved scaler.pkl ← the fitted StandardScaler")
print("\nDone! You can now run: python app.py")
