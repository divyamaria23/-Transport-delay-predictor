# ============================================================
#  Public Transport Delays — EDA & ML Pipeline Starter
#  Dataset: https://www.kaggle.com/datasets/khushikyad001/
#           public-transport-delays-with-weather-and-events
# ============================================================
#
#  SETUP
#  -----
#  pip install pandas numpy matplotlib seaborn scikit-learn xgboost kaggle
#
#  Download dataset (after setting up Kaggle API key):
#  kaggle datasets download -d khushikyad001/public-transport-delays-with-weather-and-events
#  unzip public-transport-delays-with-weather-and-events.zip -d data/
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import warnings
warnings.filterwarnings("ignore")

plt.style.use("seaborn-v0_8-whitegrid")
sns.set_palette("muted")

# ─────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────

df = pd.read_csv("data/public_transport_delays.csv")   # adjust filename if needed

print("Shape:", df.shape)
print("\nColumns:", df.columns.tolist())
print("\nData types:\n", df.dtypes)
print("\nFirst 5 rows:\n", df.head())


# ─────────────────────────────────────────────
# 2. DATA CLEANING
# ─────────────────────────────────────────────

print("\n--- Missing values ---")
print(df.isnull().sum())

# Drop duplicates
df.drop_duplicates(inplace=True)

# Fill or drop nulls — adjust strategy per column
df.dropna(subset=["delay_minutes"], inplace=True)   # target must exist

# If a date column exists, parse it
for col in df.columns:
    if "date" in col.lower() or "time" in col.lower():
        try:
            df[col] = pd.to_datetime(df[col])
            print(f"Parsed {col} as datetime")
        except Exception:
            pass

print("\nShape after cleaning:", df.shape)


# ─────────────────────────────────────────────
# 3. EXPLORATORY DATA ANALYSIS
# ─────────────────────────────────────────────

target = "delay_minutes"

# 3a. Target distribution
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].hist(df[target], bins=40, edgecolor="white")
axes[0].set_title("Delay distribution")
axes[0].set_xlabel("Delay (minutes)")
axes[1].hist(np.log1p(df[target]), bins=40, edgecolor="white", color="steelblue")
axes[1].set_title("Log(Delay + 1) distribution")
axes[1].set_xlabel("log(delay + 1)")
plt.tight_layout()
plt.savefig("outputs/01_delay_distribution.png", dpi=150)
plt.show()

# 3b. Delay by weather condition
if "weather_condition" in df.columns:
    fig, ax = plt.subplots(figsize=(10, 4))
    order = df.groupby("weather_condition")[target].mean().sort_values(ascending=False).index
    sns.barplot(data=df, x="weather_condition", y=target, order=order, ax=ax)
    ax.set_title("Avg delay by weather condition")
    ax.set_xlabel("")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig("outputs/02_delay_by_weather.png", dpi=150)
    plt.show()

# 3c. Delay by event type
if "event_type" in df.columns:
    fig, ax = plt.subplots(figsize=(10, 4))
    order = df.groupby("event_type")[target].mean().sort_values(ascending=False).index
    sns.barplot(data=df, x="event_type", y=target, order=order, ax=ax)
    ax.set_title("Avg delay by event type")
    ax.set_xlabel("")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig("outputs/03_delay_by_event.png", dpi=150)
    plt.show()

# 3d. Temperature vs delay scatter
if "temperature" in df.columns:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.scatter(df["temperature"], df[target], alpha=0.3, s=10)
    ax.set_xlabel("Temperature (°C)")
    ax.set_ylabel("Delay (minutes)")
    ax.set_title("Temperature vs delay")
    plt.tight_layout()
    plt.savefig("outputs/04_temp_vs_delay.png", dpi=150)
    plt.show()

# 3e. Correlation heatmap (numeric features)
numeric_df = df.select_dtypes(include=[np.number])
fig, ax = plt.subplots(figsize=(10, 7))
sns.heatmap(numeric_df.corr(), annot=True, fmt=".2f", cmap="coolwarm",
            center=0, linewidths=0.5, ax=ax)
ax.set_title("Correlation heatmap")
plt.tight_layout()
plt.savefig("outputs/05_correlation_heatmap.png", dpi=150)
plt.show()


# ─────────────────────────────────────────────
# 4. FEATURE ENGINEERING
# ─────────────────────────────────────────────

# Time-based features (if datetime column found)
date_cols = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]
if date_cols:
    dc = date_cols[0]
    df["hour"]        = df[dc].dt.hour
    df["day_of_week"] = df[dc].dt.dayofweek   # 0=Mon, 6=Sun
    df["month"]       = df[dc].dt.month
    df["is_weekend"]  = (df["day_of_week"] >= 5).astype(int)
    df["is_rush_hour"]= df["hour"].isin([7, 8, 9, 17, 18, 19]).astype(int)
    print("Created time features: hour, day_of_week, month, is_weekend, is_rush_hour")

# Binary event flag
if "event_type" in df.columns:
    df["has_event"] = (df["event_type"].str.lower() != "none").astype(int)

# Extreme weather flag
if "weather_condition" in df.columns:
    severe = ["Snow", "Storm", "Heavy Rain"]
    df["severe_weather"] = df["weather_condition"].isin(severe).astype(int)

# Temperature buckets
if "temperature" in df.columns:
    df["temp_bucket"] = pd.cut(df["temperature"],
                               bins=[-np.inf, 0, 10, 20, np.inf],
                               labels=["freezing", "cold", "mild", "warm"])

print("\nFeature-engineered columns added:")
print([c for c in df.columns if c not in numeric_df.columns])


# ─────────────────────────────────────────────
# 5. ENCODE CATEGORICAL FEATURES
# ─────────────────────────────────────────────

cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
cat_cols = [c for c in cat_cols if c != target]

print("\nEncoding:", cat_cols)

le = LabelEncoder()
for col in cat_cols:
    df[col + "_enc"] = le.fit_transform(df[col].astype(str))

# Drop original categoricals and datetime
drop_cols = cat_cols + date_cols
df_model = df.drop(columns=drop_cols, errors="ignore")

print("Model dataframe shape:", df_model.shape)


# ─────────────────────────────────────────────
# 6. TRAIN / TEST SPLIT
# ─────────────────────────────────────────────

feature_cols = [c for c in df_model.columns if c != target]
X = df_model[feature_cols]
y = df_model[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

print(f"\nTrain size: {X_train.shape[0]} | Test size: {X_test.shape[0]}")


# ─────────────────────────────────────────────
# 7. MODEL TRAINING & EVALUATION
# ─────────────────────────────────────────────

models = {
    "Linear Regression":    LinearRegression(),
    "Random Forest":        RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
    "Gradient Boosting":    GradientBoostingRegressor(n_estimators=100, random_state=42),
    "XGBoost":              xgb.XGBRegressor(n_estimators=100, random_state=42,
                                             verbosity=0, n_jobs=-1),
}

results = {}

for name, model in models.items():
    # Use scaled data for linear, raw for tree models
    Xtr = X_train_scaled if name == "Linear Regression" else X_train
    Xte = X_test_scaled  if name == "Linear Regression" else X_test

    model.fit(Xtr, y_train)
    preds = model.predict(Xte)

    mae  = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2   = r2_score(y_test, preds)

    results[name] = {"MAE": round(mae, 3), "RMSE": round(rmse, 3), "R2": round(r2, 3)}
    print(f"{name:25s}  MAE={mae:.3f}  RMSE={rmse:.3f}  R²={r2:.3f}")

results_df = pd.DataFrame(results).T.sort_values("RMSE")
print("\n--- Model comparison ---\n", results_df)


# ─────────────────────────────────────────────
# 8. FEATURE IMPORTANCE (best tree model)
# ─────────────────────────────────────────────

best_name = results_df.index[0]
best_model = models[best_name]

if hasattr(best_model, "feature_importances_"):
    importances = pd.Series(best_model.feature_importances_, index=feature_cols)
    importances = importances.sort_values(ascending=True).tail(15)

    fig, ax = plt.subplots(figsize=(8, 5))
    importances.plot(kind="barh", ax=ax, color="steelblue")
    ax.set_title(f"Feature importance — {best_name}")
    ax.set_xlabel("Importance")
    plt.tight_layout()
    plt.savefig("outputs/06_feature_importance.png", dpi=150)
    plt.show()


# ─────────────────────────────────────────────
# 9. RESIDUAL ANALYSIS
# ─────────────────────────────────────────────

Xte_best = X_test_scaled if best_name == "Linear Regression" else X_test
preds_best = best_model.predict(Xte_best)
residuals  = y_test - preds_best

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].scatter(preds_best, residuals, alpha=0.3, s=10)
axes[0].axhline(0, color="red", linewidth=1)
axes[0].set_xlabel("Predicted")
axes[0].set_ylabel("Residual")
axes[0].set_title("Residuals vs predicted")

axes[1].hist(residuals, bins=40, edgecolor="white")
axes[1].set_title("Residual distribution")
axes[1].set_xlabel("Residual (minutes)")

plt.tight_layout()
plt.savefig("outputs/07_residuals.png", dpi=150)
plt.show()

print("\nDone! Outputs saved to outputs/ folder.")
print(f"Best model: {best_name}  (RMSE={results_df.loc[best_name,'RMSE']})")
