# 🚌 Public Transport Delay Predictor

Predicting public transport delays using weather conditions and city events — end-to-end ML pipeline with exploratory data analysis, feature engineering, and model comparison.

---

## Overview

Delays in public transport are heavily influenced by external factors like rain, snow, and large city events. This project builds a regression model to **predict delay duration (in minutes)** based on these factors, helping transit authorities and commuters anticipate disruptions.

**Dataset:** [Public Transport Delays with Weather & Events](https://www.kaggle.com/datasets/khushikyad001/public-transport-delays-with-weather-and-events) — Kaggle

---

## Key Results

| Model | MAE | RMSE | R² |
|---|---|---|---|
| Linear Regression | 0.239 | 0.273 | 0.617 |
| Random Forest | 0.000 | 0.000 | 1.000 |
| Gradient Boosting | 0.000 | 0.000 | 1.000 |
| XGBoost | 0.000 | 0.000 | 1.000 |

> Tree-based models achieve perfect scores, indicating the `delayed` 
> column is deterministically derived from other features in the dataset. 
> Linear Regression (R²=0.617) gives a more realistic generalisation baseline.

**Best honest model:** Linear Regression — R²=0.617
---

## Project Structure

```
transport-delay-predictor/
├── data/                        # Raw dataset (not tracked — see below)
├── outputs/                     # EDA plots and model outputs
│   ├── 01_delay_distribution.png
│   ├── 02_delay_by_weather.png
│   ├── 03_delay_by_event.png
│   ├── 04_temp_vs_delay.png
│   ├── 05_correlation_heatmap.png
│   ├── 06_feature_importance.png
│   └── 07_residuals.png
├── transport_delays_eda.py      # Full pipeline: EDA → features → models
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Features Engineered

| Feature | Description |
|---|---|
| `hour` | Hour of the day extracted from timestamp |
| `day_of_week` | 0 = Monday, 6 = Sunday |
| `is_weekend` | Binary flag for Saturday/Sunday |
| `is_rush_hour` | Binary flag for 7–9 AM and 5–7 PM |
| `month` | Month extracted for seasonal patterns |
| `has_event` | Binary — whether a city event was occurring |
| `severe_weather` | Binary flag for Snow, Storm, Heavy Rain |
| `temp_bucket` | Temperature bucketed: freezing / cold / mild / warm |

---

## EDA Highlights

- **Snow and storms** cause the highest average delays (~2–3x clear weather)
- **City events** (concerts, sports matches) increase delays by ~40–60%
- **Rush hour + rain** combinations produce the worst delay spikes
- Delays peak in **winter months** (Dec–Feb) across all transport types
- **Bus** is the most delay-prone; **Metro** is the most resilient

---

## Setup & Usage

### 1. Clone the repo
```bash
git clone https://github.com/YOUR_USERNAME/transport-delay-predictor.git
cd transport-delay-predictor
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Download the dataset
Set up your [Kaggle API key](https://www.kaggle.com/docs/api), then:
```bash
kaggle datasets download -d khushikyad001/public-transport-delays-with-weather-and-events
unzip public-transport-delays-with-weather-and-events.zip -d data/
mkdir outputs
```

### 4. Run the pipeline
```bash
python transport_delays_eda.py
```

Plots are saved to `outputs/`. Model metrics are printed to the console.

---

## Tech Stack

- **Python** — pandas, numpy
- **Visualization** — matplotlib, seaborn
- **Modelling** — scikit-learn, XGBoost
- **Skills practised** — data cleaning, feature engineering, time-based analysis, regression modelling, residual analysis

---

## .gitignore

Add a `.gitignore` with:
```
data/
__pycache__/
*.zip
*.pyc
.env
```

---

## Author

**Divya** — B.Tech Information Technology  
[GitHub](https://github.com/YOUR_USERNAME) · [LinkedIn](https://linkedin.com/in/YOUR_PROFILE)
