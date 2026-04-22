# Uniqlo Stock Price Prediction

A beginner machine learning project that predicts the daily **Close price** of Fast Retailing (Uniqlo) stock using three supervised regression algorithms.

---

## Project Overview

This project walks through a complete ML pipeline — from raw stock data to evaluated predictions — using three regression algorithms. Every step is heavily commented and explained with plain-language analogies, making it a solid learning resource for anyone new to data science or financial ML.

**Goal:** Predict the `Close` price (end-of-day stock price in Japanese Yen) using features derived from daily trading data.

---

## Dataset

| File | Period | Rows | Purpose |
|------|--------|------|---------|
| `Uniqlo(FastRetailing) 2012-2016 Training - stocks2012-2016.csv` | Jan 2012 – Dec 2016 | 1,226 | Model training |
| `Uniqlo(FastRetailing) 2017 Test - stocks2017.csv` | Jan 4–13, 2017 | 7 | Model evaluation |

### Columns

| Column | Description |
|--------|-------------|
| `Date` | Trading day |
| `Open` | Opening price (Yen) |
| `High` | Highest price that day (Yen) |
| `Low` | Lowest price that day (Yen) |
| `Close` | Closing price (Yen) — **target variable** |
| `Volume` | Number of shares traded |
| `Stock Trading` | Total value traded (Volume × Price) |

No missing values were found in either dataset.

---

## Notebook: `Uniqlo_Stock.ipynb`

The notebook is structured across 5 sections:

### 1. Install and Import Tools
Imports `pandas`, `numpy`, `matplotlib`, `seaborn`, and `scikit-learn`. Loads and parses both CSV files, sorting by date so the oldest records come first.

### 2. Exploratory Data Analysis (EDA)
- **Close price over time** with a 20-day moving average — the stock rose from ~14,000 Yen (2012) to ~60,000 (2015) then fell back
- **Price distribution histogram** and daily volume bar chart
- **Correlation heatmap** — Open, High, Low, and Close are ~0.99 correlated; Volume is more independent

### 3. Preprocessing and Feature Engineering

Eight new features are constructed from the raw columns:

| Feature | Formula | Purpose |
|---------|---------|---------|
| `Price_Range` | High − Low | Intraday price movement |
| `Open_Close` | Close − Open | Daily direction |
| `Avg_Price` | (High + Low) / 2 | Midpoint price |
| `Volatility` | Price_Range / Open | Relative movement |
| `MA5` | 5-day rolling close mean | Short-term trend |
| `MA20` | 20-day rolling close mean | Long-term trend |
| `DayOfWeek` | 0 (Mon) – 4 (Fri) | Day-of-week patterns |
| `Month` | 1–12 | Seasonal patterns |

**Scaling:** All 12 features are scaled to [0, 1] using `MinMaxScaler` — fitted on training data only to prevent data leakage.

### 4. Model Training

Three models are tuned with `GridSearchCV` (5-fold cross-validation):

| Model | What it does | Key hyperparameters tuned |
|-------|-------------|--------------------------|
| **Ridge Regression** | Linear model with regularisation | `alpha` |
| **Random Forest** | Ensemble of decision trees | `n_estimators`, `max_depth`, `min_samples_split` |
| **Gradient Boosting** | Sequential error-correcting trees | `n_estimators`, `learning_rate`, `max_depth` |

### 5. Evaluation and Visualisations

Performance is measured with four metrics: RMSE, MAE, R², and MAPE. Charts produced include:
- Predicted vs actual price lines for each model
- Bar chart comparing all metrics across models
- Per-day absolute error bars
- Actual vs predicted scatter plot
- Feature importance rankings (Random Forest and Gradient Boosting)

---

## How to Run

1. **Clone or download** this repository
2. **Install dependencies:**
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn
   ```
3. **Open the notebook:**
   ```bash
   jupyter notebook Uniqlo_Stock.ipynb
   ```
4. **Update the file paths** at the top of Section 1 to point to your local CSV files (or place the CSVs in the same directory as the notebook)
5. Run all cells top to bottom — each section builds on the previous one

> The notebook was originally built for Google Colab. If running there, upload the CSV files to your Drive and update the paths accordingly.

---

## Key Concepts Covered

- Exploratory Data Analysis (EDA)
- Feature engineering from time series data
- MinMax scaling / normalisation
- Train/test split with temporal ordering
- Hyperparameter tuning with Grid Search
- 5-fold cross-validation
- Regression metrics: RMSE, MAE, R², MAPE
- Feature importance from tree-based models
- Data leakage prevention

---

## Limitations

- The test set covers only **7 trading days**, so metric scores should be interpreted cautiously
- The models use same-day features (`Open`, `High`, `Low`) which are only known after the market opens — this project is suited for **end-of-day prediction**, not intraday trading
- Past stock performance does not guarantee future results

---

## Data Source

Historical stock data for Fast Retailing Co., Ltd. (Uniqlo parent company), traded on the Tokyo Stock Exchange (ticker: 9983).
