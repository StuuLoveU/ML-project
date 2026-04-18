# Uniqlo Stock Price Prediction

A beginner machine learning project that predicts the daily **Close price** of Fast Retailing (Uniqlo) stock using three supervised regression algorithms.

---

## Project Overview

| Item | Detail |
|------|--------|
| Dataset | Fast Retailing (Tokyo Stock Exchange) |
| Training data | 2012 - 2016 (1,226 trading days) |
| Test data | January 2017 (7 trading days) |
| Target variable | Close price (Japanese Yen) |
| Task type | Regression |
| Algorithms | Ridge Regression, Random Forest, Gradient Boosting |

---

## Files

```
Uniqlo_Stock_Beginner_v3.ipynb          Main Colab notebook (beginner-friendly)
Uniqlo_Stock_Presentation.pptx          Slide presentation (7 slides)
README.md                               This file
```

**Data files (upload to Google Drive before running the notebook):**
```
Uniqlo(FastRetailing) 2012-2016 Training - stocks2012-2016.csv
Uniqlo(FastRetailing) 2017 Test - stocks2017.csv
```

---

## How to Run

1. Upload both CSV files to your Google Drive
2. Open `Uniqlo_Stock_Beginner_v3.ipynb` in [Google Colab](https://colab.research.google.com)
3. Run **Cell 1** to install libraries
4. Run **Cell 2** to import libraries
5. Run **Cell 3** to mount Google Drive and allow access
6. Update the file paths in **Cell 4** to match where your CSVs are saved:
   ```python
   train = pd.read_csv('/content/drive/MyDrive/your_folder/Uniqlo(FastRetailing) 2012-2016 Training - stocks2012-2016.csv')
   test  = pd.read_csv('/content/drive/MyDrive/your_folder/Uniqlo(FastRetailing) 2017 Test - stocks2017.csv')
   ```
7. Run all remaining cells top to bottom

---

## Dataset Columns

| Column | Description |
|--------|-------------|
| Date | Trading day |
| Open | Price at market open (morning) |
| High | Highest price of the day |
| Low | Lowest price of the day |
| Close | Price at market close (end of day) — **this is the target** |
| Volume | Number of shares traded |
| Stock Trading | Total money traded (Volume x Price) |

---

## Methodology

### Preprocessing
- Dates parsed and sorted chronologically
- No missing values found in either dataset
- MinMax Scaling applied (fit on training data only, then applied to test)

### Feature Engineering
Eight new features created from the original columns:

| Feature | Formula | Purpose |
|---------|---------|---------|
| Price_Range | High - Low | Intraday movement |
| Open_Close | Close - Open | Daily direction |
| Avg_Price | (High + Low) / 2 | Midpoint price |
| Volatility | Price_Range / Open | Relative daily movement |
| MA5 | 5-day rolling average of Close | Short-term trend |
| MA20 | 20-day rolling average of Close | Long-term trend |
| DayOfWeek | 0 = Monday, 4 = Friday | Calendar effect |
| Month | 1 - 12 | Seasonal effect |

### Hyperparameter Tuning
Grid Search with 5-fold cross-validation was used to find the best settings for each model.

| Model | Parameters Tuned |
|-------|-----------------|
| Ridge Regression | alpha |
| Random Forest | n_estimators, max_depth, min_samples_split |
| Gradient Boosting | n_estimators, learning_rate, max_depth |

---

## Results

Approximate results on the 7-day test set:

| Model | RMSE (Yen) | MAE (Yen) | R2 | MAPE (%) |
|-------|-----------|----------|-----|---------|
| Ridge Regression | ~1,200 | ~950 | ~0.85 | ~2.6% |
| Random Forest | ~800 | ~620 | ~0.93 | ~1.7% |
| **Gradient Boosting** | **~650** | **~510** | **~0.97** | **~1.4%** |

**Winner: Gradient Boosting** — achieved the lowest RMSE and highest R2 score.

### Metric Explanations
- **RMSE** (Root Mean Squared Error): Average prediction error in Yen. Lower is better. Penalises large errors more.
- **MAE** (Mean Absolute Error): Average absolute error in Yen. Lower is better. Treats all errors equally.
- **R2**: How much of the price variation the model explains. 1.0 = perfect, 0 = no better than guessing the mean.
- **MAPE**: Average percentage error. Lower is better.

---

## Key Findings

- Gradient Boosting performed best by iteratively correcting its own prediction errors
- Random Forest was a close second, benefiting from averaging many independent trees
- Ridge Regression established a solid linear baseline but missed non-linear price patterns
- Feature engineering (especially MA5, MA20, and Volatility) improved all three models
- Open, High, Low, and Avg_Price were consistently the most important features

---

## Challenges

- The test set contains only 7 days, which limits the statistical reliability of the metrics
- Stock prices are influenced by news and market sentiment that are not captured in OHLCV data
- Careful handling was required to avoid data leakage during the scaling step
- Grid Search over many hyperparameter combinations is computationally slow on CPU

---

## Requirements

All libraries are pre-installed in Google Colab. If running locally:

```
pip install pandas numpy matplotlib seaborn scikit-learn
```

---

## Disclaimer

This project is for educational purposes only. Past stock performance does not guarantee future results. This is not financial advice.
