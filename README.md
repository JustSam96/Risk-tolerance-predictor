# Investor Risk Tolerance Prediction

A regression project that predicts an individual investor's **risk tolerance** from demographic and financial profile data, using the Federal Reserve's **Survey of Consumer Finances (SCF) 2009 panel**. Multiple models are compared, tuned via grid search, and the best model (Random Forest) is saved for reuse.

Adapted from the risk-tolerance-prediction example in *Machine Learning and Data Science Blueprints for Finance*.

## Problem

Predict an investor's **true risk tolerance** — the proportion of their portfolio allocated to risky assets (stocks, bonds, mutual funds) versus risk-free assets (cash, savings bonds, CDs) — from features like age, education, marital status, income, and net worth. This is framed as a regression problem: the target is a continuous ratio between 0 and 1, not a risk category label.

## Data

**Source**: SCF 2009 panel (`SCFP2009panel.xlsx`) — a longitudinal survey tracking the same households' finances in both **2007** (pre-financial-crisis) and **2009** (post-crisis), 19,285 rows × 515 columns of raw survey data.

Having both pre- and post-crisis snapshots for the same households is what makes this dataset useful here: it lets the notebook construct a risk tolerance measure that isn't just a snapshot in time, but reflects each household's *stable* risk appetite across a major market shock.

## Target Construction

Risk tolerance for each year is computed as:

```
RiskFree = LIQ + CDS + SAVBND + CASHLI   (cash, CDs, savings bonds, life insurance cash value)
Risky     = NMMF + STOCKS + BOND          (mutual funds, stocks, bonds)
RT        = Risky / (Risky + RiskFree)
```

2009's ratio is additionally adjusted by the change in the S&P 500 between the two years, to correct for the market crash mechanically shrinking the risky-asset share even for investors who didn't change their behavior:

```
RT09 = (Risky09 / (Risky09 + RiskFree09)) * (Average_SP500_2009 / Average_SP500_2007)
```

**Filtering for behavioral stability**: households whose risk tolerance changed by more than 10% between 2007 and 2009 are dropped, on the reasoning that a large swing likely reflects a life event or panic-driven decision rather than the household's underlying, stable risk tolerance. For the remaining households, the final **`TrueRiskTolerance`** target is the average of their 2007 and 2009 ratios.

## Features

After dropping irrelevant/redundant columns, the feature set is narrowed to:

| Feature | Description |
|---|---|
| `AGE07` | Age |
| `EDCL07` | Education level |
| `MARRIED07` | Marital status |
| `KIDS07` | Number of children |
| `OCCAT107` | Occupation category |
| `INCOME07` | Household income |
| `RISK07` | Self-reported risk attitude |
| `NETWORTH07` | Net worth |

## Data Cleaning

- Rows with missing values dropped (`dropna`)
- Rows containing `inf`/`-inf` (from the risk-tolerance ratio calculations) dropped
- Distribution plots of `RT07`/`RT09` and a correlation heatmap/scatter matrix used to sanity-check the constructed target and feature relationships before modeling

## Train/Test Split

Standard 80/20 random split (`random_state=3`), since this is cross-sectional survey data rather than a time series — no chronological ordering constraint applies here.

## Models Compared

Evaluated via 10-fold cross-validation, scored on **R²**:

| Model | CV R² (mean) |
|---|---|
| Linear Regression | 0.103 |
| **Lasso** | **0.042** *(best linear model)* |
| ElasticNet | 0.048 |
| K-Nearest Neighbors | 0.425 (worse) |
| Decision Tree | 0.557 (worse) |
| SVR | 0.128 |
| MLP (untuned) | *diverged — invalid result, see note below* |
| AdaBoost | 0.372 (worse) |
| Gradient Boosting | 0.622 (worse) |
| Random Forest | 0.715 (worse) |
| Extra Trees | 0.699 (worse) |

*(Note: scores above are cross-validation error values as printed in the notebook; lower is better in this raw form since they're negated internally by scikit-learn — Random Forest and Extra Trees show the best raw cross-validated fit.)*

**MLP diverged badly** in the initial spot-check (score on the order of 10¹²), almost certainly due to unscaled input features (income and net worth range into the hundreds of thousands, versus 0/1 categorical features) — a known failure mode for neural networks without feature scaling. It was not pursued further as a candidate.

## Final Model: Tuned Random Forest

Random Forest was selected and tuned via grid search over `n_estimators`:

```python
param_grid = {'n_estimators': [50,100,150,200,250,300,350,400]}
```

**Best configuration: `n_estimators=200`** (CV R² ≈ 0.712)

Final performance:

| Metric | Train | Validation |
|---|---|---|
| R² | 0.965 | 0.761 |
| MSE | — | 0.0078 |

The gap between train and validation R² (0.965 vs 0.761) suggests some overfitting, typical of Random Forest without depth/leaf-size constraints — worth keeping in mind when interpreting the model's real-world generalization.

## Feature Importance

The top predictor by a wide margin was **net worth**, followed by **age** and **self-reported risk attitude (`RISK07`)** — income, education, marital status, kids, and occupation category contributed comparatively little.

## Model Persistence

```python
from pickle import dump
dump(model, open('risktolerance_model.sav', 'wb'))
```

To reload:
```python
from pickle import load
model = load(open('risktolerance_model.sav', 'rb'))
predictions = model.predict(X_new)  # columns: AGE07, EDCL07, MARRIED07, KIDS07, OCCAT107, INCOME07, RISK07, NETWORTH07
```

No feature scaler is required — the saved model was trained on raw (unscaled) feature values.

## Repository Contents

```
risk-tolerance-prediction.ipynb   # Full notebook: target construction → cleaning → EDA → model comparison → RF tuning → feature importance → persistence
risktolerance_model.sav           # Pickled, tuned RandomForestRegressor (n_estimators=200)
```

## Requirements

```
numpy
pandas
matplotlib
seaborn
scikit-learn
keras
scikeras
statsmodels
openpyxl   # for reading the .xlsx source file
```

## Usage

```python
from pickle import load
import pandas as pd

model = load(open('risktolerance_model.sav', 'rb'))

new_investor = pd.DataFrame([{
    'AGE07': 45, 'EDCL07': 4, 'MARRIED07': 1, 'KIDS07': 2,
    'OCCAT107': 1, 'INCOME07': 120000, 'RISK07': 3, 'NETWORTH07': 500000
}])

predicted_risk_tolerance = model.predict(new_investor)
print(predicted_risk_tolerance)  # proportion of portfolio expected in risky assets
```

## Known Limitations / Next Steps

- **Data is from 2007–2009**, spanning the global financial crisis. Investor behavior, product availability, and macroeconomic conditions have changed substantially since then — predictions may not generalize well to current investor populations without retraining on more recent survey data.
- **The 10% stability filter removes a meaningful subset of households**, specifically those with the most behavioral change — this could bias the model toward "typical, stable" investors and away from unusual scenarios or high-volatility responders.
- **Train/validation R² gap (0.965 vs 0.761) suggests overfitting.** Tuning `max_depth`, `min_samples_leaf`, or `min_samples_split` alongside `n_estimators` — rather than tuning `n_estimators` alone — would likely close this gap and improve generalization.
- **MLP was not properly evaluated** due to the scaling issue. `StandardScaler` is imported in the notebook but never actually applied to `X_train`/`X_validation` (the relevant lines are commented out) — enabling it could make MLP (and possibly SVR) more competitive.
- **No hyperparameter tuning for the other strong performers** (Gradient Boosting, Extra Trees) — only Random Forest was tuned; a comparison of tuned versions of each might change which model comes out on top.
- **Self-reported `RISK07` as a feature is somewhat circular** — since it's a self-reported risk attitude, its predictive power partly reflects investors accurately describing their own behavior, which limits how much genuine "prediction" is happening versus consistency-checking self-report against actual portfolio allocation.
