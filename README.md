# Fraud Detection Model

Can we predict which e-commerce transactions are fraudulent before payment is processed? This project builds a Random Forest classifier on 1.5M+ real-world e-commerce transactions to flag fraud — and treats the messy, imbalanced nature of the data as the actual problem to solve, not a footnote.

## Why fraud detection is a different kind of ML problem

Most ML tutorials optimize for accuracy. Fraud detection punishes that instinct.

Fraud is rare — in most real datasets, only 1–3% of transactions are fraudulent. A model that predicts "not fraud" for every single row would be 95–99% accurate, and completely useless. So this project evaluates on metrics that actually matter for a fraud team:

- **Recall** — of all actual fraud cases, how many did the model catch?
- **Precision** — of all cases the model flagged as fraud, how many were actually fraud?
- **F1 score** — the balance between the two

A model with 60% recall means 40% of fraud slipped through undetected — that's money leaving the business. A model with low precision means flooding the fraud team with false alarms and blocking legitimate customers. Which failure mode is more costly is a business decision, not a modeling one — but the model needs to be evaluated on the metrics that expose the tradeoff, not on accuracy, which hides it.

## Dataset

[Fraudulent E-Commerce Transactions](https://www.kaggle.com/datasets/shriyashjagtap/fraudulent-e-commerce-transactions) (Kaggle) — split across two CSV files, combined here into ~1.5 million transactions across 16 columns, with realistic e-commerce context: payment methods, product categories, device types, and customer demographics.

The dataset isn't included in this repo (see [Setup](#setup) below) — `.gitignore` excludes `*.csv`.

**Columns used:**

| Column | Description |
|---|---|
| `Transaction Amount` | Value of the transaction |
| `Transaction Date` | When it happened (dropped before modeling — see [Approach](#approach)) |
| `Payment Method` | Credit card, debit card, bank transfer, PayPal |
| `Product Category` | Electronics, clothing, home & garden, toys & games, health & beauty |
| `Customer Age` | Age of the customer — contains data quality issues, see below |
| `Device Used` | Mobile, desktop, tablet |
| `Account Age Days` | How old the customer's account is |
| `Transaction Hour` | Hour of day the transaction occurred |
| `Is Fraudulent` | **Target variable** (1 = fraud, 0 = legitimate) |

`Transaction ID`, `Customer ID`, `IP Address`, `Shipping/Billing Address`, and `Customer Location` are dropped before modeling — they're identifiers, not predictive signal, and would let the model memorize rather than generalize.

## Approach

1. **Combine** the two source CSVs into a single dataframe (~1.5M rows, 16 columns, no missing values).
2. **Clean** — `Customer Age` contains negative values and ages under 18. Rows with `Customer Age < 18` are dropped as a business rule: in most financial and e-commerce contexts, minors aren't valid account holders, so these are treated as data quality problems rather than a segment to model.
3. **Explore** fraud rate by payment method, product category, hour of day, and transaction amount, to understand what's actually driving fraud before throwing a model at it.
4. **Prepare** — drop identifier/address columns, one-hot encode categorical features (`Payment Method`, `Product Category`, `Device Used`).
5. **Train** a `RandomForestClassifier` (100 trees, `class_weight='balanced'` to counter the ~95/5 class imbalance, stratified 80/20 train/test split).
6. **Evaluate** using a classification report (precision/recall/F1 per class), a confusion matrix, and Random Forest feature importances — not accuracy.

## Setup

1. **Clone the repo**

   ```bash
   git clone <this-repo-url>
   cd FraudDetectionModel-main
   ```

2. **Create a virtual environment and install dependencies**

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate   # On Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Download the dataset**

   Download both CSV files from the [Kaggle dataset page](https://www.kaggle.com/datasets/shriyashjagtap/fraudulent-e-commerce-transactions) and place them in the project root as:

   ```
   Fraudulent_E-Commerce_Transaction_Data.csv
   Fraudulent_E-Commerce_Transaction_Data_2.csv
   ```

4. **Run**

   ```bash
   python fraud_detect.py
   ```

   The script prints dataset shape, null checks, class balance, and the age-cleaning summary to the console, then pops up a series of matplotlib windows (fraud rate by payment method / product category / hour, transaction amount boxplot, confusion matrix, and top-10 feature importances) as it runs. On the full 1.5M-row dataset this can take a while on a laptop — there's a commented-out sampling line (`df.sample(n=200_000, ...)`) in the script if you want to iterate faster during development.

## Findings

- **Fraud rate:** ~5% of transactions in the combined dataset are fraudulent.
- **Strongest predictors** (by Random Forest feature importance): transaction amount, account age, and transaction hour.
- Certain payment methods and product categories show a visibly higher fraud rate than others (see the bar charts the script generates).
- *Model recall/precision/F1 depend on the specific run — see the classification report printed by `fraud_detect.py` for current numbers.*

## Limitations

- **No real-time component.** This is a batch, offline model — it doesn't address streaming/low-latency scoring at the point of transaction.
- **No drift handling.** Fraud patterns evolve; this model would need periodic retraining and monitoring to stay useful in production.
- **No cost-benefit analysis.** The tradeoff between false positives (blocking legitimate customers) and false negatives (missed fraud, direct financial loss) isn't quantified here — in practice that tradeoff should drive the classification threshold, not the default 0.5 cutoff used by `.predict()`.
- **No hyperparameter tuning.** The Random Forest uses reasonable defaults (100 trees, balanced class weights) rather than a tuned configuration.

## Requirements

See `requirements.txt`. Core dependencies: `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`.

## License

Not specified.
