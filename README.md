# Detecting-Financial-Impulse-Behaviour-in-Young-Adults
This project detects impulsive spending behaviour by analyzing transaction patterns such as timing, frequency, and category trends. It uses behavioural features and machine learning to generate an impulse risk score, identify key triggers, and provide personalized recommendations to encourage responsible financial decisions.
# Detecting Financial Impulse Behaviour in Young Adults

**Behavioural Analytics Hackathon — Problem Statement 2**

---

## Problem Overview

Young individuals often overspend due to emotional or impulsive decision-making. This project builds a behavioural analytics system that:

- Detects **impulsive spending patterns** from transaction data
- Identifies **emotional or timing-based triggers** (night, weekend, end-of-month)
- Predicts **high-risk upcoming spending behaviour**
- Generates **personalised behavioural nudges**

---

## Dataset

| Field | Details |
|---|---|
| **Source** | [Credit Card Transactions Dataset – Kaggle](https://www.kaggle.com/datasets/priyamchoksi/credit-card-transactions-dataset) |
| **License** | Apache 2.0 |
| **Size** | ~1.85M rows, 24 columns |
| **Date Range** | Jan 2019 – Jun 2020 |

### Why this dataset fits Behavioural Analytics

The dataset contains transaction **timestamps, amounts, merchant categories, and cardholder demographics** — exactly the signals needed to model impulsive financial behaviour. We engineer behavioural features (time patterns, velocity, z-scores, EOM surges) from raw transactions.

### Behavioural Features Engineered

| Feature | Behavioural Signal |
|---|---|
| `is_night / is_late_night` | Emotional/impulsive spending after hours |
| `is_weekend` | Leisure-driven purchases |
| `is_end_of_month` | Financial pressure → impulse spending |
| `amt_zscore` | Deviation from personal spending baseline |
| `txn_velocity_6h` | Buying spree / binge behaviour |
| `rapid_repeat` | Immediate re-purchase (< 5 min) |
| `impulse_cat_ratio_last10` | Category drift toward discretionary |
| `eom_spike` | End-of-month surge above 1.5× average |
| `rolling_mean_7/30` | Personal baseline for context |

---

## Architecture

```
credit_card_transactions.csv
         │
         ▼
 impulse_detection.py
  ├── load_and_preprocess()
  ├── engineer_features()       ← 24 behavioural features
  ├── create_impulse_label()    ← Rule-based ground truth (score ≥ 3/6 criteria)
  ├── train_model()             ← Random Forest, class_weight='balanced'
  ├── score_transactions()      ← Impulse Risk Score 0–100
  └── save_visualizations()     ← 6 analysis charts
         │
         ▼
 dashboard.py
  └── generate_html_dashboard() ← Interactive HTML dashboard
```

---

## How to Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Place dataset
Download `credit_card_transactions.csv` from Kaggle and place it in this folder.

### 3. Run the main pipeline
```bash
python impulse_detection.py credit_card_transactions.csv
```

### 4. Generate dashboard
```bash
python dashboard.py
```

### 5. View outputs
Open `outputs/dashboard.html` in a browser.

---

## Outputs

| File | Description |
|---|---|
| `outputs/cardholder_risk_profiles.csv` | Per-card risk scores, tiers, behavioural stats |
| `outputs/transaction_scores.csv` | Per-transaction impulse risk scores |
| `outputs/dashboard.html` | **Interactive HTML dashboard** |
| `outputs/fig1_risk_distribution.png` | Risk score & tier distribution |
| `outputs/fig2_spending_patterns.png` | Hourly patterns & category impulse rates |
| `outputs/fig3_feature_importance.png` | Model feature importance |
| `outputs/fig4_roc_curve.png` | Model ROC-AUC curve |
| `outputs/fig5_eom_surge.png` | End-of-month spending analysis |
| `outputs/fig6_nudge_card.png` | Sample personalised nudge card |

---

## Risk Scoring Logic

### Label Creation (Ground Truth)
A transaction is labelled **IMPULSE** if it satisfies ≥ 3 of 6 behavioural criteria:

1. Impulse merchant category (shopping, entertainment, misc, food_dining)
2. Amount z-score > 1.5 (significantly above personal baseline)
3. Night-time OR weekend transaction
4. High velocity (≥ 3 txns within 6 hours)
5. Rapid repeat (< 5 minutes since last transaction)
6. End-of-month AND above 1.5× usual spend

### Impulse Risk Score (0–100)
The Random Forest classifier outputs a probability → scaled to 0–100.

| Score Range | Risk Tier | Action |
|---|---|---|
| 60 – 100 | HIGH | Immediate intervention nudges |
| 35 – 59 | MEDIUM | Awareness nudges |
| 0 – 34 | LOW | Positive reinforcement |

---

## Model

- **Algorithm**: Random Forest Classifier (`n_estimators=150, max_depth=12`)
- **Class balancing**: `class_weight='balanced'`
- **Split**: 80/20 train/test, stratified
- **Evaluation**: ROC-AUC, F1-score, Classification Report

---

## Personalised Nudges

**HIGH Risk:**
-  Pause non-essential purchases for 24 hours
-  Set a daily spending cap in your banking app
-  Review this month's shopping & entertainment spend

**MEDIUM Risk:**
-  Try the 24-hour rule before buying
-  Move 20% of discretionary funds to savings
-  Batch shopping trips to reduce impulse buys

**LOW Risk:**
-  Your spending patterns look healthy
-  Automate savings with the surplus
-  Essential-to-impulse ratio is well balanced

---


Behavioural Analytics Hackathon · OrgX · 2025
