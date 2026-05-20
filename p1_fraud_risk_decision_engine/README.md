# Fraud Risk Decision Engine

Fraud detection project using the IEEE-CIS dataset.

The project focuses on:
- temporal validation
- leakage-safe feature engineering
- historical and velocity features
- threshold tuning
- online scoring simulation

The idea was to move beyond a standard Kaggle workflow and think through some of the problems that show up in real fraud systems.


## Dataset

Dataset used:
- IEEE-CIS Fraud Detection

Main files:
- `train_transaction.csv`
- `train_identity.csv`

Both tables are joined using `TransactionID`.

`TransactionDT` is treated as a relative timeline and is used for:
- train-validation split
- historical feature generation
- velocity features

Dataset:
https://www.kaggle.com/competitions/ieee-fraud-detection


## Repo Structure

```text
Fraud-Risk-Decision-Engine/
│
├── notebooks/
│   ├── 01_data_join_and_split.ipynb
│   ├── 02_baseline_model.ipynb
│   ├── 03_feature_iteration.ipynb
│   ├── 04_history_features.ipynb
│   ├── 05_velocity_features.ipynb
│   └── 06_online_scoring_simulation.ipynb
│
├── docs/
│   ├── 1_problem_framing.md
│   ├── 2_data_sources_and_structure.md
│   ├── 3_exploratory_data_analysis.md
│   ├── 4_feature_engineering.md
│   ├── 5_data_leakage_prevention.md
│   ├── 6_modeling_and_calibration.md
│   ├── 7_threshold_optimization.md
│   └── 8_from_ieee_cis_to_production.md
│
├── src/
├── configs/
├── images/
├── requirements.txt
└── README.md
```



## Project Flow

```text
Raw Transactions
      ↓
Feature Engineering
      ↓
Temporal Validation
      ↓
Model Training
      ↓
Calibration
      ↓
Threshold Tuning
      ↓
Online Scoring Simulation
```

## Installation

```bash
git clone https://github.com/suthashok/ml-systems.git

cd Fraud-Risk-Decision-Engine

pip install -r requirements.txt
```

## Documentation

The docs follow the same order as the project flow:

1. `docs/1_problem_framing.md`
2. `docs/2_data_sources_and_structure.md`
3. `docs/3_exploratory_data_analysis.md`
4. `docs/4_feature_engineering.md`
5. `docs/5_data_leakage_prevention.md`
6. `docs/6_modelling_and_calibration.md`
7. `docs/7_threshold_optimization.md`
8. `docs/8_from_ieee_cis_to_production.md`

## Tech Stack

```python
Python
Pandas
NumPy
Scikit-learn
LightGBM
XGBoost
Matplotlib
Seaborn
```