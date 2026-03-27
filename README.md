# Credit Risk Prediction System

A production-style machine learning project designed for fintech credit underwriting. The system predicts borrower default probability, segments risk into actionable tiers, and surfaces the key drivers behind model decisions.

## Business Problem
Consumer lending teams need to identify high-risk applicants before approval while maintaining portfolio growth. Manual rule-based screening misses nonlinear risk signals and does not scale.

## Objective
Build an end-to-end predictive system that:
- Estimates default probability for each borrower.
- Segments risk into Low, Medium, and High tiers.
- Provides feature importance insights for credit strategy and policy updates.

## Solution Overview
The pipeline ingests Lending Club loan records, cleans and engineers predictive features, trains a gradient-boosted classifier (XGBoost with CatBoost fallback), and exports model outputs for reporting and deployment.

## Dataset
- Source: Lending Club historical loans (`loan.csv`)
- Core fields used: `annual_inc`, `dti`, `int_rate`, `grade`, `term`, `loan_amnt`, `installment`, and categorical borrower attributes.
- Target: `default` where Charged Off = 1 and Fully Paid = 0.

## Modeling Approach
- Primary model: XGBoost (`XGBClassifier`)
- Fallback model: CatBoost (`CatBoostClassifier`)
- Final fallback: scikit-learn Gradient Boosting
- Evaluation metrics: AUC, Accuracy, classification report
- Explainability: feature importance export (`results/feature_importance.csv`)

## Key Results
- Main KPI: AUC on holdout set (stored in `results/metrics.json`)
- Business insight outputs:
  - Top risk drivers by importance
  - Probability-level scoring for applicants
  - Risk segmentation: Low (<0.20), Medium (0.20-0.49), High (>=0.50)

## How To Run
1. Create and activate a virtual environment.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Place dataset at `data/loan.csv`.
4. Run pipeline:
   ```bash
   python -m src.pipeline
   ```
5. Review outputs in `results/`.

## Project Structure
```text
credit-risk-prediction-system/
├── data/
├── notebooks/
│   └── credit_risk_analysis.ipynb
├── src/
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── model_training.py
│   ├── evaluation.py
│   ├── inference.py
│   └── pipeline.py
├── results/
├── tests/
├── requirements.txt
└── README.md
```

## Portfolio Value
This project demonstrates production-oriented ML engineering for credit risk: modular architecture, reproducible pipeline execution, model interpretability, and business-aligned decision outputs.
