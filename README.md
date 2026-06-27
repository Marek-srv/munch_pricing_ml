# Munch Onam Pricing ML Project

This project builds a machine learning pricing recommendation system for Nestlé Munch using historical Nielsen-style chocolate pricing, sales, distribution, geography, market, and competitor data.

The main business objective is to estimate the recommended Munch ASP ₹/10g for the upcoming Onam season, especially across South India and Kerala-relevant markets.

The final output is an Excel-based pricing report containing model performance, feature importance, segment-level recommendations, top market recommendations, and final pricing bands.

---

## 1. Project Objective

The goal of this project is not just to forecast sales or market share.

The main goal is to estimate:

Recommended Munch ASP Rs per 10g

for Onam-relevant markets using:

- Historical Munch ASP
- Perk ASP
- Snakker ASP
- Market geography
- Market type
- Period/year trends
- Volume share
- Value share
- Numeric distribution
- Weighted distribution
- Retailer distribution
- PDO volume
- Onam market relevance

The final recommendation is given as a pricing range:

- Conservative ASP
- Recommended ASP
- Aggressive ASP

---

## 2. Final Business Result

The final Onam pricing recommendation from the robust model is:

Recommended Munch ASP: ₹5.33 per 10g  
Conservative ASP: ₹5.15 per 10g  
Aggressive ASP: ₹5.51 per 10g  

So the recommended pricing band is:

₹5.15 – ₹5.51 per 10g

The central business recommendation is:

₹5.33 per 10g

This should be treated as a decision-support recommendation, not an exact fixed price.

---

## 3. Dataset

The original Excel file is placed in:

data/raw/munch_pricing_dataset.xlsx

The main sheet used is:

ML_Dataset

Original dataset shape:

Rows: 16,704  
Columns: 43  

After cleaning:

Rows: 15,354  
Columns: 34  

After feature engineering:

Rows: 15,354  
Columns: 56  

---

## 4. Main Target Variable

The target variable is:

Munch_ASP_Rs_per_10g

This represents Munch Average Selling Price in rupees per 10 grams.

Target summary after cleaning:

Mean ASP: ₹4.87 per 10g  
Min ASP: ₹1.71 per 10g  
Max ASP: ₹6.94 per 10g  

---

## 5. Important Input Variables

The model uses variables from these groups.

### Time Variables

Report_Year  
Period_Year  
Period_Type  

These are used to capture pricing trend and inflation-like changes over time.

### Market Variables

Source_Sheet  
Market_Type  
Market  

These are used to capture geography, channel, metro/base metro, and regional market behavior.

### Munch Business Variables

Munch_VolShare_pct  
Munch_ValShare_pct  
Munch_ND_pct  
Munch_WD_pct  
Munch_Distribution_Retailers  
Munch_PDO_Volume  

These are used to capture Munch's market strength, distribution, and productivity.

### Competitor Variables

Perk_ASP_Rs_per_10g  
Perk_VolShare_pct  
Perk_ValShare_pct  
Perk_ND_pct  
Perk_WD_pct  
Perk_Distribution_Retailers  
Perk_PDO_Volume  

Snakker_ASP_Rs_per_10g  
Snakker_VolShare_pct  
Snakker_ValShare_pct  
Snakker_ND_pct  
Snakker_WD_pct  
Snakker_Distribution_Retailers  
Snakker_PDO_Volume  

These are used to capture competitor pricing and competitive strength.

---

## 6. Leakage Handling

Some columns were removed because they directly use the target variable.

These columns were dropped:

RelativePrice_Munch_vs_Perk  
RelativePrice_Munch_vs_Snakker  
PriceGap_Munch_Perk_Rs_per_10g  
PriceGap_Munch_Snakker_Rs_per_10g  

These columns were also excluded from the final robust decision model:

Munch_Volume_KG  
Munch_Value_INR000  
Period  

Reason:

- Munch_Value_INR000 and Munch_Volume_KG can indirectly recreate ASP.
- Period contains highly specific old labels like DEC22, Q4'22, MAT '23, which can cause overfitting.
- The robust model is intended to generalize better for future Onam pricing.

---

## 7. Feature Engineering

Feature engineering creates stronger business signals from raw data.

Created features include:

Competitor_Avg_ASP  
Perk_Snakker_PriceGap  
Perk_to_Snakker_ASP_Ratio  

Munch_vs_Perk_VolShare_Gap  
Munch_vs_Perk_ValShare_Gap  
Munch_vs_Snakker_VolShare_Gap  
Munch_vs_Snakker_ValShare_Gap  

Munch_Distribution_Strength  
Perk_Distribution_Strength  
Snakker_Distribution_Strength  

Munch_vs_Perk_Distribution_Gap  
Munch_vs_Snakker_Distribution_Gap  

Is_South_Market  
Is_Kerala_Market  
Is_Metro_Market  
Is_Channel_Market  

Is_Fiscal_Year  
Is_MAT  
Is_Quarter  
Is_Month  
Is_Onam_Period  
Is_Onam_Relevant  

The Onam flag is a proxy based on:

South/Kerala market + Aug/Sep/Q2/Q3 period relevance

---

## 8. Model Validation Approach

The model uses time-based validation.

Training data:

2016, 2019, 2020, 2023

Test data:

2026

This is better than random train-test split because the business problem is future pricing estimation.

---

## 9. Models Trained

The following models were trained and compared:

Linear Regression  
Ridge Regression  
Random Forest  
Gradient Boosting  
XGBoost  

Two model styles were tested.

### Full Accuracy Model

Uses most variables, including Munch volume and value.

Purpose:

Understand maximum possible prediction accuracy.

### Decision Model

Removes direct leakage variables.

Purpose:

Create a realistic pricing recommendation model.

### Robust Decision Model

Removes leakage variables and high-cardinality Period labels.

Purpose:

Create a safer model for future Onam pricing recommendation.

---

## 10. Final Selected Model

The final selected model is:

Robust Gradient Boosting Model

Saved at:

models/robust_final_model.pkl

Final robust model test performance:

MAE: 0.1801  
RMSE: 0.2773  
R2: 0.0463  
Bias: 0.0192  

Interpretation:

The model predicts Munch ASP within around ₹0.18 per 10g on average.

---

## 11. Feature Importance

Top model drivers:

Period_Year  
Snakker_ASP_Rs_per_10g  
Munch_Distribution_Retailers  
Munch_VolShare_pct  
Perk_ASP_Rs_per_10g  
Munch_PDO_Volume  
Period_Type  
Munch_ValShare_pct  
Market  
Source_Sheet  

Business interpretation:

Munch ASP is mainly driven by time trend, competitor prices, distribution reach, Munch share strength, and geography.

---

## 12. Onam Recommendation Output

The Onam recommendation engine filters latest-year South/Onam-relevant markets and predicts Munch ASP.

Final Onam summary:

Rows: 891  
Actual ASP Mean: ₹5.28  
Predicted ASP Mean: ₹5.33  
Conservative ASP Mean: ₹5.15  
Aggressive ASP Mean: ₹5.51  

Top opportunity markets include:

Kerala Rural  
Kerala U+R  
Kerala Urban  
Kochi  
South Zone Rural  
Karnataka Rural  
Tamil Nadu Rural  
Telangana Rural  

Kerala markets show the strongest pricing opportunity due to higher distribution strength and stronger value share.

---

## 13. Project Folder Structure

munch_pricing_ml/
│
├── data/
│   ├── raw/
│   │   └── munch_pricing_dataset.xlsx
│   │
│   └── processed/
│       ├── cleaned_munch_pricing.csv
│       └── featured_munch_pricing.csv
│
├── models/
│   ├── decision_model_best_model.pkl
│   ├── final_decision_model.pkl
│   └── robust_final_model.pkl
│
├── reports/
│   ├── figures/
│   │   ├── 01_munch_asp_distribution.png
│   │   ├── 02_munch_asp_trend_by_year.png
│   │   ├── 03_munch_asp_by_market_type.png
│   │   ├── 04_munch_asp_by_source_sheet.png
│   │   ├── 05_munch_vs_perk_asp.png
│   │   ├── 06_munch_vs_snakker_asp.png
│   │   ├── 07_munch_vs_competitor_avg_asp.png
│   │   ├── 08_onam_relevant_asp.png
│   │   └── 09_correlation_heatmap.png
│   │
│   └── outputs/
│       ├── eda_summary.xlsx
│       ├── baseline_model_results.xlsx
│       ├── decision_model_diagnostics.xlsx
│       ├── advanced_model_results.xlsx
│       ├── advanced_model_predictions.xlsx
│       ├── robust_model_results.xlsx
│       ├── robust_model_predictions.xlsx
│       ├── robust_model_feature_importance.xlsx
│       ├── onam_price_recommendations.xlsx
│       └── final_munch_onam_pricing_report.xlsx
│
├── src/
│   ├── data_inspection.py
│   ├── data_cleaning.py
│   ├── feature_engineering.py
│   ├── eda_analysis.py
│   ├── train_baseline_model.py
│   ├── model_diagnostics.py
│   ├── train_advanced_models.py
│   ├── train_robust_model.py
│   ├── onam_price_recommendation.py
│   └── final_report_generator.py
│
├── requirements.txt
└── README.md

---

## 14. Setup Instructions

Create a Python virtual environment:

python3.10 -m venv .venv

Activate the environment:

source .venv/bin/activate

Upgrade pip:

pip install --upgrade pip

Install requirements:

pip install -r requirements.txt

If LightGBM gives installation issues, it can be skipped. The final selected model does not require LightGBM.

---

## 15. Requirements

Recommended requirements.txt:

pandas  
numpy  
openpyxl  
scikit-learn  
matplotlib  
xgboost  
joblib  

Optional:

lightgbm

---

## 16. Execution Order

Run the project scripts in this order.

### Step 1: Inspect Excel File

python src/data_inspection.py

Purpose:

Check sheet names, columns, data types, missing values, and sample rows.

---

### Step 2: Clean Data

python src/data_cleaning.py

Output:

data/processed/cleaned_munch_pricing.csv

Purpose:

Remove missing target rows, drop metadata/leakage columns, handle missing values.

---

### Step 3: Feature Engineering

python src/feature_engineering.py

Output:

data/processed/featured_munch_pricing.csv

Purpose:

Create competitor, share, distribution, geography, and Onam-relevance features.

---

### Step 4: EDA Analysis

python src/eda_analysis.py

Outputs:

reports/figures/  
reports/outputs/eda_summary.xlsx  

Purpose:

Understand ASP distribution, yearly trends, market differences, competitor relationship, and correlations.

---

### Step 5: Baseline Model

python src/train_baseline_model.py

Outputs:

reports/outputs/baseline_model_results.xlsx  
models/decision_model_best_model.pkl  

Purpose:

Compare Linear Regression, Ridge Regression, and Random Forest.

---

### Step 6: Model Diagnostics

python src/model_diagnostics.py

Output:

reports/outputs/decision_model_diagnostics.xlsx

Purpose:

Check 2026 prediction errors, model bias, market-level errors, and Onam-specific errors.

---

### Step 7: Advanced Models

python src/train_advanced_models.py

Outputs:

reports/outputs/advanced_model_results.xlsx  
reports/outputs/advanced_model_predictions.xlsx  
models/final_decision_model.pkl  

Purpose:

Compare Random Forest, Gradient Boosting, XGBoost, and optionally LightGBM.

---

### Step 8: Robust Final Model

python src/train_robust_model.py

Outputs:

models/robust_final_model.pkl  
reports/outputs/robust_model_results.xlsx  
reports/outputs/robust_model_predictions.xlsx  
reports/outputs/robust_model_feature_importance.xlsx  

Purpose:

Train the final robust pricing model without leakage and unstable period labels.

---

### Step 9: Onam Price Recommendation

python src/onam_price_recommendation.py

Output:

reports/outputs/onam_price_recommendations.xlsx

Purpose:

Generate Onam-relevant pricing recommendations and pricing bands.

---

### Step 10: Final Report

python src/final_report_generator.py

Output:

reports/outputs/final_munch_onam_pricing_report.xlsx

Purpose:

Generate the final formatted Excel report for business presentation.

---

## 17. Final Output Files

The most important final files are:

models/robust_final_model.pkl  
reports/outputs/onam_price_recommendations.xlsx  
reports/outputs/final_munch_onam_pricing_report.xlsx  

The final business report contains:

Executive_Summary  
Segment_Recommendation  
Top_Market_Recommendations  
Market_Type_Summary  
Model_Performance  
Feature_Importance  
Methodology  
Assumptions  
Detailed_Data  

---

## 18. EDA Findings

Main EDA insights:

Munch ASP is strongly linked with competitor pricing.  
Perk ASP and competitor average ASP are strong predictors.  
Year and period trend are important.  
Onam-relevant markets have slightly lower average ASP than non-Onam markets.  
Market type differences exist, but are not extremely large.  

Strongest correlations with Munch ASP:

Competitor_Avg_ASP: 0.636  
Perk_ASP_Rs_per_10g: 0.620  
Period_Year: 0.526  
Report_Year: 0.508  
Snakker_ASP_Rs_per_10g: 0.433  

---

## 19. Model Interpretation

The final model suggests that Munch pricing power depends mainly on:

1. Year/period trend
2. Competitor ASP
3. Distribution strength
4. Retailer presence
5. Market geography
6. Munch volume share
7. Munch value share
8. Metro/South/Kerala relevance

This means Munch should not use one single flat price for all regions.

A better approach is:

Kerala and high-distribution South markets: upper pricing band  
Broader South markets: central pricing band  
Base metro or weaker markets: conservative pricing band  

---

## 20. Recommended Business Action

Final recommendation:

For the upcoming Onam season, use ₹5.33 per 10g as the central Munch ASP recommendation across Onam-relevant South markets.

Recommended pricing range:

₹5.15 – ₹5.51 per 10g

Market-specific interpretation:

Kerala markets: ₹5.40 – ₹5.55 per 10g  
South non-Kerala markets: ₹5.25 – ₹5.35 per 10g  
Base metro markets: ₹5.15 – ₹5.25 per 10g  

Kerala markets can support premium pricing because they show stronger Onam relevance, stronger distribution, and stronger value share.

---

## 21. Important Limitations

This model has some limitations:

1. Onam is not directly available as a festival column.
2. Onam relevance is created using a proxy.
3. Model predictions are conservative and compressed compared to actual ASP variation.
4. R2 is low because 2026 market-level price variation is difficult to explain fully.
5. The model should support business decisions, not replace pricing judgment.

The model MAE is around:

₹0.18 per 10g

So the pricing band is more reliable than a single exact price.

---

## 22. Future Improvements

Future improvements can include:

Add actual festival calendar data.  
Add SKU-level pack size information.  
Add promotion and discount variables.  
Add retailer/channel-level pricing.  
Add elasticity modeling.  
Add volume-response modeling.  
Add optimization layer for revenue/profit maximization.  
Add scenario simulator for Perk and Snakker price changes.  
Add confidence intervals by market.  

---

## 23. Project Status

Current project status:

Data inspection: Completed  
Data cleaning: Completed  
Feature engineering: Completed  
EDA: Completed  
Baseline modeling: Completed  
Advanced modeling: Completed  
Robust final model: Completed  
Onam recommendation engine: Completed  
Final Excel report: Completed  

Final selected model:

Robust Gradient Boosting Model

Final recommendation:

₹5.33 per 10g central ASP  
₹5.15 – ₹5.51 per 10g pricing band  

---

## 24. Disclaimer

This project is a machine learning based pricing decision-support system.

The output should be reviewed with business context, trade plans, pack strategy, competitor actions, retailer margin requirements, and brand objectives before final pricing decisions are made.

The model does not guarantee actual sales lift or profitability. It estimates a statistically supported ASP recommendation based on historical market behavior.# munch_pricing_ml
