# Food Delivery Delay Analysis and Prediction Using Random Forest

## Problem Statement
A food-delivery platform promises customers delivery within a specified time.
Customers in some locations are reporting frequent delays. Management wants to
understand the operational reasons behind these delays and predict which orders
are likely to be delayed.

## Objectives
1. Identify why orders are getting delayed
2. Separate restaurant preparation delays from delivery operation delays
3. Investigate whether distance is the biggest contributor to delay
4. Identify restaurants with operational problems
5. Identify locations with frequent delays
6. Analyze time periods with concentrated delays
7. Understand how traffic and weather affect delivery performance
8. Build a Random Forest model to predict delayed orders

## Dataset
File: `food_delivery_200.csv` — 200 orders, 11 columns

| Column | Description |
|---|---|
| Order_ID | Unique order identifier |
| Restaurant | Restaurant name (10 restaurants) |
| Area | Delivery area (10 areas) |
| Distance_KM | Distance from restaurant to customer (KM) |
| Preparation_Time | Kitchen preparation time (minutes) |
| Delivery_Time | Rider delivery time (minutes) |
| Traffic | Traffic level: Low / Medium / High |
| Weather | Weather condition: Clear / Cloudy / Rain / Storm |
| Rider_ID | Rider identifier (R001–R020) |
| Order_Time | Timestamp of order placement |
| Peak_Hour | Whether order was placed during peak hour (Yes/No) |

## Technologies
- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Random Forest (RandomForestClassifier)

## Methodology
```
Data Collection → Data Cleaning → EDA → Feature Engineering →
Visualization → Delay Analysis → Random Forest → Model Evaluation →
Business Insights → Action Plan
```

## How to Run
```bash
pip install -r requirements.txt
python food_delivery_analysis.py
```

## Output Structure
```
output/
├── visualizations/   # All charts (PNG)
├── models/           # Saved Random Forest model
├── reports/          # Key insights and action plan
└── results/          # CSV analysis tables
```

## Key Insights
See `output/reports/key_insights_and_action_plan.txt` after running the script.

## Model
- Algorithm: Random Forest Classifier
- Features: Distance_KM, Preparation_Time, Peak_Hour, Order_Hour, Traffic, Weather, Area, Restaurant
- Target: Delay (1 = Delayed, 0 = Normal)
- Delay defined as Total_Time > 75th percentile of Total_Time
- Train/Test Split: 80% / 20%, stratified, random_state=42
