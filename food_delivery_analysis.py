import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix,
                             classification_report)

warnings.filterwarnings('ignore')

VIZ_DIR    = "output/visualizations"
MODEL_DIR  = "output/models"
REPORT_DIR = "output/reports"
RESULT_DIR = "output/results"
for d in [VIZ_DIR, MODEL_DIR, REPORT_DIR, RESULT_DIR]:
    os.makedirs(d, exist_ok=True)

print("=" * 65)
print("  FOOD DELIVERY DELAY ANALYSIS AND PREDICTION")
print("=" * 65)


print("\n--- PHASE 1: LOADING DATASET ---")
df = pd.read_csv("food_delivery_200.csv")
print(f"Shape          : {df.shape[0]} rows x {df.shape[1]} columns")
print(f"\nColumn Names:\n{list(df.columns)}")
print(f"\nFirst 5 Rows:\n{df.head()}")
print(f"\nData Types:\n{df.dtypes}")
print(f"\nMissing Values:\n{df.isnull().sum()}")
print(f"\nDuplicate Rows : {df.duplicated().sum()}")
print(f"Duplicate Order_IDs: {df['Order_ID'].duplicated().sum()}")
print("\nUnique values per categorical column:")
for col in ['Restaurant','Area','Traffic','Weather','Peak_Hour','Rider_ID']:
    print(f"  {col}: {sorted(df[col].unique())}")

print("\n--- PHASE 2: DATA CLEANING ---")
df['Order_Time'] = pd.to_datetime(df['Order_Time'], errors='coerce')
print(f"Invalid Order_Time values: {df['Order_Time'].isnull().sum()}")

for col in ['Distance_KM','Preparation_Time','Delivery_Time']:
    df[col] = pd.to_numeric(df[col], errors='coerce')
    neg = (df[col] < 0).sum()
    print(f"Negative values in {col}: {neg}")
    if df[col].isnull().sum() > 0:
        df[col].fillna(df[col].median(), inplace=True)

for col in ['Restaurant','Area','Traffic','Weather','Rider_ID']:
    if df[col].isnull().sum() > 0:
        df[col].fillna(df[col].mode()[0], inplace=True)

before = len(df)
df = df[(df['Distance_KM'] >= 0) & (df['Preparation_Time'] >= 0) & (df['Delivery_Time'] >= 0)]
df.drop_duplicates(inplace=True)
print(f"Rows removed (invalid/duplicate): {before - len(df)}")

df['Peak_Hour'] = df['Peak_Hour'].map({'Yes': 1, 'No': 0})
print(f"Clean dataset shape: {df.shape}")
print(f"\nDescriptive Statistics:\n{df.describe()}")


print("\n--- PHASE 3: FEATURE ENGINEERING ---")
df['Total_Time']   = df['Preparation_Time'] + df['Delivery_Time']
df['Order_Hour']   = df['Order_Time'].dt.hour

def get_time_period(h):
    if 5 <= h < 12:   return 'Morning'
    elif 12 <= h < 17: return 'Afternoon'
    elif 17 <= h < 21: return 'Evening'
    else:              return 'Night'

df['Time_Period'] = df['Order_Hour'].apply(get_time_period)

# Delay target — 75th percentile of Total_Time (no explicit SLA column in dataset)
delay_threshold = df['Total_Time'].quantile(0.75)
df['Delay'] = (df['Total_Time'] > delay_threshold).astype(int)

print(f"Delay threshold (75th pct of Total_Time): {delay_threshold:.2f} min")
print(f"Delayed orders (Delay=1): {df['Delay'].sum()}")
print(f"Normal  orders (Delay=0): {(df['Delay']==0).sum()}")
print(f"Overall delay rate      : {df['Delay'].mean()*100:.1f}%")
print(f"\nTime Period counts:\n{df['Time_Period'].value_counts()}")
df.to_csv(f"{RESULT_DIR}/cleaned_food_delivery.csv", index=False)
print(f"Cleaned dataset saved -> {RESULT_DIR}/cleaned_food_delivery.csv")


print("\n--- PHASE 4: EXPLORATORY DATA ANALYSIS ---")

print(f"\nTotal Orders       : {len(df)}")
print(f"Restaurants        : {df['Restaurant'].nunique()}")
print(f"Areas              : {df['Area'].nunique()}")
print(f"Riders             : {df['Rider_ID'].nunique()}")
print(f"Avg Distance       : {df['Distance_KM'].mean():.2f} KM")
print(f"Avg Prep Time      : {df['Preparation_Time'].mean():.2f} min")
print(f"Avg Delivery Time  : {df['Delivery_Time'].mean():.2f} min")
print(f"Avg Total Time     : {df['Total_Time'].mean():.2f} min")
print(f"Min Delivery Time  : {df['Delivery_Time'].min():.2f} min")
print(f"Max Delivery Time  : {df['Delivery_Time'].max():.2f} min")
print(f"Median Delivery Time: {df['Delivery_Time'].median():.2f} min")

print("\n--- Average Delivery Time by Restaurant ---")
rest_avg = df.groupby('Restaurant')['Delivery_Time'].mean().sort_values(ascending=False)
print(rest_avg.round(2))

print("\n--- Average Delivery Time by Area ---")
area_avg = df.groupby('Area')['Delivery_Time'].mean().sort_values(ascending=False)
print(area_avg.round(2))

print("\n--- Average Delivery Time by Traffic ---")
print(df.groupby('Traffic')['Delivery_Time'].mean().round(2))

print("\n--- Average Delivery Time by Weather ---")
print(df.groupby('Weather')['Delivery_Time'].mean().round(2))

print("\n--- Average Delivery Time by Peak Hour ---")
print(df.groupby('Peak_Hour')['Delivery_Time'].mean().round(2))

print("\n--- Average Delivery Time by Time Period ---")
print(df.groupby('Time_Period')['Delivery_Time'].mean().round(2))


print("\n--- PHASE 5: RESTAURANT ANALYSIS ---")

rest_analysis = df.groupby('Restaurant').agg(
    Orders=('Order_ID','count'),
    Avg_Prep=('Preparation_Time','mean'),
    Avg_Delivery=('Delivery_Time','mean'),
    Avg_Total=('Total_Time','mean'),
    Delay_Rate=('Delay','mean')
).round(2)
rest_analysis['Delay_Rate_%'] = (rest_analysis['Delay_Rate'] * 100).round(1)
rest_analysis = rest_analysis.drop(columns='Delay_Rate').sort_values('Delay_Rate_%', ascending=False)
print(rest_analysis.to_string())
rest_analysis.to_csv(f"{RESULT_DIR}/restaurant_analysis.csv")

overall_avg_prep     = df['Preparation_Time'].mean()
overall_avg_delivery = df['Delivery_Time'].mean()
print(f"\nOverall Avg Prep Time    : {overall_avg_prep:.2f} min")
print(f"Overall Avg Delivery Time: {overall_avg_delivery:.2f} min")
print("\nRestaurant Problem Classification:")
for rest, row in rest_analysis.iterrows():
    prep_issue     = row['Avg_Prep']     > overall_avg_prep
    delivery_issue = row['Avg_Delivery'] > overall_avg_delivery
    if prep_issue and delivery_issue:
        issue = "COMBINED (Prep + Delivery)"
    elif prep_issue:
        issue = "PREPARATION issue"
    elif delivery_issue:
        issue = "DELIVERY issue"
    else:
        issue = "No major issue"
    print(f"  {rest:<18}: {issue}")


print("\n--- PHASE 6: AREA ANALYSIS ---")
area_analysis = df.groupby('Area').agg(
    Orders=('Order_ID','count'),
    Avg_Prep=('Preparation_Time','mean'),
    Avg_Delivery=('Delivery_Time','mean'),
    Avg_Total=('Total_Time','mean'),
    Delay_Rate=('Delay','mean')
).round(2)
area_analysis['Delay_Rate_%'] = (area_analysis['Delay_Rate'] * 100).round(1)
area_analysis = area_analysis.drop(columns='Delay_Rate').sort_values('Delay_Rate_%', ascending=False)
print(area_analysis.to_string())
area_analysis.to_csv(f"{RESULT_DIR}/area_analysis.csv")


print("\n--- PHASE 7: TRAFFIC ANALYSIS ---")
traffic_analysis = df.groupby('Traffic').agg(
    Orders=('Order_ID','count'),
    Avg_Delivery=('Delivery_Time','mean'),
    Avg_Total=('Total_Time','mean'),
    Delay_Rate=('Delay','mean')
).round(2)
traffic_analysis['Delay_Rate_%'] = (traffic_analysis['Delay_Rate'] * 100).round(1)
traffic_analysis = traffic_analysis.drop(columns='Delay_Rate')
print(traffic_analysis.to_string())
traffic_analysis.to_csv(f"{RESULT_DIR}/traffic_analysis.csv")


print("\n--- PHASE 8: WEATHER ANALYSIS ---")
weather_analysis = df.groupby('Weather').agg(
    Orders=('Order_ID','count'),
    Avg_Delivery=('Delivery_Time','mean'),
    Avg_Total=('Total_Time','mean'),
    Delay_Rate=('Delay','mean')
).round(2)
weather_analysis['Delay_Rate_%'] = (weather_analysis['Delay_Rate'] * 100).round(1)
weather_analysis = weather_analysis.drop(columns='Delay_Rate')
print(weather_analysis.to_string())
weather_analysis.to_csv(f"{RESULT_DIR}/weather_analysis.csv")


print("\n--- PHASE 9: PEAK HOUR ANALYSIS ---")
peak_analysis = df.groupby('Peak_Hour').agg(
    Orders=('Order_ID','count'),
    Avg_Prep=('Preparation_Time','mean'),
    Avg_Delivery=('Delivery_Time','mean'),
    Avg_Total=('Total_Time','mean'),
    Delay_Rate=('Delay','mean')
).round(2)
peak_analysis['Delay_Rate_%'] = (peak_analysis['Delay_Rate'] * 100).round(1)
peak_analysis.index = peak_analysis.index.map({1:'Peak', 0:'Non-Peak'})
peak_analysis = peak_analysis.drop(columns='Delay_Rate')
print(peak_analysis.to_string())
peak_analysis.to_csv(f"{RESULT_DIR}/peak_hour_analysis.csv")


print("\n--- PHASE 10: TIME PERIOD ANALYSIS ---")
period_order = ['Morning','Afternoon','Evening','Night']
period_analysis = df.groupby('Time_Period').agg(
    Orders=('Order_ID','count'),
    Avg_Delivery=('Delivery_Time','mean'),
    Avg_Total=('Total_Time','mean'),
    Delay_Rate=('Delay','mean'),
    Delayed_Orders=('Delay','sum')
).round(2)
period_analysis['Delay_Rate_%'] = (period_analysis['Delay_Rate'] * 100).round(1)
period_analysis = period_analysis.drop(columns='Delay_Rate')
period_analysis = period_analysis.reindex([p for p in period_order if p in period_analysis.index])
print(period_analysis.to_string())
period_analysis.to_csv(f"{RESULT_DIR}/time_period_analysis.csv")


print("\n--- PHASE 11: RIDER ANALYSIS ---")
MIN_ORDERS = 5
rider_analysis = df.groupby('Rider_ID').agg(
    Orders=('Order_ID','count'),
    Avg_Delivery=('Delivery_Time','mean'),
    Avg_Total=('Total_Time','mean'),
    Delay_Rate=('Delay','mean')
).round(2)
rider_analysis['Delay_Rate_%'] = (rider_analysis['Delay_Rate'] * 100).round(1)
rider_analysis = rider_analysis.drop(columns='Delay_Rate').sort_values('Delay_Rate_%', ascending=False)
qualified = rider_analysis[rider_analysis['Orders'] >= MIN_ORDERS]
print(f"Riders with >= {MIN_ORDERS} orders (reliable sample):")
print(qualified.to_string())
rider_analysis.to_csv(f"{RESULT_DIR}/rider_analysis.csv")


print("\n--- PHASE 12: DISTANCE ANALYSIS ---")
corr_dist_delivery = df['Distance_KM'].corr(df['Delivery_Time'])
corr_dist_total    = df['Distance_KM'].corr(df['Total_Time'])
corr_prep_total    = df['Preparation_Time'].corr(df['Total_Time'])
print(f"Correlation Distance_KM vs Delivery_Time : {corr_dist_delivery:.3f}")
print(f"Correlation Distance_KM vs Total_Time    : {corr_dist_total:.3f}")
print(f"Correlation Preparation_Time vs Total_Time: {corr_prep_total:.3f}")
print("\nNote: Correlation does not imply causation.")
print("Random Forest feature importance will show predictive usefulness.")


print("\n--- PHASE 4: EXPLORATORY DATA ANALYSIS ---")
print(f"Total Orders        : {len(df)}")
print(f"Restaurants         : {df['Restaurant'].nunique()}")
print(f"Areas               : {df['Area'].nunique()}")
print(f"Riders              : {df['Rider_ID'].nunique()}")
print(f"Avg Distance        : {df['Distance_KM'].mean():.2f} KM")
print(f"Avg Prep Time       : {df['Preparation_Time'].mean():.2f} min")
print(f"Avg Delivery Time   : {df['Delivery_Time'].mean():.2f} min")
print(f"Avg Total Time      : {df['Total_Time'].mean():.2f} min")
print(f"Min Delivery Time   : {df['Delivery_Time'].min():.2f} min")
print(f"Max Delivery Time   : {df['Delivery_Time'].max():.2f} min")
print(f"Median Delivery Time: {df['Delivery_Time'].median():.2f} min")
print(f"\nDescriptive Stats:\n{df.describe().round(2)}")

print("\n-- Avg Delivery Time by Restaurant --")
print(df.groupby('Restaurant')['Delivery_Time'].mean().sort_values(ascending=False).round(2))
print("\n-- Avg Delivery Time by Area --")
print(df.groupby('Area')['Delivery_Time'].mean().sort_values(ascending=False).round(2))
print("\n-- Avg Delivery Time by Traffic --")
print(df.groupby('Traffic')['Delivery_Time'].mean().round(2))
print("\n-- Avg Delivery Time by Weather --")
print(df.groupby('Weather')['Delivery_Time'].mean().round(2))
print("\n-- Avg Delivery Time by Peak Hour --")
print(df.groupby('Peak_Hour')['Delivery_Time'].mean().round(2))
print("\n-- Avg Delivery Time by Time Period --")
print(df.groupby('Time_Period')['Delivery_Time'].mean().round(2))


print("\n--- PHASE 5: RESTAURANT ANALYSIS ---")
rest_analysis = df.groupby('Restaurant').agg(
    Orders=('Order_ID','count'),
    Avg_Prep=('Preparation_Time','mean'),
    Avg_Delivery=('Delivery_Time','mean'),
    Avg_Total=('Total_Time','mean'),
    Delay_Rate=('Delay','mean')
).round(2)
rest_analysis['Delay_Rate_%'] = (rest_analysis['Delay_Rate']*100).round(1)
rest_analysis.drop(columns='Delay_Rate', inplace=True)
rest_analysis.sort_values('Delay_Rate_%', ascending=False, inplace=True)
print(rest_analysis.to_string())
rest_analysis.to_csv(f"{RESULT_DIR}/restaurant_analysis.csv")

overall_avg_prep     = df['Preparation_Time'].mean()
overall_avg_delivery = df['Delivery_Time'].mean()
print(f"\nOverall Avg Prep Time    : {overall_avg_prep:.2f} min")
print(f"Overall Avg Delivery Time: {overall_avg_delivery:.2f} min")
print("\nRestaurant Problem Classification:")
for rest, row in rest_analysis.iterrows():
    p = row['Avg_Prep']     > overall_avg_prep
    d = row['Avg_Delivery'] > overall_avg_delivery
    if p and d:   label = "COMBINED (Prep + Delivery)"
    elif p:       label = "PREPARATION issue"
    elif d:       label = "DELIVERY issue"
    else:         label = "No major issue"
    print(f"  {rest:<18}: {label}")

print("\n--- PHASE 6: AREA ANALYSIS ---")
area_analysis = df.groupby('Area').agg(
    Orders=('Order_ID','count'),
    Avg_Prep=('Preparation_Time','mean'),
    Avg_Delivery=('Delivery_Time','mean'),
    Avg_Total=('Total_Time','mean'),
    Delay_Rate=('Delay','mean')
).round(2)
area_analysis['Delay_Rate_%'] = (area_analysis['Delay_Rate']*100).round(1)
area_analysis.drop(columns='Delay_Rate', inplace=True)
area_analysis.sort_values('Delay_Rate_%', ascending=False, inplace=True)
print(area_analysis.to_string())
area_analysis.to_csv(f"{RESULT_DIR}/area_analysis.csv")

print("\n--- PHASE 7: TRAFFIC ANALYSIS ---")
traffic_analysis = df.groupby('Traffic').agg(
    Orders=('Order_ID','count'),
    Avg_Delivery=('Delivery_Time','mean'),
    Avg_Total=('Total_Time','mean'),
    Delay_Rate=('Delay','mean')
).round(2)
traffic_analysis['Delay_Rate_%'] = (traffic_analysis['Delay_Rate']*100).round(1)
traffic_analysis.drop(columns='Delay_Rate', inplace=True)
print(traffic_analysis.to_string())
traffic_analysis.to_csv(f"{RESULT_DIR}/traffic_analysis.csv")

print("\n--- PHASE 8: WEATHER ANALYSIS ---")
weather_analysis = df.groupby('Weather').agg(
    Orders=('Order_ID','count'),
    Avg_Delivery=('Delivery_Time','mean'),
    Avg_Total=('Total_Time','mean'),
    Delay_Rate=('Delay','mean')
).round(2)
weather_analysis['Delay_Rate_%'] = (weather_analysis['Delay_Rate']*100).round(1)
weather_analysis.drop(columns='Delay_Rate', inplace=True)
print(weather_analysis.to_string())
weather_analysis.to_csv(f"{RESULT_DIR}/weather_analysis.csv")

print("\n--- PHASE 9: PEAK HOUR ANALYSIS ---")
peak_analysis = df.groupby('Peak_Hour').agg(
    Orders=('Order_ID','count'),
    Avg_Prep=('Preparation_Time','mean'),
    Avg_Delivery=('Delivery_Time','mean'),
    Avg_Total=('Total_Time','mean'),
    Delay_Rate=('Delay','mean')
).round(2)
peak_analysis['Delay_Rate_%'] = (peak_analysis['Delay_Rate']*100).round(1)
peak_analysis.drop(columns='Delay_Rate', inplace=True)
peak_analysis.index = peak_analysis.index.map({1:'Peak', 0:'Non-Peak'})
print(peak_analysis.to_string())
peak_analysis.to_csv(f"{RESULT_DIR}/peak_hour_analysis.csv")

print("\n--- PHASE 10: TIME PERIOD ANALYSIS ---")
period_analysis = df.groupby('Time_Period').agg(
    Orders=('Order_ID','count'),
    Avg_Delivery=('Delivery_Time','mean'),
    Avg_Total=('Total_Time','mean'),
    Delay_Rate=('Delay','mean'),
    Delayed_Orders=('Delay','sum')
).round(2)
period_analysis['Delay_Rate_%'] = (period_analysis['Delay_Rate']*100).round(1)
period_analysis.drop(columns='Delay_Rate', inplace=True)
for p in ['Morning','Afternoon','Evening','Night']:
    if p in period_analysis.index:
        print(f"  {p}: {period_analysis.loc[p].to_dict()}")
period_analysis.to_csv(f"{RESULT_DIR}/time_period_analysis.csv")


print("\n--- PHASE 11: RIDER ANALYSIS ---")
MIN_ORDERS = 5
rider_analysis = df.groupby('Rider_ID').agg(
    Orders=('Order_ID','count'),
    Avg_Delivery=('Delivery_Time','mean'),
    Avg_Total=('Total_Time','mean'),
    Delay_Rate=('Delay','mean')
).round(2)
rider_analysis['Delay_Rate_%'] = (rider_analysis['Delay_Rate']*100).round(1)
rider_analysis.drop(columns='Delay_Rate', inplace=True)
rider_analysis.sort_values('Delay_Rate_%', ascending=False, inplace=True)
qualified = rider_analysis[rider_analysis['Orders'] >= MIN_ORDERS]
print(f"Riders with >= {MIN_ORDERS} orders:")
print(qualified.to_string())
rider_analysis.to_csv(f"{RESULT_DIR}/rider_analysis.csv")


print("\n--- PHASE 12: DISTANCE ANALYSIS ---")
print(f"Corr Distance_KM vs Delivery_Time  : {df['Distance_KM'].corr(df['Delivery_Time']):.3f}")
print(f"Corr Distance_KM vs Total_Time     : {df['Distance_KM'].corr(df['Total_Time']):.3f}")
print(f"Corr Preparation_Time vs Total_Time: {df['Preparation_Time'].corr(df['Total_Time']):.3f}")
print("Note: Correlation does not imply causation.")


print("\n--- PHASE 13: CREATING VISUALIZATIONS ---")


fig, ax = plt.subplots(figsize=(12, 6))
data = df.groupby('Restaurant')['Delivery_Time'].mean().sort_values(ascending=False)
sns.barplot(x=data.index, y=data.values, palette='coolwarm', ax=ax)
ax.set_title('Average Delivery Time by Restaurant', fontsize=14, fontweight='bold')
ax.set_xlabel('Restaurant', fontsize=12)
ax.set_ylabel('Avg Delivery Time (min)', fontsize=12)
ax.tick_params(axis='x', rotation=45)
for i, v in enumerate(data.values):
    ax.text(i, v + 0.3, f'{v:.1f}', ha='center', fontsize=9)
plt.tight_layout()
plt.savefig(f"{VIZ_DIR}/01_avg_delivery_by_restaurant.png", dpi=150)
plt.close()
print("Saved: 01_avg_delivery_by_restaurant.png")


prep = df.groupby('Restaurant')['Preparation_Time'].mean()
delv = df.groupby('Restaurant')['Delivery_Time'].mean()
x = np.arange(len(prep))
fig, ax = plt.subplots(figsize=(12, 6))
ax.bar(x - 0.2, prep.values, 0.4, label='Avg Prep Time', color='steelblue')
ax.bar(x + 0.2, delv.values, 0.4, label='Avg Delivery Time', color='tomato')
ax.set_xticks(x)
ax.set_xticklabels(prep.index, rotation=45, ha='right')
ax.set_title('Preparation Time vs Delivery Time by Restaurant', fontsize=14, fontweight='bold')
ax.set_xlabel('Restaurant', fontsize=12)
ax.set_ylabel('Time (min)', fontsize=12)
ax.legend(fontsize=11)
plt.tight_layout()
plt.savefig(f"{VIZ_DIR}/02_prep_vs_delivery_by_restaurant.png", dpi=150)
plt.close()
print("Saved: 02_prep_vs_delivery_by_restaurant.png")


fig, ax = plt.subplots(figsize=(8, 6))
order = ['Low','Medium','High']
sns.boxplot(data=df, x='Traffic', y='Delivery_Time', order=order, palette='Set2', ax=ax)
ax.set_title('Traffic Condition vs Delivery Time', fontsize=14, fontweight='bold')
ax.set_xlabel('Traffic Level', fontsize=12)
ax.set_ylabel('Delivery Time (min)', fontsize=12)
plt.tight_layout()
plt.savefig(f"{VIZ_DIR}/03_traffic_vs_delivery_time.png", dpi=150)
plt.close()
print("Saved: 03_traffic_vs_delivery_time.png")


fig, ax = plt.subplots(figsize=(9, 6))
colors = {'Low':'green','Medium':'orange','High':'red'}
for traffic, grp in df.groupby('Traffic'):
    ax.scatter(grp['Distance_KM'], grp['Delivery_Time'],
               c=colors[traffic], label=traffic, alpha=0.6, edgecolors='k', linewidths=0.3)
ax.set_title('Distance vs Delivery Time (colored by Traffic)', fontsize=14, fontweight='bold')
ax.set_xlabel('Distance (KM)', fontsize=12)
ax.set_ylabel('Delivery Time (min)', fontsize=12)
ax.legend(title='Traffic', fontsize=10)
plt.tight_layout()
plt.savefig(f"{VIZ_DIR}/04_distance_vs_delivery_time.png", dpi=150)
plt.close()
print("Saved: 04_distance_vs_delivery_time.png")

fig, ax = plt.subplots(figsize=(10, 6))
area_delay = df.groupby('Area')['Delay'].mean().sort_values(ascending=False) * 100
sns.barplot(x=area_delay.index, y=area_delay.values, palette='Reds_r', ax=ax)
ax.set_title('Delay Rate by Area (%)', fontsize=14, fontweight='bold')
ax.set_xlabel('Area', fontsize=12)
ax.set_ylabel('Delay Rate (%)', fontsize=12)
ax.tick_params(axis='x', rotation=45)
for i, v in enumerate(area_delay.values):
    ax.text(i, v + 0.3, f'{v:.1f}%', ha='center', fontsize=9)
plt.tight_layout()
plt.savefig(f"{VIZ_DIR}/05_delay_rate_by_area.png", dpi=150)
plt.close()
print("Saved: 05_delay_rate_by_area.png")


fig, ax = plt.subplots(figsize=(8, 6))
sns.boxplot(data=df, x='Weather', y='Delivery_Time', palette='pastel', ax=ax)
ax.set_title('Weather Condition vs Delivery Time', fontsize=14, fontweight='bold')
ax.set_xlabel('Weather', fontsize=12)
ax.set_ylabel('Delivery Time (min)', fontsize=12)
plt.tight_layout()
plt.savefig(f"{VIZ_DIR}/06_weather_vs_delivery_time.png", dpi=150)
plt.close()
print("Saved: 06_weather_vs_delivery_time.png")


fig, ax = plt.subplots(figsize=(8, 5))
tp_counts = df['Time_Period'].value_counts().reindex(['Morning','Afternoon','Evening','Night'])
sns.barplot(x=tp_counts.index, y=tp_counts.values, palette='Blues_d', ax=ax)
ax.set_title('Order Volume by Time Period', fontsize=14, fontweight='bold')
ax.set_xlabel('Time Period', fontsize=12)
ax.set_ylabel('Number of Orders', fontsize=12)
for i, v in enumerate(tp_counts.values):
    ax.text(i, v + 0.3, str(v), ha='center', fontsize=10)
plt.tight_layout()
plt.savefig(f"{VIZ_DIR}/07_orders_by_time_period.png", dpi=150)
plt.close()
print("Saved: 07_orders_by_time_period.png")


print("\n--- PHASE 14: RANDOM FOREST MODEL ---")



FEATURES = ['Distance_KM', 'Preparation_Time', 'Peak_Hour', 'Order_Hour']
CAT_FEATURES = ['Traffic', 'Weather', 'Area', 'Restaurant']


df_ml = pd.get_dummies(df[FEATURES + CAT_FEATURES + ['Delay']], columns=CAT_FEATURES)
feature_cols = [c for c in df_ml.columns if c != 'Delay']

X = df_ml[feature_cols]
y = df_ml['Delay']

print(f"Features used : {len(feature_cols)}")
print(f"Class balance : {y.value_counts().to_dict()}")


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)
print(f"Train size: {len(X_train)} | Test size: {len(X_test)}")


rf = RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced')
rf.fit(X_train, y_train)
print("Model trained.")


y_pred      = rf.predict(X_test)
y_pred_prob = rf.predict_proba(X_test)[:, 1]


print("\n--- PHASE 15: MODEL EVALUATION ---")
acc  = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, zero_division=0)
rec  = recall_score(y_test, y_pred, zero_division=0)
f1   = f1_score(y_test, y_pred, zero_division=0)
auc  = roc_auc_score(y_test, y_pred_prob)
cm   = confusion_matrix(y_test, y_pred)

print(f"Accuracy  : {acc:.4f}  ({acc*100:.1f}%)")
print(f"Precision : {prec:.4f}")
print(f"Recall    : {rec:.4f}")
print(f"F1-Score  : {f1:.4f}")
print(f"ROC-AUC   : {auc:.4f}")
print(f"\nConfusion Matrix:\n{cm}")
print(f"\nClassification Report:\n{classification_report(y_test, y_pred, target_names=['Normal','Delayed'])}")



metrics = pd.DataFrame({
    'Metric':['Accuracy','Precision','Recall','F1-Score','ROC-AUC'],
    'Score':[acc, prec, rec, f1, auc]
})
metrics.to_csv(f"{RESULT_DIR}/model_metrics.csv", index=False)


joblib.dump(rf, f"{MODEL_DIR}/random_forest_model.pkl")
print(f"Model saved -> {MODEL_DIR}/random_forest_model.pkl")


fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Normal','Delayed'],
            yticklabels=['Normal','Delayed'], ax=ax)
ax.set_title('Confusion Matrix — Random Forest', fontsize=13, fontweight='bold')
ax.set_xlabel('Predicted', fontsize=11)
ax.set_ylabel('Actual', fontsize=11)
plt.tight_layout()
plt.savefig(f"{VIZ_DIR}/08_confusion_matrix.png", dpi=150)
plt.close()
print("Saved: 08_confusion_matrix.png")


print("\n--- PHASE 16: FEATURE IMPORTANCE ---")
importance_df = pd.DataFrame({
    'Feature': feature_cols,
    'Importance': rf.feature_importances_
}).sort_values('Importance', ascending=False).reset_index(drop=True)

print(importance_df.head(15).to_string())
importance_df.to_csv(f"{RESULT_DIR}/feature_importance.csv", index=False)


top15 = importance_df.head(15)
fig, ax = plt.subplots(figsize=(10, 7))
sns.barplot(data=top15, x='Importance', y='Feature', palette='viridis', ax=ax)
ax.set_title('Random Forest — Top 15 Feature Importances', fontsize=13, fontweight='bold')
ax.set_xlabel('Importance Score', fontsize=11)
ax.set_ylabel('Feature', fontsize=11)
plt.tight_layout()
plt.savefig(f"{VIZ_DIR}/09_feature_importance.png", dpi=150)
plt.close()
print("Saved: 09_feature_importance.png")


print("\n--- PHASE 17: KEY INSIGHTS ---")

top_rest_delay   = rest_analysis['Delay_Rate_%'].idxmax()
top_area_delay   = area_analysis['Delay_Rate_%'].idxmax()
top_traffic_del  = traffic_analysis['Avg_Delivery'].idxmax()
top_weather_del  = weather_analysis['Avg_Delivery'].idxmax()
top_feature      = importance_df.iloc[0]['Feature']
peak_delay       = peak_analysis.loc['Peak','Delay_Rate_%']   if 'Peak'     in peak_analysis.index else 'N/A'
nonpeak_delay    = peak_analysis.loc['Non-Peak','Delay_Rate_%'] if 'Non-Peak' in peak_analysis.index else 'N/A'
dist_corr        = round(df['Distance_KM'].corr(df['Delivery_Time']), 3)

insights = f"""
=============================================================
  KEY INSIGHTS
=============================================================

Insight 1:
  Finding: Delay threshold is {delay_threshold:.1f} minutes (75th percentile of Total_Time).
  Evidence: {df['Delay'].sum()} out of {len(df)} orders ({df['Delay'].mean()*100:.1f}%) are classified as delayed.
  Operational meaning: 1 in 4 orders exceeds the acceptable operational time window.

Insight 2:
  Finding: {top_rest_delay} has the highest delay rate among all restaurants.
  Evidence: Delay Rate = {rest_analysis.loc[top_rest_delay,'Delay_Rate_%']}%,
            Avg Prep = {rest_analysis.loc[top_rest_delay,'Avg_Prep']} min,
            Avg Delivery = {rest_analysis.loc[top_rest_delay,'Avg_Delivery']} min.
  Operational meaning: This restaurant requires immediate kitchen and dispatch review.

Insight 3:
  Finding: {top_area_delay} area has the highest delay rate.
  Evidence: Delay Rate = {area_analysis.loc[top_area_delay,'Delay_Rate_%']}%,
            Avg Total Time = {area_analysis.loc[top_area_delay,'Avg_Total']} min.
  Operational meaning: Delivery zone planning and rider allocation need review for this area.

Insight 4:
  Finding: {top_traffic_del} traffic has the longest average delivery time.
  Evidence: Avg Delivery Time = {traffic_analysis.loc[top_traffic_del,'Avg_Delivery']} min.
  Operational meaning: Additional rider capacity should be pre-positioned during high-traffic periods.

Insight 5:
  Finding: {top_weather_del} weather is associated with the longest average delivery time.
  Evidence: Avg Delivery Time = {weather_analysis.loc[top_weather_del,'Avg_Delivery']} min.
  Operational meaning: Operational planning should account for adverse weather conditions.

Insight 6:
  Finding: Peak hour orders show a delay rate of {peak_delay}% vs {nonpeak_delay}% for non-peak.
  Evidence: See peak_hour_analysis.csv in output/results.
  Operational meaning: Staffing and rider pre-positioning during peak hours can reduce delays.

Insight 7:
  Finding: Distance correlation with Delivery_Time is {dist_corr} — moderate, not dominant.
  Evidence: Random Forest top feature = '{top_feature}', suggesting other factors matter more.
  Operational meaning: Distance alone does not explain delays; traffic, prep time, and area
                       are equally or more important predictive factors.

=============================================================
  PRACTICAL ACTION PLAN
=============================================================

Restaurant Operations:
  - Monitor kitchen preparation time for {top_rest_delay} — consider staffing review.
  - Set internal SLA alerts when Preparation_Time exceeds {overall_avg_prep:.0f} minutes.
  - Increase kitchen staff during peak hours at high-delay restaurants.

Delivery Operations:
  - Review delivery routes for {top_area_delay} area.
  - Assign experienced riders to long-distance or high-traffic zones.
  - Track rider-level delay rates monthly (use rider_analysis.csv).

Traffic Management:
  - Pre-position additional riders during High traffic periods.
  - Use route optimization tools to avoid congested zones.

Peak-Hour Management:
  - Increase rider availability during peak hours.
  - Notify customers of expected longer wait times during peak periods.

Weather Management:
  - During Storm/Rain conditions, proactively adjust estimated delivery times.
  - Consider weather-based dynamic staffing.

=============================================================
"""
print(insights)

# Save insights to report
with open(f"{REPORT_DIR}/key_insights_and_action_plan.txt", "w") as f:
    f.write(insights)
print(f"Insights saved -> {REPORT_DIR}/key_insights_and_action_plan.txt")

print("\n" + "=" * 65)
print("  ALL PHASES COMPLETE")
print(f"  Visualizations : {VIZ_DIR}/")
print(f"  Results        : {RESULT_DIR}/")
print(f"  Model          : {MODEL_DIR}/")
print(f"  Report         : {REPORT_DIR}/")
print("=" * 65)
