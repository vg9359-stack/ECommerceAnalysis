import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Set visual style for charts
sns.set_theme(style="whitegrid")
plt.rcParams.update({'font.size': 10})

# =========================================================
# Step 1: Data Acquisition & Loading
# =========================================================
basket_df = pd.read_csv(r"C:\Users\Viswa\Documents\ECommerceAnalysis\basket_details.csv")
customer_df = pd.read_csv(r"C:\Users\Viswa\Documents\ECommerceAnalysis\customer_details.csv")
print("--- Raw Basket Details Overview ---")
print(basket_df.info())

print("\n--- Raw Customer Details Overview ---")
print(customer_df.info())

# =========================================================
# Step 2: Data Cleaning & Preprocessing
# =========================================================
# 1. Convert basket_date string object to datetime format
basket_df['basket_date'] = pd.to_datetime(basket_df['basket_date'])

# 2. Handle invalid customer ages (negative values and outliers > 100)
valid_age_median = customer_df.loc[
    (customer_df['customer_age'] >= 18) & (customer_df['customer_age'] <= 100),
    'customer_age',
].median()

customer_df_cleaned = customer_df.copy()
customer_df_cleaned['customer_age_cleaned'] = customer_df_cleaned[
    'customer_age'
].apply(lambda age: age if 18 <= age <= 100 else valid_age_median)

# 3. Deduplicate across datasets
basket_df_cleaned = basket_df.drop_duplicates()
customer_df_cleaned = customer_df_cleaned.drop_duplicates()

# 4. Merge transaction and customer records on customer_id
merged_df = pd.merge(
    basket_df_cleaned, customer_df_cleaned, on='customer_id', how='inner'
)

print("\n--- Merged Dataset Summary Statistics ---")
print(merged_df[['basket_count', 'customer_age_cleaned', 'tenure']].describe())

# =========================================================
# Step 3: Exploratory Data Analysis & Visualizations
# =========================================================

# --- Visualization 1: Customer Age Boxplot (Raw) vs. Histogram (Cleaned) ---
fig, ax = plt.subplots(1, 2, figsize=(11, 4))

sns.boxplot(data=customer_df, y='customer_age', ax=ax[0], color='#ff7f0e')
ax[0].set_title('Raw Customer Age (Outliers Present)')
ax[0].set_ylabel('Age (Years)')

sns.histplot(
    customer_df_cleaned['customer_age_cleaned'],
    bins=25,
    kde=True,
    ax=ax[1],
    color='#1f77b4',
)
ax[1].set_title('Cleaned Customer Age Distribution')
ax[1].set_xlabel('Age (Years)')

plt.tight_layout()
plt.savefig(os.path.abspath('viz1_data_quality.png'), dpi=300)
plt.close()

# --- Visualization 2: Correlation Matrix Heatmap ---
plt.figure(figsize=(6, 4.5))
corr_matrix = merged_df[
    ['basket_count', 'customer_age_cleaned', 'tenure']
].corr()

sns.heatmap(corr_matrix, annot=True, cmap='Blues', fmt='.3f', vmin=-1, vmax=1)
plt.title('Correlation Matrix (Basket Count, Age, Tenure)')
plt.tight_layout()
plt.savefig(os.path.abspath('viz2_correlation_heatmap.png'), dpi=300)
plt.close()

# --- Visualization 3: Daily Total Basket Items Purchasing Trend (Line Graph) ---
plt.figure(figsize=(9, 4))
daily_trend = (
    merged_df.groupby('basket_date')['basket_count'].sum().reset_index()
)

sns.lineplot(
    data=daily_trend,
    x='basket_date',
    y='basket_count',
    marker='o',
    color='teal',
    linewidth=2,
)
plt.title('Daily Total Basket Items Trend')
plt.xlabel('Basket Date')
plt.ylabel('Total Items Purchased')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.abspath('viz3_daily_trend.png'), dpi=300)
plt.close()

print("\nExecution complete! All 3 figures generated and saved successfully.")

