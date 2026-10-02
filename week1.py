import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Set visual style
sns.set_theme(style="whitegrid")
plt.rcParams.update({"font.size": 11, "figure.autolayout": True})

# ---------------------------------------------------------
# 1. DATA LOADING & INITIAL INSPECTION
# ---------------------------------------------------------
print("--- 1. Loading Datasets ---")
basket_df = pd.read_csv(r"C:\Users\Viswa\OneDrive\Documents\ECommerceAnalysis\basket_details.csv")
customer_df = pd.read_csv(r"C:\Users\Viswa\OneDrive\Documents\ECommerceAnalysis\customer_details.csv")

print(f"Raw Basket Data Shape: {basket_df.shape}")
print(f"Raw Customer Data Shape: {customer_df.shape}\n")

# ---------------------------------------------------------
# 2. DATA CLEANING & PREPROCESSING
# ---------------------------------------------------------
print("--- 2. Data Cleaning & Preprocessing ---")

# Step 2.1: Deduplication
basket_df = basket_df.drop_duplicates()
customer_df = customer_df.drop_duplicates()

# Step 2.2: Date Parsing & Temporal Feature Extraction
basket_df["basket_date"] = pd.to_datetime(basket_df["basket_date"])
basket_df["year_month"] = basket_df["basket_date"].dt.to_period("M")
basket_df["day_of_week"] = basket_df["basket_date"].dt.day_name()

# Step 2.3: Cleaning Customer Demographics
# Filtering plausible age bounds (18 to 100) and imputing invalid/missing with median
median_age = customer_df.loc[
    (customer_df["customer_age"] >= 18) & (customer_df["customer_age"] <= 100),
    "customer_age",
].median()

raw_age_series = customer_df["customer_age"].copy()
customer_df["customer_age_cleaned"] = customer_df["customer_age"].apply(
    lambda x: median_age if (pd.isna(x) or x < 18 or x > 100) else x
)

# Step 2.4: Merging Datasets
df_merged = pd.merge(basket_df, customer_df, on="customer_id", how="inner")
print(f"Cleaned Merged Data Shape: {df_merged.shape}\n")

# ---------------------------------------------------------
# 3. VISUALIZATION GENERATION
# ---------------------------------------------------------
print("--- 3. Generating Visualizations ---")

# Visualization 1: Data Quality Check (Raw Age Outliers vs Cleaned Distribution)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

sns.boxplot(y=raw_age_series, ax=axes[0], color="#e74c3c")
axes[0].set_title("Raw Customer Age (With Outliers & Errors)")
axes[0].set_ylabel("Customer Age")

sns.histplot(
    customer_df["customer_age_cleaned"],
    bins=20,
    kde=True,
    ax=axes[1],
    color="#2ecc71",
)
axes[1].set_title("Cleaned Customer Age Distribution (Median Imputed)")
axes[1].set_xlabel("Customer Age")
axes[1].set_ylabel("Frequency")

plt.suptitle("Visualization 1: Demographics Data Cleaning Impact", fontsize=14)
plt.savefig("viz1_data_quality.png", dpi=300)
plt.close()

# Visualization 2: Correlation Heatmap
fig, ax = plt.subplots(figsize=(8, 6))
corr_matrix = df_merged[
    ["basket_count", "customer_age_cleaned", "tenure"]
].corr()

sns.heatmap(
    corr_matrix, annot=True, cmap="Blues", fmt=".3f", linewidths=1, ax=ax
)
ax.set_title(
    "Visualization 2: Correlation Matrix (Basket Count, Age, Tenure)",
    fontsize=12,
)
plt.savefig("viz2_correlation_heatmap.png", dpi=300)
plt.close()

# Visualization 3: Daily Purchase Volume Trend
fig, ax = plt.subplots(figsize=(12, 5))
daily_trend = (
    df_merged.groupby("basket_date")["basket_count"].sum().reset_index()
)

ax.plot(
    daily_trend["basket_date"],
    daily_trend["basket_count"],
    color="#3498db",
    linewidth=1.5,
)
ax.set_title("Visualization 3: Total Daily Basket Volume Over Time", fontsize=12)
ax.set_xlabel("Date")
ax.set_ylabel("Total Items Purchased")
plt.xticks(rotation=45)
plt.savefig("viz3_daily_trend.png", dpi=300)
plt.close()

print("All visualizations saved successfully as PNG files.")
