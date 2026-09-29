import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set global visual style and cross-platform font setup
sns.set_theme(style="whitegrid")
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 11

# 1. Ingest datasets
basket_df = pd.read_csv(r"C:\Users\Viswa\OneDrive\Documents\ECommerceAnalysis\basket_details.csv")
customer_df = pd.read_csv(r"C:\Users\Viswa\OneDrive\Documents\ECommerceAnalysis\customer_details.csv")

# Clean & convert datetime and filter age bounds
basket_df['basket_date'] = pd.to_datetime(basket_df['basket_date'])
customer_clean = customer_df[(customer_df['customer_age'] >= 10) & (customer_df['customer_age'] <= 100)].copy()

# =========================================================
# VISUAL 1: Cumulative Basket Item Growth Over Time
# =========================================================
daily_baskets = basket_df.groupby('basket_date')['basket_count'].sum().reset_index()
daily_baskets['cumulative_count'] = daily_baskets['basket_count'].cumsum()

plt.figure(figsize=(10, 5))
plt.fill_between(daily_baskets['basket_date'], daily_baskets['cumulative_count'], color='#2b5c8f', alpha=0.5)
plt.plot(daily_baskets['basket_date'], daily_baskets['cumulative_count'], color='#2b5c8f', linewidth=2)
plt.title("Cumulative Basket Items Over Time", fontsize=14, fontweight="bold", pad=15)
plt.xlabel("Date")
plt.ylabel("Cumulative Basket Items")
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig("area_chart_cumulative_baskets.png", dpi=300)
plt.close()

# =========================================================
# VISUAL 2: Daily Basket Activity & Peak Event Annotation
# =========================================================
plt.figure(figsize=(10, 5))
plt.plot(daily_baskets['basket_date'], daily_baskets['basket_count'], color="#1f77b4", linewidth=2, marker='o', markersize=4)

max_row = daily_baskets.loc[daily_baskets['basket_count'].idxmax()]
plt.annotate(
    f'Peak: {max_row["basket_count"]} items\n({max_row["basket_date"].strftime("%Y-%m-%d")})',
    xy=(max_row['basket_date'], max_row['basket_count']),
    xytext=(max_row['basket_date'], max_row['basket_count'] * 0.85),
    arrowprops=dict(facecolor='red', shrink=0.08, width=1.5, headwidth=6),
    fontweight='bold',
    color='darkred'
)

plt.title("Daily Total Basket Activity with Peak Event", fontsize=14, fontweight="bold", pad=15)
plt.xlabel("Date")
plt.ylabel("Daily Basket Count")
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig("annotated_line_chart.png", dpi=300)
plt.close()

# =========================================================
# VISUAL 3: Day-of-Week Demand for Top 5 Products
# =========================================================
top_products = basket_df['product_id'].value_counts().head(5).index
df_top = basket_df[basket_df['product_id'].isin(top_products)].copy()
df_top['product_id'] = df_top['product_id'].astype(str)
df_top['day_of_week'] = df_top['basket_date'].dt.day_name()
day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

grouped_bar_data = df_top.groupby(['day_of_week', 'product_id'])['basket_count'].sum().reset_index()

plt.figure(figsize=(10, 5))
sns.barplot(
    data=grouped_bar_data, 
    x="day_of_week", 
    y="basket_count", 
    hue="product_id", 
    order=day_order,
    palette="Set2"
)
plt.title("Basket Count by Day of Week for Top 5 Products", fontsize=14, fontweight="bold", pad=15)
plt.xlabel("Day of Week")
plt.ylabel("Total Basket Count")
plt.legend(title="Product ID", bbox_to_anchor=(1.02, 1), loc='upper left')
plt.tight_layout()
plt.savefig("grouped_bar_chart.png", dpi=300)
plt.close()

# =========================================================
# VISUAL 4: Customer Age Distribution by Gender (Violin + Strip)
# =========================================================
plt.figure(figsize=(10, 5))
sns.violinplot(
    data=customer_clean, 
    x="sex", 
    y="customer_age", 
    hue="sex", 
    legend=False, 
    inner=None, 
    palette="Pastel1", 
    cut=0
)
sns.stripplot(
    data=customer_clean.sample(min(500, len(customer_clean)), random_state=42), 
    x="sex", 
    y="customer_age", 
    hue="sex", 
    legend=False, 
    jitter=0.2, 
    alpha=0.4, 
    palette="dark:black"
)

plt.title("Customer Age Distribution by Gender", fontsize=14, fontweight="bold", pad=15)
plt.xlabel("Gender")
plt.ylabel("Customer Age")
plt.tight_layout()
plt.savefig("violin_strip_plot.png", dpi=300)
plt.close()

# =========================================================
# VISUAL 5: Customer Tenure vs. Age Cohort Analysis (Regression)
# =========================================================
plt.figure(figsize=(10, 5))
sns.scatterplot(
    data=customer_clean.sample(min(1000, len(customer_clean)), random_state=42), 
    x="customer_age", 
    y="tenure", 
    alpha=0.6, 
    color="#2b5c8f",
    label="Customer Sample"
)
sns.regplot(
    data=customer_clean, 
    x="customer_age", 
    y="tenure", 
    scatter=False, 
    color="crimson", 
    line_kws={"linewidth": 2, "label": "Linear Trend Line"}
)

plt.title("Customer Tenure vs. Customer Age Cohort", fontsize=14, fontweight="bold", pad=15)
plt.xlabel("Customer Age")
plt.ylabel("Tenure (Months)")
plt.legend(loc="upper right")
plt.tight_layout()
plt.savefig("tenure_vs_age_analysis.png", dpi=300)
plt.close()

print("All 5 high-resolution visualizations created and saved successfully!")
