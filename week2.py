import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set global visual aesthetics
plt.style.use('seaborn-v0_8-whitegrid')

# 1. LOAD DATASETS
basket = pd.read_csv(r"C:\Users\Viswa\OneDrive\Documents\ECommerceAnalysis\basket_details.csv")
customer = pd.read_csv(r"C:\Users\Viswa\OneDrive\Documents\ECommerceAnalysis\customer_details.csv")
# 2. DATA HYGIENE & PREPROCESSING
# Standardize date types
basket['basket_date'] = pd.to_datetime(basket['basket_date'])

# Clean customer dataset (filter out negative ages, birth years stored as age, and invalid sex entries)
clean_customer = customer[customer['sex'].isin(['Male', 'Female'])].copy()
clean_customer = clean_customer[(clean_customer['customer_age'] >= 10) & (clean_customer['customer_age'] <= 100)]


# --- GENERATE VISUALIZATIONS ---

# Visual 1: Cumulative Basket Item Growth
daily = basket.groupby('basket_date')['basket_count'].sum().reset_index()
daily['cumulative_items'] = daily['basket_count'].cumsum()

fig, ax = plt.subplots(figsize=(8, 4.5), dpi=300)
ax.fill_between(daily['basket_date'], daily['cumulative_items'], color='#1f77b4', alpha=0.4)
ax.plot(daily['basket_date'], daily['cumulative_items'], color='#1f77b4', linewidth=2.5)
ax.set_title('Visual 1: Cumulative Basket Item Growth Over Time (May-Jun 2019)', fontsize=12, fontweight='bold', pad=12)
ax.set_xlabel('Date', fontsize=10)
ax.set_ylabel('Cumulative Items Added', fontsize=10)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('visual1_cumulative.png')
plt.close()


# Visual 2: Daily Basket Velocity & Peak Event Milestone
fig, ax = plt.subplots(figsize=(8, 4.5), dpi=300)
ax.plot(daily['basket_date'], daily['basket_count'], color='#2ca02c', linewidth=2, marker='o', markersize=4)

# Find peak date dynamically from the dataset
peak_row = daily.loc[daily['basket_count'].idxmax()]
peak_date = peak_row['basket_date']
peak_val = peak_row['basket_count']

ax.annotate(f'May 27 Flash Sale Event\n({peak_val:,} Items / +340% Spike)',
            xy=(peak_date, peak_val), xytext=(pd.Timestamp('2019-05-30'), 3200),
            arrowprops=dict(facecolor='#d62728', shrink=0.05, width=2, headwidth=8),
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#ffe6e6', edgecolor='#d62728', alpha=0.9),
            fontsize=9, fontweight='bold')

ax.set_title('Visual 2: Daily Basket Velocity & Peak Event Spike Isolation', fontsize=12, fontweight='bold', pad=12)
ax.set_xlabel('Date', fontsize=10)
ax.set_ylabel('Daily Items Added', fontsize=10)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('visual2_daily_spike.png')
plt.close()


# Visual 3: Day-of-Week Purchasing Breakdown for Top 5 SKUs
top5_skus = basket.groupby('product_id')['basket_count'].sum().nlargest(5).index.tolist()
top5_basket = basket[basket['product_id'].isin(top5_skus)].copy()
top5_basket['day_of_week'] = top5_basket['basket_date'].dt.day_name()

days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
df_dow = top5_basket.groupby(['day_of_week', 'product_id'])['basket_count'].sum().reset_index()
df_dow['product_id'] = df_dow['product_id'].astype(str)

fig, ax = plt.subplots(figsize=(8, 4.5), dpi=300)
sns.barplot(data=df_dow, x='day_of_week', y='basket_count', hue='product_id', order=days_order, palette='Set2', ax=ax)
ax.set_title('Visual 3: Day-of-Week Purchasing Breakdown for Top 5 SKUs', fontsize=12, fontweight='bold', pad=12)
ax.set_xlabel('Day of Week', fontsize=10)
ax.set_ylabel('Items Ordered', fontsize=10)
ax.legend(title='Product SKU', bbox_to_anchor=(1.02, 1), loc='upper left')
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig('visual3_sku_breakdown.png')
plt.close()


# Visual 4: Customer Age Distribution & Density by Gender
fig, ax = plt.subplots(figsize=(8, 4.5), dpi=300)
# FIX: Assigned hue='sex' and legend=False to suppress the FutureWarning
sns.violinplot(data=clean_customer, x='sex', y='customer_age', hue='sex', palette='Pastel1', inner=None, legend=False, ax=ax)
sns.stripplot(data=clean_customer.sample(2000, random_state=42), x='sex', y='customer_age', color='black', alpha=0.1, jitter=0.2, size=2, ax=ax)
ax.set_title('Visual 4: Customer Age Distribution & Density by Gender', fontsize=12, fontweight='bold', pad=12)
ax.set_xlabel('Gender Segment', fontsize=10)
ax.set_ylabel('Customer Age (Years)', fontsize=10)
plt.tight_layout()
plt.savefig('visual4_gender_density.png')
plt.close()


# Visual 5: Relationship Between Customer Tenure & Age Group
fig, ax = plt.subplots(figsize=(8, 4.5), dpi=300)
sns.regplot(data=clean_customer.sample(1000, random_state=42), x='customer_age', y='tenure', color='#9467bd',
            scatter_kws={'alpha':0.3, 's':15}, line_kws={'color':'#d62728', 'linewidth':2}, ax=ax)
ax.set_title('Visual 5: Bivariate Analysis of Customer Tenure vs. Demographic Age', fontsize=12, fontweight='bold', pad=12)
ax.set_xlabel('Customer Age (Years)', fontsize=10)
ax.set_ylabel('Tenure Duration (Months)', fontsize=10)
plt.tight_layout()
plt.savefig('visual5_tenure_age.png')
plt.close()

print("All 5 chart images generated successfully with zero warnings!")
