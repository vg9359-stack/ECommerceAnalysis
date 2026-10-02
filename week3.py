import json
import zlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import seaborn as sns
# =========================================================
# 1. LOAD & CLEAN DATASETS
# =========================================================
customers = pd.read_csv(r"C:\Users\Viswa\OneDrive\Documents\ECommerceAnalysis\customer_details.csv")
baskets = pd.read_csv(r"C:\Users\Viswa\OneDrive\Documents\ECommerceAnalysis\basket_details.csv")

# Clean column headers (remove leading/trailing spaces and lowercase)
customers.columns = customers.columns.str.strip().str.lower()
baskets.columns = baskets.columns.str.strip().str.lower()

# Print column names for verification
print("Customers CSV Columns:", customers.columns.tolist())
print("Baskets CSV Columns:", baskets.columns.tolist())

# Detect gender/sex column name automatically
sex_col = None
for col in ["sex", "gender"]:
    if col in customers.columns:
        sex_col = col
        break

if sex_col is None:
    raise KeyError(
        f"Could not find a 'sex' or 'gender' column in customers dataset. Available columns: {list(customers.columns)}"
    )

# Demographic filtering (valid age range & recorded sex/gender)
clean_customers = customers[
    (customers[sex_col].astype(str).str.title().isin(["Male", "Female"]))
    & (customers["customer_age"] >= 18)
    & (customers["customer_age"] <= 100)
].copy()

# Standardize gender column values to Title Case
clean_customers[sex_col] = clean_customers[sex_col].astype(str).str.title()

# Categorize tenure into discrete Loyalty Tiers
clean_customers["loyalty_tier"] = pd.cut(
    clean_customers["tenure"],
    bins=[0, 30, 60, 150],
    labels=["Bronze (<30m)", "Silver (30-60m)", "Gold (>60m)"],
)

# Parse transaction dates
baskets["basket_date"] = pd.to_datetime(baskets["basket_date"])
baskets["day_of_week"] = baskets["basket_date"].dt.day_name()


# =========================================================
# 2. PERFORM STATISTICAL TESTS & VISUALIZATIONS
# =========================================================
sns.set_theme(style="whitegrid")
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# --- TEST 1: Two-Sample Welch's t-Test (Tenure across Gender) ---
male_tenure = clean_customers[clean_customers[sex_col] == "Male"]["tenure"]
female_tenure = clean_customers[clean_customers[sex_col] == "Female"]["tenure"]
t_stat, p_val_t = stats.ttest_ind(male_tenure, female_tenure, equal_var=False)

sns.boxplot(
    data=clean_customers,
    x=sex_col,
    y="tenure",
    ax=axes[0],
    hue=sex_col,
    palette=["#1f77b4", "#ff7f0e"],
)
axes[0].set_title(
    f"1. Customer Tenure by Gender\n(t={t_stat:.2f}, p={p_val_t:.2e})",
    fontsize=11,
    fontweight="bold",
)
axes[0].set_xlabel("Gender")
axes[0].set_ylabel("Tenure (Months)")

# --- TEST 2: Chi-Square Test of Independence (Gender vs Loyalty Tier) ---
contingency_tab = pd.crosstab(
    clean_customers[sex_col], clean_customers["loyalty_tier"]
)
chi2_stat, p_val_chi2, dof, _ = stats.chi2_contingency(contingency_tab)

sns.heatmap(
    contingency_tab, annot=True, fmt="d", cmap="Blues", ax=axes[1], cbar=False
)
axes[1].set_title(
    f"2. Gender vs. Loyalty Tier Contingency\n(Chi2={chi2_stat:.2f}, p={p_val_chi2:.2e})",
    fontsize=11,
    fontweight="bold",
)
axes[1].set_xlabel("Loyalty Tier")
axes[1].set_ylabel("Gender")

# --- TEST 3: One-Way ANOVA (Basket Count across Day of Week) ---
dow_order = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
]
dow_groups = [
    group["basket_count"].values
    for name, group in baskets.groupby("day_of_week")
]
f_stat, p_val_anova = stats.f_oneway(*dow_groups)

sns.barplot(
    data=baskets,
    x="day_of_week",
    y="basket_count",
    order=dow_order,
    ax=axes[2],
    hue="day_of_week",
    palette="viridis",
)
axes[2].set_title(
    f"3. Basket Size by Day of Week\n(F={f_stat:.2f}, p={p_val_anova:.2e})",
    fontsize=11,
    fontweight="bold",
)
axes[2].set_xlabel("Day of Week")
axes[2].set_ylabel("Mean Basket Count")
axes[2].tick_params(axis="x", rotation=45)

plt.tight_layout()
plt.savefig("week3_hypothesis_tests.png", dpi=300)
plt.close()
print("Saved plot to 'week3_hypothesis_tests.png'")


# =========================================================
# 3. USE BUILT-IN `zlib` FOR RESULTS COMPRESSION & OUTPUT
# =========================================================
test_summary = {
    "test_1_ttest": {
        "description": "Welch's Two-Sample t-Test (Male vs Female Tenure)",
        "male_mean_tenure": round(float(male_tenure.mean()), 2),
        "female_mean_tenure": round(float(female_tenure.mean()), 2),
        "t_statistic": round(float(t_stat), 4),
        "p_value": float(p_val_t),
    },
    "test_2_chisq": {
        "description": "Chi-Square Test of Independence (Gender vs Loyalty Tier)",
        "chi2_statistic": round(float(chi2_stat), 4),
        "degrees_of_freedom": int(dof),
        "p_value": float(p_val_chi2),
    },
    "test_3_anova": {
        "description": "One-Way ANOVA (Basket Size by Day of Week)",
        "f_statistic": round(float(f_stat), 4),
        "p_value": float(p_val_anova),
    },
}

json_bytes = json.dumps(test_summary, indent=2).encode("utf-8")
compressed_payload = zlib.compress(json_bytes, level=zlib.Z_BEST_COMPRESSION)
decompressed_payload = zlib.decompress(compressed_payload)
restored_summary = json.loads(decompressed_payload.decode("utf-8"))

print("\n--- STATISTICAL RESULTS ---")
print(json.dumps(restored_summary, indent=2))
