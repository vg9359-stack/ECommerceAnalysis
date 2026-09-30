import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# Set random seed for reproducibility
SEED = 42

# ==========================================
# 1. DATA LOADING & MERGING
# ==========================================
print("--- Step 1: Loading and Merging Datasets ---")
# Adjust filenames as needed
basket_df = pd.read_csv(r"C:\Users\Viswa\Documents\ECommerceAnalysis\basket_details.csv")
customer_df = pd.read_csv(r"C:\Users\Viswa\Documents\ECommerceAnalysis\customer_details.csv")

# Resolve common column naming variations for merging
if "customer_id" in basket_df.columns and "customer_id" in customer_df.columns:
    merged_df = pd.merge(basket_df, customer_df, on="customer_id", how="inner")
elif "Customer_ID" in basket_df.columns and "Customer_ID" in customer_df.columns:
    merged_df = pd.merge(basket_df, customer_df, on="Customer_ID", how="inner")
else:
    # Attempt merge on the first common column or fallback to index matching if needed
    common_cols = list(set(basket_df.columns).intersection(set(customer_df.columns)))
    if common_cols:
        merged_df = pd.merge(basket_df, customer_df, on=common_cols[0], how="inner")
    else:
        merged_df = pd.concat([basket_df, customer_df], axis=1)

print(f"Merged Dataset Shape: {merged_df.shape}")

# ==========================================
# 2. FEATURE ENGINEERING & TARGET CREATION
# ==========================================
print("\n--- Step 2: Feature Engineering & Target Definition ---")

# Define target variable 'is_high_value' based on total spend or order total
# Look for relevant numeric target columns (e.g., basket_value, total_amount, price)
possible_value_cols = [col for col in merged_df.columns if any(k in col.lower() for k in ["val", "price", "amount", "total", "spend"])]

if possible_value_cols:
    target_base_col = possible_value_cols[0]
    median_val = merged_df[target_base_col].median()
    merged_df["target"] = (merged_df[target_base_col] > median_val).astype(int)
    print(f"Target 'target' generated based on median threshold of '{target_base_col}' ({median_val}).")
elif "target" in merged_df.columns:
    merged_df["target"] = merged_df["target"].astype(int)
else:
    # Synthesize target if specific column isn't found
    np.random.seed(SEED)
    merged_df["target"] = np.random.choice([0, 1], size=len(merged_df), p=[0.6, 0.4])
    print("Target generated using default binary distribution.")

# Separate Features (X) and Target (y)
drop_cols = ["target", "customer_id", "Customer_ID", "basket_id", "Basket_ID", "order_id", "Order_ID"]
drop_cols = [col for col in drop_cols if col in merged_df.columns]

X = merged_df.drop(columns=drop_cols)
y = merged_df["target"]

# Identify numeric and categorical columns
num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
cat_cols = X.select_dtypes(include=["object", "string", "category", "bool"]).columns.tolist()

print(f"Numeric Features ({len(num_cols)}): {num_cols}")
print(f"Categorical Features ({len(cat_cols)}): {cat_cols}")

# ==========================================
# 3. PREPROCESSING PIPELINE & TRAIN-TEST SPLIT
# ==========================================
print("\n--- Step 3: Train-Test Split and Preprocessing Pipeline ---")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=SEED, stratify=y
)

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])

preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_transformer, num_cols),
    ("cat", categorical_transformer, cat_cols)
])

# ==========================================
# 4. MODEL SELECTION & TRAINING
# ==========================================
print("\n--- Step 4: Training Models ---")

# Baseline Model: Logistic Regression
lr_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(random_state=SEED, max_iter=1000))
])

# Advanced Model: Random Forest Classifier
rf_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(n_estimators=100, random_state=SEED))
])

# Train Logistic Regression
lr_pipeline.fit(X_train, y_train)
lr_preds = lr_pipeline.predict(X_test)
lr_probs = lr_pipeline.predict_proba(X_test)[:, 1]

# Train Random Forest
rf_pipeline.fit(X_train, y_train)
rf_preds = rf_pipeline.predict(X_test)
rf_probs = rf_pipeline.predict_proba(X_test)[:, 1]

# Print Performance Metrics
print("\n--- Model Performance Summary ---")
for name, preds, probs in [("Logistic Regression", lr_preds, lr_probs), ("Random Forest", rf_preds, rf_probs)]:
    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds)
    rec = recall_score(y_test, preds)
    f1 = f1_score(y_test, preds)
    print(f"\n{name}:")
    print(f"  Accuracy  : {acc:.4f}")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(f"  F1-Score  : {f1:.4f}")

# ==========================================
# 5. VISUALIZATIONS
# ==========================================
print("\n--- Step 5: Generating Visualization Artifacts ---")

plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

# Visual 1: Confusion Matrices Comparison
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

cm_lr = confusion_matrix(y_test, lr_preds)
sns.heatmap(cm_lr, annot=True, fmt="d", cmap="Blues", ax=axes[0], cbar=False)
axes[0].set_title("Confusion Matrix - Logistic Regression", fontsize=12, fontweight="bold")
axes[0].set_xlabel("Predicted Label")
axes[0].set_ylabel("True Label")

cm_rf = confusion_matrix(y_test, rf_preds)
sns.heatmap(cm_rf, annot=True, fmt="d", cmap="Greens", ax=axes[1], cbar=False)
axes[1].set_title("Confusion Matrix - Random Forest", fontsize=12, fontweight="bold")
axes[1].set_xlabel("Predicted Label")
axes[1].set_ylabel("True Label")

plt.tight_layout()
plt.savefig("confusion_matrices_comparison.png", dpi=300)
plt.show()

# Visual 2: ROC Curves Comparison
fpr_lr, tpr_lr, _ = roc_curve(y_test, lr_probs)
roc_auc_lr = auc(fpr_lr, tpr_lr)

fpr_rf, tpr_rf, _ = roc_curve(y_test, rf_probs)
roc_auc_rf = auc(fpr_rf, tpr_rf)

plt.figure(figsize=(8, 6))
plt.plot(fpr_lr, tpr_lr, color="blue", lw=2, label=f"Logistic Regression (AUC = {roc_auc_lr:.3f})")
plt.plot(fpr_rf, tpr_rf, color="green", lw=2, label=f"Random Forest (AUC = {roc_auc_rf:.3f})")
plt.plot([0, 1], [0, 1], color="gray", lw=1.5, linestyle="--")

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=11)
plt.ylabel("True Positive Rate (Sensitivity)", fontsize=11)
plt.title("Receiver Operating Characteristic (ROC) Curve Comparison", fontsize=13, fontweight="bold")
plt.legend(loc="lower right", fontsize=11)
plt.tight_layout()
plt.savefig("roc_curve_comparison.png", dpi=300)
plt.show()

print("Visualizations successfully saved to disk.")
