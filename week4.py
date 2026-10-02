import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# 1. Load Datasets
baskets = pd.read_csv(r"C:\Users\Viswa\OneDrive\Documents\ECommerceAnalysis\basket_details.csv")
customers = pd.read_csv(r"C:\Users\Viswa\OneDrive\Documents\ECommerceAnalysis\customer_details.csv")

# Clean column headers
baskets.columns = baskets.columns.str.strip().str.lower()
customers.columns = customers.columns.str.strip().str.lower()

# Debug print to verify column names
print("Baskets columns:", baskets.columns.tolist())
print("Customers columns:", customers.columns.tolist())

# Dynamic Gender/Sex Column Identification
gender_candidates = [col for col in customers.columns if 'gender' in col or 'sex' in col]

if gender_candidates:
    gender_col = gender_candidates[0]
    print(f"Detected gender column: '{gender_col}' -> Standardizing to 'customer_gender'")
    customers.rename(columns={gender_col: 'customer_gender'}, inplace=True)
    customers['customer_gender'] = customers['customer_gender'].replace(['Unknown', 'unknown', 'U'], np.nan)
else:
    print("Warning: No gender/sex column found in customers dataset. Creating dummy 'customer_gender' column.")
    customers['customer_gender'] = np.nan

# 2. Preprocessing & Data Cleaning
if 'customer_age' in customers.columns:
    customers['customer_age'] = customers['customer_age'].apply(
        lambda x: np.nan if pd.isna(x) or x < 0 or x > 100 else x
    )

# Merge datasets
df = pd.merge(baskets, customers, on='customer_id', how='inner')

# 3. Target Definition & Feature Engineering
df['target'] = (df['basket_count'] > 2).astype(int)
df['basket_date'] = pd.to_datetime(df['basket_date'])
df['basket_month'] = df['basket_date'].dt.month
df['basket_dayofweek'] = df['basket_date'].dt.dayofweek

# Feature Selection
feature_cols = ['customer_age', 'customer_gender', 'basket_month', 'basket_dayofweek']
X = df[feature_cols]
y = df['target']

# 4. Pipeline Setup
num_cols = ['customer_age', 'basket_month', 'basket_dayofweek']
cat_cols = ['customer_gender']

num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

cat_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer([
    ('num', num_pipeline, num_cols),
    ('cat', cat_pipeline, cat_cols)
])

# Stratified Train-Test Split (75/25)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# 5. Model Pipelines
lr_model = Pipeline([('prep', preprocessor), ('clf', LogisticRegression(random_state=42))])
rf_model = Pipeline([('prep', preprocessor), ('clf', RandomForestClassifier(random_state=42))])

# Train Models
lr_model.fit(X_train, y_train)
rf_model.fit(X_train, y_train)

# Predictions & Probabilities
y_pred_lr = lr_model.predict(X_test)
y_prob_lr = lr_model.predict_proba(X_test)[:, 1]

y_pred_rf = rf_model.predict(X_test)
y_prob_rf = rf_model.predict_proba(X_test)[:, 1]

print("\nModel training executed successfully!")
# Merge datasets on customer_id
df = pd.merge(baskets, customers, on='customer_id', how='inner')

# Verify merged columns
print("Merged DataFrame Columns:", df.columns.tolist())

# 3. Target Definition & Feature Engineering
df['target'] = (df['basket_count'] > 2).astype(int)
df['basket_date'] = pd.to_datetime(df['basket_date'])
df['basket_month'] = df['basket_date'].dt.month
df['basket_dayofweek'] = df['basket_date'].dt.dayofweek

# Feature Selection
feature_cols = ['customer_age', 'customer_gender', 'basket_month', 'basket_dayofweek']

# Verify required columns exist in merged dataframe
missing_features = [col for col in feature_cols if col not in df.columns]
if missing_features:
    raise KeyError(f"The following required features are missing from the merged DataFrame: {missing_features}")

X = df[feature_cols]
y = df['target']

# 4. Pipeline Setup
num_cols = ['customer_age', 'basket_month', 'basket_dayofweek']
cat_cols = ['customer_gender']

num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

cat_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer([
    ('num', num_pipeline, num_cols),
    ('cat', cat_pipeline, cat_cols)
])

# Stratified Train-Test Split (75/25)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# 5. Model Pipelines
lr_model = Pipeline([('prep', preprocessor), ('clf', LogisticRegression(random_state=42))])
rf_model = Pipeline([('prep', preprocessor), ('clf', RandomForestClassifier(random_state=42))])

# Train Models
lr_model.fit(X_train, y_train)
rf_model.fit(X_train, y_train)

# Predictions & Probabilities
y_pred_lr = lr_model.predict(X_test)
y_prob_lr = lr_model.predict_proba(X_test)[:, 1]

y_pred_rf = rf_model.predict(X_test)
y_prob_rf = rf_model.predict_proba(X_test)[:, 1]

print("Script executed successfully without KeyError.")
