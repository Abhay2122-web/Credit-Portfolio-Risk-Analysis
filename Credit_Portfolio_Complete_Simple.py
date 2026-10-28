"""
CREDIT PORTFOLIO PROJECT — COMPLETE SIMPLE CSV WORKFLOW
=======================================================

Run:
    python -m pip install pandas scikit-learn
    python Credit_Portfolio_Complete_Simple.py

Keep your credit portfolio CSV in the same folder and name it:
    dataset.csv

Creates one output CSV:
    credit_portfolio_prepared_with_clusters.csv

The original dataset.csv is never overwritten.

This script demonstrates:
1. Load data
2. Standardize common column names
3. Remove duplicate rows
4. Convert numeric columns
5. Create issue_year when a date is available
6. Handle missing numeric and categorical values
7. Limit extreme values using the 1st and 99th percentiles
8. Scale numeric features and encode categories for clustering
9. Create K-Means borrower segments
10. Save all original/prepared columns plus Cluster in one CSV

Note:
The clustering is an analytical example. Confirm the columns and cluster
meaning with your actual dashboard before presenting results as final findings.
"""

from pathlib import Path
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.cluster import KMeans


# FILE SETTINGS
FOLDER = Path(__file__).resolve().parent
INPUT_FILE = FOLDER / "dataset.csv"
OUTPUT_FILE = FOLDER / "credit_portfolio_prepared_with_clusters.csv"
NUMBER_OF_CLUSTERS = 4


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Cannot find {INPUT_FILE.name}. Put your credit portfolio CSV "
            "in this folder and rename it dataset.csv, or change INPUT_FILE."
        )

    # STEP 1: Load the original CSV.
    df = pd.read_csv(INPUT_FILE, low_memory=False)
    print("\nSTEP 1 — LOAD DATA")
    print("Original rows:", f"{len(df):,}")
    print("Original columns:", len(df.columns))

    # STEP 2: Standardize common column-name variations.
    aliases = {
        "revenue": ["annual_income", "income"],
        "dti_n": ["dti", "debt_to_income", "debt-to-income ratio"],
        "loan_amnt": ["loan_amount", "loan amount"],
        "fico_n": ["fico", "credit_score", "credit score"],
        "experience_c": ["experience", "emp_length_years"],
        "emp_length": ["employment_length", "employment length"],
        "purpose": ["loan_purpose", "loan purpose"],
        "home_ownership_n": ["home_ownership", "home ownership"],
        "addr_state": ["state", "geography", "region"],
        "issue_d": ["issue_date", "loan_issue_date", "loan issue date"],
        "Default": ["default", "default_status", "default status"],
    }
    lookup = {str(c).strip().lower(): c for c in df.columns}
    rename = {}
    for standard, alternatives in aliases.items():
        if standard.lower() in lookup:
            continue
        for name in alternatives:
            if name.lower() in lookup:
                rename[lookup[name.lower()]] = standard
                break
    df = df.rename(columns=rename)
    print("\nSTEP 2 — STANDARDIZE COLUMN NAMES")
    print("Renamed columns:", rename if rename else "No renaming needed")

    # STEP 3: Remove exact duplicate rows.
    duplicate_count = int(df.duplicated().sum())
    df = df.drop_duplicates().copy()
    print("\nSTEP 3 — REMOVE DUPLICATES")
    print("Duplicate rows removed:", f"{duplicate_count:,}")
    print("Rows remaining:", f"{len(df):,}")

    # STEP 4: Check the main columns required by the credit dashboard.
    required = ["revenue", "dti_n", "loan_amnt", "fico_n"]
    missing = [c for c in required if c not in df.columns]
    print("\nSTEP 4 — CHECK REQUIRED COLUMNS")
    print("Required columns:", required)
    if missing:
        raise ValueError(
            "These required columns are missing: " + ", ".join(missing)
            + ". Check your dataset or update the column aliases in this script."
        )
    print("All required columns are present.")

    # STEP 5: Create issue_year if a loan issue date is available.
    if "issue_d" in df.columns:
        df["issue_year"] = pd.to_datetime(df["issue_d"], errors="coerce").dt.year
        print("\nSTEP 5 — CREATE ISSUE YEAR")
        print("Rows with a valid issue year:", f"{int(df['issue_year'].notna().sum()):,}")
    else:
        print("\nSTEP 5 — CREATE ISSUE YEAR")
        print("No issue date column found; no issue_year was created.")

    # STEP 6: Convert important numeric fields.
    numeric_candidates = [
        "revenue", "dti_n", "loan_amnt", "fico_n", "experience_c", "issue_year"
    ]
    numeric_cols = []
    for col in numeric_candidates:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            if df[col].notna().any():
                numeric_cols.append(col)
    print("\nSTEP 6 — CONVERT NUMERIC FIELDS")
    print("Numeric columns:", numeric_cols)

    # STEP 7: Cap extreme values instead of deleting those rows.
    capped = {}
    for col in numeric_cols:
        valid = df[col].dropna()
        if len(valid) > 1:
            low, high = valid.quantile([0.01, 0.99])
            capped[col] = int(
                (df[col].notna() & ((df[col] < low) | (df[col] > high))).sum()
            )
            df[col] = df[col].clip(lower=low, upper=high)
    print("\nSTEP 7 — HANDLE EXTREME VALUES")
    print("Values capped by column:", capped)

    # STEP 8: Prepare available category columns.
    categorical_candidates = [
        "emp_length", "purpose", "home_ownership_n", "addr_state"
    ]
    categorical_cols = [c for c in categorical_candidates if c in df.columns]
    for col in categorical_cols:
        df[col] = df[col].fillna("Unknown").astype(str).str.strip()
        df[col] = df[col].replace("", "Unknown")
    print("\nSTEP 8 — HANDLE MISSING CATEGORIES")
    print("Categorical columns:", categorical_cols)
    print("Missing categories are labelled 'Unknown'.")

    # STEP 9: Fill missing numeric values with each column's median.
    for col in numeric_cols:
        median = df[col].median()
        if pd.notna(median):
            df[col] = df[col].fillna(median)
    print("\nSTEP 9 — FILL MISSING NUMERIC VALUES")
    print("Missing numeric values filled using the median.")

    # STEP 10: Prepare the model inputs: scale numbers and encode categories.
    model_numeric = [
        c for c in ["revenue", "dti_n", "loan_amnt", "fico_n", "experience_c", "issue_year"]
        if c in df.columns and df[c].notna().any()
    ]
    model_categorical = [c for c in categorical_cols if df[c].notna().any()]

    transformers = []
    if model_numeric:
        numeric_pipeline = Pipeline([
            ("fill_missing", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ])
        transformers.append(("numbers", numeric_pipeline, model_numeric))

    if model_categorical:
        category_pipeline = Pipeline([
            ("fill_missing", SimpleImputer(strategy="most_frequent")),
            ("encode", OneHotEncoder(handle_unknown="ignore")),
        ])
        transformers.append(("categories", category_pipeline, model_categorical))

    if not transformers:
        raise ValueError("No usable numeric or categorical features were found.")

    prep = ColumnTransformer(transformers=transformers)
    model_data = prep.fit_transform(df)

    print("\nSTEP 10 — PREPARE DATA FOR CLUSTERING")
    print("Numeric features scaled:", model_numeric)
    print("Categorical features encoded:", model_categorical)

    # STEP 11: Create borrower segments with K-Means.
    if len(df) < NUMBER_OF_CLUSTERS:
        raise ValueError(
            f"At least {NUMBER_OF_CLUSTERS} rows are required for clustering."
        )
    model = KMeans(n_clusters=NUMBER_OF_CLUSTERS, random_state=42, n_init=10)
    df["Cluster"] = model.fit_predict(model_data)
    print("\nSTEP 11 — CREATE BORROWER SEGMENTS")
    print("Method: K-Means clustering")
    print("Number of clusters:", NUMBER_OF_CLUSTERS)
    print("Rows in each cluster:")
    print(df["Cluster"].value_counts().sort_index().to_string())

    # STEP 12: Save one complete CSV; original source remains unchanged.
    df.to_csv(OUTPUT_FILE, index=False)
    print("\nSTEP 12 — SAVE FINAL CSV")
    print("Output file:", OUTPUT_FILE.name)
    print("Final rows:", f"{len(df):,}")
    print("Final columns:", len(df.columns))
    print("Duplicate rows remaining:", f"{int(df.duplicated().sum()):,}")
    print("Missing cells remaining:", f"{int(df.isna().sum().sum()):,}")
    print("Original dataset.csv was not changed.")

    print("\nSIMPLE EXPLANATION FOR FACULTY")
    print("I loaded the credit portfolio CSV, standardized column names, and removed duplicates.")
    print("I converted key financial fields to numeric values and created issue_year when possible.")
    print("I handled missing values, capped extreme numeric values, and filled missing categories.")
    print("I scaled numeric features and encoded categories so K-Means could analyse them.")
    print("Finally, I added a Cluster column and saved the complete result to a new CSV.")
    print("Cluster numbers are group labels; they do not automatically mean low or high risk.")


if __name__ == "__main__":
    main()
