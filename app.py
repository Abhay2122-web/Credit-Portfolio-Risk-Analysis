"""Credit Portfolio Segmentation and Risk Pattern Analysis.
Run: python -m streamlit run app.py
Keep the assignment CSV as dataset.csv in this folder; this avoids repeated uploads.
"""
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

st.set_page_config(page_title="Credit Portfolio Risk Dashboard", page_icon="📊", layout="wide")
st.title("Credit Portfolio Segmentation and Risk Pattern Analysis")
st.caption("Borrower segmentation • Historical risk patterns • Exploratory analysis")

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "dataset.csv"

st.sidebar.header("Dataset")
if DATA_PATH.exists():
    st.sidebar.success("dataset.csv found — no upload needed")
    uploaded = None
else:
    st.sidebar.warning("Place your dataset in this folder and name it dataset.csv.")
    uploaded = st.sidebar.file_uploader("Temporary upload if needed", type=["csv"])

max_rows = st.sidebar.slider("Maximum rows to analyse", 1000, 30000, 10000, 1000)
run = st.sidebar.button("Run analysis", type="primary")

if DATA_PATH.exists():
    data_source = DATA_PATH
elif uploaded is not None:
    data_source = uploaded
else:
    st.info("Save your assignment CSV beside app.py as `dataset.csv`. Then refresh this page.")
    st.stop()

if not run:
    st.info("Dataset is ready. Click **Run analysis** in the sidebar.")
    st.stop()

try:
    raw = pd.read_csv(data_source, low_memory=False)
except Exception as e:
    st.error(f"Could not read dataset: {e}")
    st.stop()

# Normalize known column names used by loan datasets.
aliases = {
    "revenue": ["revenue", "annual_income", "annual income", "income"],
    "dti_n": ["dti_n", "dti", "debt_to_income", "debt-to-income ratio"],
    "loan_amnt": ["loan_amnt", "loan_amount", "loan amount"],
    "fico_n": ["fico_n", "fico", "credit_score", "credit score"],
    "experience_c": ["experience_c", "experience", "employment_length", "emp_length_years"],
    "emp_length": ["emp_length", "employment_length", "employment length"],
    "purpose": ["purpose", "loan_purpose", "loan purpose"],
    "home_ownership_n": ["home_ownership_n", "home_ownership", "home ownership"],
    "addr_state": ["addr_state", "state", "geography", "region"],
    "issue_d": ["issue_d", "issue_date", "loan_issue_date", "loan issue date"],
    "Default": ["Default", "default", "default_status", "default status"]
}
lower = {str(c).strip().lower(): c for c in raw.columns}
rename = {}
for target, names in aliases.items():
    for name in names:
        if name.lower() in lower:
            rename[lower[name.lower()]] = target
            break
df = raw.rename(columns=rename).drop_duplicates().copy()

required = ["revenue", "dti_n", "loan_amnt", "fico_n"]
missing = [c for c in required if c not in df.columns]
if missing:
    st.error("Required columns missing: " + ", ".join(missing))
    st.write("Available columns:", list(raw.columns))
    st.stop()
if "Default" not in df:
    df["Default"] = np.nan

if len(df) > max_rows:
    df = df.sample(max_rows, random_state=42)
df.reset_index(drop=True, inplace=True)

if "issue_d" in df:
    df["issue_year"] = pd.to_datetime(df["issue_d"], errors="coerce").dt.year
else:
    df["issue_year"] = np.nan

numeric_candidates = ["revenue", "dti_n", "loan_amnt", "fico_n", "experience_c", "issue_year"]
categorical_candidates = ["emp_length", "purpose", "home_ownership_n", "addr_state"]
num_cols = [c for c in numeric_candidates if c in df and pd.to_numeric(df[c], errors="coerce").notna().any()]
cat_cols = [c for c in categorical_candidates if c in df]
for c in num_cols:
    df[c] = pd.to_numeric(df[c], errors="coerce")
    valid = df[c].dropna()
    if not valid.empty:
        low_q, high_q = valid.quantile([0.01, 0.99])
        df[c] = df[c].clip(low_q, high_q)
for c in cat_cols:
    df[c] = df[c].fillna("Unknown").astype(str).str.strip().replace("", "Unknown")

numeric_pipe = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
categorical_pipe = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                             ("encoder", OneHotEncoder(handle_unknown="ignore", min_frequency=10))])
transformers = []
if num_cols:
    transformers.append(("numeric", numeric_pipe, num_cols))
if cat_cols:
    transformers.append(("categorical", categorical_pipe, cat_cols))
X = ColumnTransformer(transformers, remainder="drop").fit_transform(df)
X = X.toarray().astype("float32") if hasattr(X, "toarray") else np.asarray(X, dtype="float32")

with st.spinner("Cleaning, clustering, and generating charts..."):
    labels = KMeans(n_clusters=4, random_state=42, n_init=10).fit_predict(X)
    df["Cluster"] = labels
    sil = silhouette_score(X, labels, sample_size=min(2000, len(df)), random_state=42)
    coords = PCA(n_components=2, random_state=42).fit_transform(X)
    profile = df.groupby("Cluster").agg(
        Borrowers=("Cluster", "size"),
        Average_Annual_Income=("revenue", "mean"),
        Average_DTI=("dti_n", "mean"),
        Average_Loan_Amount=("loan_amnt", "mean"),
        Average_FICO=("fico_n", "mean")).reset_index()
    df["Default_numeric"] = pd.to_numeric(df["Default"], errors="coerce")
    if df["Default_numeric"].notna().any():
        rates = df.groupby("Cluster")["Default_numeric"].mean().rename("Historical_Default_Rate")
        profile = profile.merge(rates, on="Cluster", how="left")
        profile["Historical_Default_Rate_Pct"] = profile["Historical_Default_Rate"] * 100

st.success(f"Analysis completed for {len(df):,} borrowers.")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Borrowers analysed", f"{len(df):,}")
c2.metric("Average annual income", f"${df.revenue.mean():,.0f}")
c3.metric("Average loan amount", f"${df.loan_amnt.mean():,.0f}")
c4.metric("K-Means silhouette", f"{sil:.3f}")

st.subheader("Cluster profiles")
st.dataframe(profile, use_container_width=True)
left, right = st.columns(2)
with left:
    st.subheader("PCA cluster visualization")
    fig, ax = plt.subplots(figsize=(7, 4))
    for cluster in sorted(np.unique(labels)):
        mask = labels == cluster
        ax.scatter(coords[mask, 0], coords[mask, 1], s=8, alpha=.5, label=f"Cluster {cluster}")
    ax.set_xlabel("Principal Component 1")
    ax.set_ylabel("Principal Component 2")
    ax.legend()
    st.pyplot(fig)
    plt.close(fig)
with right:
    st.subheader("Historical default rate by cluster")
    if "Historical_Default_Rate_Pct" in profile:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.bar(profile.Cluster.astype(str), profile.Historical_Default_Rate_Pct)
        ax.set_xlabel("Cluster")
        ax.set_ylabel("Observed default rate (%)")
        st.pyplot(fig)
        plt.close(fig)
    else:
        st.info("No usable Default column; default-rate chart unavailable.")

st.subheader("EDA distributions")
columns = st.columns(3)
for holder, field, title in zip(columns, ["fico_n", "loan_amnt", "revenue"],
                                ["FICO score", "Loan amount", "Annual income"]):
    with holder:
        fig, ax = plt.subplots(figsize=(5, 3))
        df[field].dropna().hist(bins=30, ax=ax)
        ax.set_title(title)
        st.pyplot(fig)
        plt.close(fig)

st.subheader("Clustering algorithm comparison")
n = min(2000, len(df))
indices = np.random.RandomState(42).choice(len(df), n, replace=False)
Xs = X[indices]
hierarchical = AgglomerativeClustering(n_clusters=4, linkage="ward").fit_predict(Xs)
reduced = PCA(n_components=min(10, Xs.shape[1]), random_state=42).fit_transform(Xs)
dbscan = DBSCAN(eps=2.5, min_samples=10).fit_predict(reduced)

def cluster_silhouette(matrix, cluster_labels):
    cluster_labels = np.asarray(cluster_labels)
    valid = cluster_labels != -1
    if len(set(cluster_labels[valid])) < 2 or valid.sum() < 3:
        return np.nan
    try:
        return silhouette_score(matrix[valid], cluster_labels[valid],
                                sample_size=min(1000, int(valid.sum())), random_state=42)
    except Exception:
        return np.nan

comparison = pd.DataFrame([
    {"Algorithm": "K-Means", "Rows evaluated": len(df), "Clusters excluding noise": len(set(labels)),
     "Noise points": 0, "Silhouette score": sil},
    {"Algorithm": "Hierarchical (Ward)", "Rows evaluated": n, "Clusters excluding noise": len(set(hierarchical)),
     "Noise points": 0, "Silhouette score": cluster_silhouette(Xs, hierarchical)},
    {"Algorithm": "DBSCAN", "Rows evaluated": n, "Clusters excluding noise": len(set(dbscan) - {-1}),
     "Noise points": int((dbscan == -1).sum()), "Silhouette score": cluster_silhouette(reduced, dbscan)}
])
st.dataframe(comparison, use_container_width=True)

if "Historical_Default_Rate_Pct" in profile and profile.Historical_Default_Rate_Pct.notna().any():
    low = profile.loc[profile.Historical_Default_Rate_Pct.idxmin()]
    high = profile.loc[profile.Historical_Default_Rate_Pct.idxmax()]
    st.subheader("Business observations")
    st.write(f"Lowest observed default rate: Cluster {int(low.Cluster)} ({low.Historical_Default_Rate_Pct:.2f}%).")
    st.write(f"Highest observed default rate: Cluster {int(high.Cluster)} ({high.Historical_Default_Rate_Pct:.2f}%).")
    st.caption("Historical group-level patterns are exploratory and do not prove causation or justify individual loan decisions.")

st.subheader("Borrower records (first 300 rows)")
st.dataframe(df.head(300), use_container_width=True)
st.download_button("Download analysed borrower data", df.to_csv(index=False).encode("utf-8"),
                   file_name="borrowers_with_clusters.csv", mime="text/csv")
st.download_button("Download cluster profiles", profile.to_csv(index=False).encode("utf-8"),
                   file_name="cluster_profiles.csv", mime="text/csv")
st.caption("Default is excluded from clustering features and used only after clustering for historical outcome comparison.")
