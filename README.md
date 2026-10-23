# Credit Portfolio Segmentation and Risk Pattern Analysis

## Files (4 total)
- `app.py` — Streamlit dashboard and analysis.
- `requirements.txt` — Python dependencies.
- `README.md` — setup instructions.
- `dataset.csv` — the assignment's loan dataset; keep it beside `app.py`.

## One-time setup
Open a terminal in this folder and run:

```powershell
python -m pip install -r requirements.txt
```

## Run during viva
```powershell
python -m streamlit run app.py
```

The app automatically reads `dataset.csv`; no browser upload is needed when the file is present.

## Dataset preparation
Rename the assignment file `loans_granting_model_dataset.csv` to `dataset.csv`, then put it in the same folder as `app.py`.

## GitHub
The dataset provided for this assignment is about 160 MB. A regular GitHub file must be under 100 MiB, so do not try to commit it as a normal file. Keep the four files locally for the viva, but for GitHub upload `app.py`, `requirements.txt`, and `README.md`; store `dataset.csv` using Git LFS only if permitted by the dataset licence and your GitHub plan, or document how an evaluator can obtain it. If the dataset licence does not permit redistribution, do not upload it.

## Methodology
- Remove duplicate records, cap numeric extremes at the 1st and 99th percentiles, impute missing values, encode categorical columns, and scale numeric columns.
- K-Means creates four borrower clusters.
- Hierarchical Clustering and DBSCAN are compared on a reproducible subset for runtime.
- PCA projects features into two dimensions for visualization.
- `Default` is excluded from clustering inputs and used only to compare historical default rates after clustering.

## Limitations
Cluster IDs are arbitrary. Silhouette scores can be low; DBSCAN may not discover useful clusters with the initial settings. Results are exploratory, not a credit approval model. t-SNE and UMAP are not included in this minimal quick-run version and should be added if explicitly required by the evaluator.
