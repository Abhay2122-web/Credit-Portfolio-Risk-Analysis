# CSV Update Log — Credit Portfolio Segmentation

## Purpose
This file documents what the dashboard prepares or changes when it reads `dataset.csv`, and how those changes are made. It is the fifth project file.

## Important distinction
`dataset.csv` is kept as the original input file. The current `app.py` does **not** overwrite or save changes back into `dataset.csv`. Its data-cleaning steps are applied in memory each time you run the dashboard. Therefore, the original CSV remains unchanged.

## Data preparation performed by `app.py`

| Step | What is prepared | How it is done | Is the original CSV overwritten? |
|---|---|---|---|
| 1. Read data | Loads the rows and columns from `dataset.csv` | `pandas.read_csv(...)` | No |
| 2. Standardize column names | Recognizes common alternative names for fields such as income, loan amount, credit score, state, issue date and default | Maps recognized aliases to standard internal names such as `revenue`, `loan_amnt`, `fico_n`, `addr_state`, `issue_d` and `Default` | No |
| 3. Remove duplicate rows | Removes rows that are exact duplicates across the loaded columns | `drop_duplicates()` | No |
| 4. Check required fields | Checks that `revenue`, `dti_n`, `loan_amnt` and `fico_n` exist | Stops with an error message if required columns are missing | No |
| 5. Limit analysis size | If there are more rows than the sidebar limit, selects a reproducible sample | `sample(..., random_state=42)` | No |
| 6. Derive issue year | Creates `issue_year` from `issue_d` when the date column is present | `pandas.to_datetime(..., errors="coerce").dt.year` | No |
| 7. Convert numeric fields | Converts selected numeric columns to numeric values; unparseable values become missing | `pandas.to_numeric(..., errors="coerce")` | No |
| 8. Cap extreme numeric values | Caps selected numeric fields at their 1st and 99th percentiles | Calculates the 1% and 99% quantiles and applies `clip()` | No |
| 9. Fill categorical gaps | Replaces missing or blank values in selected categorical columns with `Unknown` | `fillna("Unknown")`, string conversion and blank replacement | No |
| 10. Prepare model features | Fills missing numeric values with medians, scales numeric features, and one-hot encodes categorical features | Scikit-learn `SimpleImputer`, `StandardScaler` and `OneHotEncoder` pipeline | No |
| 11. Create analysis output | Adds `Cluster` and `Default_numeric` columns to the in-memory analysis table | K-Means cluster labels and numeric conversion of `Default` | No |
| 12. Export results | Lets the user download analysed borrower rows and cluster profiles from the dashboard | Streamlit download buttons | Creates downloadable output files only when selected; does not overwrite `dataset.csv` |

## How to demonstrate this in a viva

1. Open `dataset.csv` and show the original column names and sample records.
2. Open `app.py` and point to the data-loading line: `pd.read_csv(data_source, low_memory=False)`.
3. Show the `aliases` dictionary and the `rename` / `drop_duplicates()` steps.
4. Explain numeric conversion and outlier capping using the 1st and 99th percentiles.
5. Show the imputation, scaling and categorical encoding pipeline.
6. Run the dashboard and explain that the cluster and analysis columns are added to the in-memory DataFrame.
7. Use the dashboard's download buttons to export the analysed borrower data or cluster profiles if asked to show the output.

## What is not claimed
This log describes the operations implemented in the current dashboard; it does not claim that the original CSV has already been permanently edited. To make a permanently cleaned CSV, a separate export step would need to be added and explicitly run.
