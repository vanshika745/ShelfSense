# Dataset folder

ShelfSense works with **any** shop sales file you upload in the app. For the built-in demo it uses:

**Online Retail II**, UCI Machine Learning Repository (licence: CC BY 4.0)
https://archive.ics.uci.edu/dataset/502/online+retail+ii

- `online_retail_II.parquet` (included, 6 MB) is a compact copy of the full dataset: the same 1,067,371 rows and 8 columns, stored as text exactly as in the original file. It is redistributed under CC BY 4.0 with attribution below.
- The original Excel file can be used instead: download `online+retail+ii.zip` from the page above, unzip it and place `online_retail_II.xlsx` in this folder. The first time, ShelfSense converts it to `online_retail_II.csv` (about 1–2 minutes). The Excel and CSV files are too large for GitHub and are not included.

Citation: Chen, D. (2012). Online Retail II [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5CG6D
