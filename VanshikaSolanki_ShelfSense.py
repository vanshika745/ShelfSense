"""
ShelfSense - Dead-Stock, Expiry & Reorder Analyzer for Small Shops
==================================================================

Author  : Vanshika Solanki
Program : IBM SkillsBuild Data Analytics with AI Internship 2026
          (BharatCares in association with AICTE)

What it does
    Upload a shop's sales file (CSV / Excel). ShelfSense maps the columns,
    checks and cleans the data, analyses sales (ABC-XYZ), uses machine learning
    to predict dead stock, forecast demand and segment products, then tells
    the shopkeeper what to reorder, what to discount and what to stop stocking.

Demo dataset
    UCI "Online Retail II" (CC BY 4.0)
    https://archive.ics.uci.edu/dataset/502/online+retail+ii
    Put online_retail_II.xlsx (or .csv) inside the data/ folder.

How to run
    Interactive web app :  streamlit run VanshikaSolanki_ShelfSense.py
    Command-line demo   :  python VanshikaSolanki_ShelfSense.py
                           (runs the full analysis on the demo dataset and
                            saves an HTML report + Excel workbook in outputs/)
"""

import argparse
import hashlib
import io
import json
import re
import sys
import time
from pathlib import Path
from statistics import NormalDist

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (f1_score, mean_absolute_error, precision_score,
                             recall_score, roc_auc_score, silhouette_score)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# ============================================================================
# 0. CONFIGURATION
# ============================================================================
APP_DIR = Path(__file__).resolve().parent
DATA_DIR = APP_DIR / "data"
OUTPUT_DIR = APP_DIR / "outputs"
DEMO_CSV = DATA_DIR / "online_retail_II.csv"
DEMO_XLSX = DATA_DIR / "online_retail_II.xlsx"
DATASET_URL = "https://archive.ics.uci.edu/dataset/502/online+retail+ii"
RANDOM_STATE = 42

# Column mapping: field -> (label shown to the user, required?, known column names)
FIELDS = {
    "date": ("Date of sale", True,
             ["invoicedate", "date", "invoice_date", "orderdate", "billdate", "saledate",
              "transactiondate", "datetime", "timestamp", "created"]),
    "product_code": ("Product code / SKU", False,
                     ["stockcode", "sku", "itemcode", "productcode", "productid", "itemid",
                      "barcode", "code", "article"]),
    "product_name": ("Product name", False,
                     ["description", "productname", "itemname", "product", "item", "name",
                      "itemdescription", "particulars"]),
    "quantity": ("Quantity sold", True, ["quantity", "qty", "units", "pcs", "unitssold", "quantitysold"]),
    "price": ("Unit selling price", False,
              ["price", "unitprice", "rate", "mrp", "sellingprice", "saleprice", "priceperunit"]),
    "amount": ("Line total (only if there is no unit price)", False,
               ["amount", "total", "linetotal", "netamount", "value", "sales", "revenue"]),
    "invoice": ("Invoice / bill number", False,
                ["invoice", "invoiceno", "invoicenumber", "billno", "billnumber", "bill", "orderid",
                 "orderno", "receiptno", "transactionid", "voucherno"]),
    "customer": ("Customer ID", False, ["customerid", "customer", "customerno", "clientid", "custid"]),
    "location": ("Country / branch / store", False, ["country", "branch", "store", "location", "city", "region", "outlet"]),
}
STOCK_FIELDS = {
    "product": ("Product code or name", True, ["stockcode", "sku", "productcode", "itemcode", "code",
                                               "product", "productname", "item", "description", "name"]),
    "stock": ("Current stock (units)", True, ["currentstock", "stock", "stockqty", "onhand", "closingstock",
                                              "qtyinstock", "quantity", "qty", "balance"]),
    "cost": ("Unit cost (purchase price)", False, ["unitcost", "cost", "costprice", "purchaseprice", "buyprice"]),
    "expiry": ("Expiry date", False, ["expirydate", "expiry", "exp", "bestbefore", "useby", "expdate"]),
}

# Lines that are not products (postage, fees, manual adjustments ...)
NON_PRODUCT_CODES = {"POST", "DOT", "M", "C2", "D", "S", "B", "BANK CHARGES", "ADJUST", "ADJUST2",
                     "AMAZONFEE", "CRUK", "TEST001", "TEST002", "PADS"}
NON_PRODUCT_WORDS = re.compile(
    r"\b(?:POSTAGE|CARRIAGE|MANUAL|BANK CHARGES|ADJUST(?:MENT)?|AMAZON FEE|DISCOUNT|SAMPLES?|"
    r"TEST|COMMISSION|GIFT VOUCHER|DELIVERY CHARGES?|SHIPPING|PACKING CHARGES?)\b")

ABC_STRATEGY = {
    "AX": "Top earner, steady demand: never run out. Tight reorder, low safety stock.",
    "AY": "Top earner, variable demand: review weekly, keep moderate safety stock.",
    "AZ": "Top earner, erratic demand: order in smaller batches, watch closely.",
    "BX": "Mid earner, steady: automate reorders at the reorder point.",
    "BY": "Mid earner, variable: standard reorder with safety stock.",
    "BZ": "Mid earner, erratic: order on demand, avoid bulk buying.",
    "CX": "Low earner, steady: keep minimal stock, order in economical lots.",
    "CY": "Low earner, variable: reduce range or order only when needed.",
    "CZ": "Low earner, erratic: prime dead-stock candidates. Consider delisting.",
}


# ============================================================================
# 1. DATA LOADING & COLUMN MAPPING
# ============================================================================
def _norm_col(c):
    return re.sub(r"[^a-z0-9]", "", str(c).lower())


def auto_map(columns, fields=FIELDS):
    """Guess which uploaded column matches each field (exact name first, then partial)."""
    norm = {c: _norm_col(c) for c in columns}
    mapping, used = {}, set()
    for f, (_, _, syns) in fields.items():               # pass 1: exact names
        for c, n in norm.items():
            if c not in used and n in syns:
                mapping[f] = c
                used.add(c)
                break
    for f, (_, _, syns) in fields.items():               # pass 2: partial names
        if f in mapping:
            continue
        for c, n in norm.items():
            if c not in used and any(s in n for s in syns if len(s) > 3):
                mapping[f] = c
                used.add(c)
                break
    return mapping


def read_table(data: bytes, name: str) -> pd.DataFrame:
    """Read CSV or Excel bytes into a DataFrame of strings (cleaned later)."""
    if name.lower().endswith((".xlsx", ".xlsm", ".xls")):
        sheets = pd.read_excel(io.BytesIO(data), sheet_name=None, dtype=str)
        frames = [s for s in sheets.values() if not s.empty]
        if not frames:
            raise ValueError("The Excel file has no data.")
        first_cols = list(frames[0].columns)
        # Combine sheets that share the same columns (e.g. one sheet per year)
        same = [f for f in frames if list(f.columns) == first_cols]
        return pd.concat(same, ignore_index=True)
    for enc in ("utf-8", "latin-1"):
        try:
            df = pd.read_csv(io.BytesIO(data), dtype=str, encoding=enc)
            if df.shape[1] == 1 and ";" in df.columns[0]:
                df = pd.read_csv(io.BytesIO(data), dtype=str, encoding=enc, sep=";")
            return df
        except UnicodeDecodeError:
            continue
    raise ValueError("Could not read the file. Save it as CSV (UTF-8) or Excel.")


def load_demo_data() -> pd.DataFrame:
    """Load the UCI Online Retail II demo dataset from data/ (Excel is converted
    to CSV once, because reading 1 million Excel rows is slow)."""
    if DEMO_CSV.is_file():
        return pd.read_csv(DEMO_CSV, dtype=str)
    if DEMO_XLSX.is_file():
        df = read_table(DEMO_XLSX.read_bytes(), DEMO_XLSX.name)
        df.to_csv(DEMO_CSV, index=False)
        return df
    raise FileNotFoundError(
        f"Demo dataset not found. Download it from {DATASET_URL}, unzip it and place "
        f"online_retail_II.xlsx in {DATA_DIR}")


# ============================================================================
# 2. DATA QUALITY CHECK & CLEANING
# ============================================================================
def _to_number(s: pd.Series) -> pd.Series:
    """'Rs 1,250.00' / '1,250' / ' 12 ' -> float. Keeps minus signs."""
    s = s.astype("string").str.replace(r"[^\d.\-]", "", regex=True)
    return pd.to_numeric(s.replace("", pd.NA), errors="coerce")


def _parse_dates(s: pd.Series, mode: str) -> pd.Series:
    s = s.astype("string").str.strip()
    if mode == "Day-first (DD/MM/YYYY)":
        return pd.to_datetime(s, errors="coerce", dayfirst=True)
    if mode == "Month-first (MM/DD/YYYY)":
        return pd.to_datetime(s, errors="coerce", dayfirst=False)
    a = pd.to_datetime(s, errors="coerce", dayfirst=False)
    b = pd.to_datetime(s, errors="coerce", dayfirst=True)
    if b.notna().sum() != a.notna().sum():
        return b if b.notna().sum() > a.notna().sum() else a
    return b   # both parse equally well: prefer DD/MM (common in India)


def normalise_name(s: pd.Series) -> pd.Series:
    """'Maggi 70 g ' and 'MAGGI 70G' -> 'MAGGI 70G' (used to find duplicate names)."""
    s = s.astype("string").str.upper().str.replace(r"[^A-Z0-9 ]", " ", regex=True)
    s = s.str.replace(r"(\d)\s+(G|GM|GMS|KG|ML|L|LTR|PCS|PC|MG)\b", r"\1\2", regex=True)
    s = s.str.replace(r"(\d)(GM|GMS)\b", r"\1G", regex=True)
    return s.str.replace(r"\s+", " ", regex=True).str.strip()


def quality_report(raw: pd.DataFrame, mapping: dict) -> pd.DataFrame:
    """Issues found BEFORE cleaning (shown to the user)."""
    rows = []
    n = len(raw)
    for f, col in mapping.items():
        miss = int(raw[col].isna().sum() + (raw[col].astype("string").str.strip() == "").sum())
        if miss:
            rows.append(("Missing values", f"{FIELDS[f][0]} ('{col}')", miss))
    cols = list(mapping.values())
    rows.append(("Exact duplicate rows", "All mapped columns", int(raw[cols].duplicated().sum())))
    qty = _to_number(raw[mapping["quantity"]])
    rows.append(("Quantity not a number", mapping["quantity"], int(qty.isna().sum())))
    rows.append(("Negative quantity (returns)", mapping["quantity"], int((qty < 0).sum())))
    rows.append(("Zero quantity", mapping["quantity"], int((qty == 0).sum())))
    if "price" in mapping:
        pr = _to_number(raw[mapping["price"]])
        rows.append(("Zero or negative price", mapping["price"], int((pr <= 0).sum())))
    if "invoice" in mapping:
        canc = raw[mapping["invoice"]].astype("string").str.upper().str.startswith("C").fillna(False)
        rows.append(("Cancelled invoices (start with 'C')", mapping["invoice"], int(canc.sum())))
    code = raw[mapping["product_code"]].astype("string").str.strip().str.upper() \
        if "product_code" in mapping else pd.Series(pd.NA, index=raw.index, dtype="string")
    name = raw[mapping["product_name"]].astype("string").str.upper() \
        if "product_name" in mapping else pd.Series("", index=raw.index, dtype="string")
    nonprod = code.isin(NON_PRODUCT_CODES).fillna(False) | name.str.contains(NON_PRODUCT_WORDS).fillna(False)
    rows.append(("Non-product lines (postage, fees, adjustments)", "Product code / name", int(nonprod.sum())))
    out = pd.DataFrame(rows, columns=["Issue", "Column", "Rows"])
    out["% of rows"] = (out["Rows"] / max(n, 1) * 100).round(2)
    return out[out["Rows"] > 0].reset_index(drop=True)


def clean_data(raw: pd.DataFrame, mapping: dict, opts: dict):
    """Clean the mapped data. Returns (sales, returns, cleaning_log, extras)."""
    log = []

    def add(step, issue, rows, action, why):
        log.append({"Step": step, "Issue": issue, "Rows affected": int(rows),
                    "Action": action, "Why": why})

    df = pd.DataFrame({f: raw[c] for f, c in mapping.items()})
    n0 = len(df)

    # --- text columns: trim, collapse spaces, blank -> missing
    for f in ("invoice", "product_code", "product_name", "customer", "location"):
        if f in df:
            s = df[f].astype("string").str.strip().str.replace(r"\s+", " ", regex=True)
            df[f] = s.mask(s.str.upper().isin(["", "NAN", "NONE", "NULL", "NA", "-"]))
    if "product_code" in df:
        df["product_code"] = df["product_code"].str.upper()
    if "customer" in df:
        df["customer"] = df["customer"].str.replace(r"\.0$", "", regex=True)

    # --- 1. dates
    df["date"] = _parse_dates(df["date"], opts.get("date_format", "Auto-detect"))
    bad = df["date"].isna()
    add("1. Dates", "Date missing or unreadable", bad.sum(), "Removed",
        "A sale without a valid date cannot be placed on the timeline.")
    df = df[~bad]

    # --- 2. numbers
    df["qty"] = _to_number(df.pop("quantity"))
    bad = df["qty"].isna()
    add("2. Quantity", "Quantity missing / not a number", bad.sum(), "Removed",
        "Quantity is essential for demand analysis.")
    df = df[~bad]
    has_price = "price" in df or "amount" in df
    if "price" in df:
        df["price"] = _to_number(df["price"])
    elif "amount" in df:
        df["price"] = _to_number(df["amount"]) / df["qty"].replace(0, np.nan)
        add("2. Price", "No unit price column", 0, "Unit price = line total / quantity",
            "Revenue needs a unit price.")
    else:
        df["price"] = 1.0
        add("2. Price", "No price or amount column", 0, "Revenue analysis uses units instead of money",
            "Without prices, ABC ranking falls back to quantity sold.")
    df = df.drop(columns=["amount"], errors="ignore")

    # --- 3. product key & names
    if "product_code" not in df and "product_name" not in df:
        raise ValueError("Map at least a product code or a product name column.")
    if "product_code" in df:
        miss_code = df["product_code"].isna()
        if "product_name" in df:
            df.loc[miss_code, "product_code"] = normalise_name(df.loc[miss_code, "product_name"])
        miss = df["product_code"].isna()
        add("3. Products", "Product code missing", miss_code.sum(),
            "Filled from product name where possible, otherwise removed",
            "Every sale must belong to a product.")
        df = df[~miss]
        df["key"] = df["product_code"]
        if "product_name" in df:
            miss_name = df["product_name"].isna()
            pairs = df.loc[~miss_name, ["key", "product_name"]].value_counts().reset_index()
            canon = pairs.drop_duplicates("key").set_index("key")["product_name"]
            multi = int((pairs.groupby("key").size() > 1).sum())
            df["name"] = df["key"].map(canon)
            df["name"] = df["name"].fillna("Item " + df["key"])
            add("3. Products", "Product name missing", miss_name.sum(),
                "Filled with the most common name used for that product code",
                "Blank names make reports unreadable.")
            add("3. Products", "Same code recorded with different names", multi,
                "Standardised to the most frequent name per code",
                "Spelling variants of one product should be reported once.")
        else:
            df["name"] = df["key"]
    else:
        miss = df["product_name"].isna()
        add("3. Products", "Product name missing", miss.sum(), "Removed",
            "Every sale must belong to a product.")
        df = df[~miss]
        df["key"] = normalise_name(df["product_name"])
        variants = df.groupby("key")["product_name"].nunique()
        add("3. Products", "Same product written in different ways (e.g. 'Maggi 70g' / 'MAGGI 70 G')",
            int((variants > 1).sum()), "Merged using a normalised name",
            "Otherwise one product would be split into several.")
        pairs = df[["key", "product_name"]].value_counts().reset_index()
        df["name"] = df["key"].map(pairs.drop_duplicates("key").set_index("key")["product_name"])
    df = df.drop(columns=["product_code", "product_name"], errors="ignore")

    # --- 4. duplicates
    if opts.get("remove_duplicates", True):
        dup = df.duplicated()
        add("4. Duplicates", "Exact duplicate lines", dup.sum(), "Removed",
            "The same line recorded twice would double-count sales.")
        df = df[~dup]

    # --- 5. returns / cancellations (kept separately for return-rate analysis)
    is_ret = df["qty"] < 0
    if "invoice" in df:
        is_ret |= df["invoice"].str.upper().str.startswith("C").fillna(False)
    returns = df[is_ret].copy()
    returns["qty"] = returns["qty"].abs()
    add("5. Returns", "Cancelled invoices / negative quantities", is_ret.sum(),
        "Moved to a separate returns table", "Returns are not sales, but return rates are useful.")
    df = df[~is_ret]
    matched = 0
    if "customer" in df and len(returns):
        # A cancellation with the same customer, product and quantity as an earlier sale
        # means that order never really happened: remove both lines.
        r = returns.dropna(subset=["customer"])[["customer", "key", "qty", "date"]].reset_index(names="rid")
        cand = df["customer"].notna() & df["key"].isin(r["key"].unique())
        sl = df.loc[cand, ["customer", "key", "qty", "date"]].reset_index(names="sid")
        m = r.merge(sl, on=["customer", "key", "qty"], suffixes=("_r", "_s"))
        m = m[m["date_s"] <= m["date_r"]].sort_values("date_s", ascending=False)
        m = m.drop_duplicates("rid").drop_duplicates("sid")
        matched = len(m)
        df = df.drop(index=m["sid"])
        returns = returns.drop(index=m["rid"])
    add("5. Returns", "Orders cancelled in full (sale matched to its cancellation: same customer, "
        "product and quantity)", matched, "Both lines removed",
        "A cancelled order was never really sold; keeping it would inflate revenue and demand.")
    zero = df["qty"] == 0
    add("5. Returns", "Zero quantity", zero.sum(), "Removed", "No units were sold.")
    df = df[~zero]

    # --- 6. invalid prices
    if has_price:
        badp = df["price"].isna() | (df["price"] <= 0)
        add("6. Prices", "Price missing, zero or negative", badp.sum(), "Removed",
            "Free or mis-keyed lines distort revenue (e.g. adjustments, samples).")
        df = df[~badp]

    # --- 7. non-product lines
    if opts.get("remove_non_products", True):
        nonprod = df["key"].isin(NON_PRODUCT_CODES) | \
            df["name"].astype("string").str.upper().str.contains(NON_PRODUCT_WORDS).fillna(False)
        add("7. Non-products", "Postage, fees, manual adjustments, samples", nonprod.sum(),
            "Removed", "They are not stock items and must not be reordered.")
        df = df[~nonprod]
        rnp = returns["key"].isin(NON_PRODUCT_CODES) |             returns["name"].astype("string").str.upper().str.contains(NON_PRODUCT_WORDS).fillna(False)
        returns = returns[~rnp]

    # --- 8. extreme bulk orders (outliers) - flagged, capped only for demand planning
    g = df.groupby("key")["qty"]
    q1, q3, cnt = g.transform("quantile", 0.25), g.transform("quantile", 0.75), g.transform("size")
    # Upper fence: Q3 + 3 x IQR, but at least 10 x Q3. In wholesale data orders of
    # a few hundred units are normal, so only truly exceptional orders are flagged.
    fence = np.maximum(q3 + 3 * (q3 - q1), 10 * q3)
    out = (df["qty"] > fence) & (cnt >= 5) & (df["qty"] > 1)
    if opts.get("cap_outliers", True):
        df["qty_demand"] = np.where(out, np.ceil(fence), df["qty"])
        action = "Kept for revenue; capped at the product's upper fence for demand planning"
    else:
        df["qty_demand"] = df["qty"]
        action = "Kept unchanged (option switched off)"
    add("8. Outliers", "Exceptional bulk orders (above max(Q3 + 3 x IQR, 10 x Q3) for that product)",
        out.sum(), action,
        "One-off bulk orders are real sales, but would inflate normal reorder levels.")

    df["revenue"] = df["qty"] * df["price"]
    df["week"] = df["date"].dt.to_period("W-SUN").dt.start_time
    df = df.reset_index(drop=True)

    # Potential duplicate products across different codes (reported, not merged)
    dupe_names = pd.DataFrame()
    if "product_code" in mapping and "product_name" in mapping:
        nm = df[["key", "name"]].drop_duplicates("key").copy()
        nm["norm"] = normalise_name(nm["name"])
        grp = nm.groupby("norm")["key"].agg(["count", lambda s: ", ".join(s.head(5))])
        grp.columns = ["codes", "example_codes"]
        dupe_names = grp[grp["codes"] > 1].sort_values("codes", ascending=False).reset_index() \
            .rename(columns={"norm": "Normalised name", "codes": "Different codes",
                             "example_codes": "Codes"})

    extras = {"rows_before": n0, "rows_after": len(df), "returns_rows": len(returns),
              "has_price": has_price, "has_invoice": "invoice" in df,
              "has_customer": "customer" in df, "has_location": "location" in df,
              "has_time": bool((df["date"].dt.hour > 0).mean() > 0.5),
              "outliers": int(out.sum()), "dupe_names": dupe_names}
    return df, returns, pd.DataFrame(log), extras


# ============================================================================
# 3. DESCRIPTIVE ANALYTICS: PRODUCT TABLE, ABC-XYZ, SEGMENTS
# ============================================================================
def weekly_stats(sales, start, end_week, n_weeks):
    """Mean / std / active weeks of weekly demand between start and end_week (zeros included)."""
    w = sales[(sales["week"] >= start) & (sales["week"] <= end_week)]
    wk = w.groupby(["key", "week"])["qty_demand"].sum()
    s, ss = wk.groupby(level=0).sum(), (wk ** 2).groupby(level=0).sum()
    mean = s / n_weeks
    std = np.sqrt(np.maximum(ss / n_weeks - mean ** 2, 0))
    return pd.DataFrame({"weekly_mean": mean, "weekly_std": std,
                         "active_weeks": wk.groupby(level=0).size()})


def product_table(sales, returns, extras, window_days):
    end = sales["date"].max()
    last_full_week = sales["week"].max() if end.dayofweek == 6 else \
        sales["week"].max() - pd.Timedelta(weeks=1)
    n_weeks = max(int(round(window_days / 7)), 4)
    wstart = last_full_week - pd.Timedelta(weeks=n_weeks - 1)

    g = sales.groupby("key")
    p = pd.DataFrame({
        "name": g["name"].first(),
        "first_sale": g["date"].min(), "last_sale": g["date"].max(),
        "units_all_time": g["qty"].sum(), "revenue_all_time": g["revenue"].sum()})
    w = sales[sales["date"] > end - pd.Timedelta(days=window_days)]
    gw = w.groupby("key")
    p["units_window"] = gw["qty"].sum()
    p["revenue_window"] = gw["revenue"].sum()
    p["orders_window"] = gw["invoice"].nunique() if extras["has_invoice"] else gw.size()
    if extras["has_customer"]:
        p["customers_window"] = gw["customer"].nunique()
    p = p.fillna({"units_window": 0, "revenue_window": 0, "orders_window": 0, "customers_window": 0})
    p["avg_price"] = (g["revenue"].sum() / g["qty"].sum()).round(2)
    p["days_since_last_sale"] = (end - p["last_sale"]).dt.days
    ret = returns.groupby("key")["qty"].sum() if len(returns) else pd.Series(dtype=float)
    p["units_returned"] = ret.reindex(p.index).fillna(0)
    rate = p["units_returned"] / p["units_all_time"] * 100
    # Returns larger than recorded sales mean the original sale is outside the data
    # (e.g. sold before the file starts); such rates are not meaningful.
    p["return_rate_pct"] = rate.where(p["units_returned"] <= p["units_all_time"]).round(1)

    ws = weekly_stats(sales, wstart, last_full_week, n_weeks)
    p = p.join(ws).fillna({"weekly_mean": 0, "weekly_std": 0, "active_weeks": 0})
    p["cv"] = (p["weekly_std"] / p["weekly_mean"]).replace([np.inf], np.nan)

    # ABC (value) on the analysis window
    act = p[p["revenue_window"] > 0].sort_values("revenue_window", ascending=False)
    share = act["revenue_window"].cumsum() / act["revenue_window"].sum()
    abc = pd.Series(np.where(share.shift(fill_value=0) < 0.80, "A",
                             np.where(share.shift(fill_value=0) < 0.95, "B", "C")), index=act.index)
    p["ABC"] = abc.reindex(p.index).fillna("-")
    p["revenue_share_pct"] = (p["revenue_window"] / max(p["revenue_window"].sum(), 1e-9) * 100).round(3)
    # XYZ (demand variability)
    p["XYZ"] = np.select([p["cv"] < 0.5, p["cv"] < 1.0], ["X", "Y"], "Z")
    p.loc[p["ABC"] == "-", "XYZ"] = "-"
    p["class"] = np.where(p["ABC"] == "-", "Not sold in window", p["ABC"] + p["XYZ"])
    meta = {"end": end, "last_full_week": last_full_week, "window_start_week": wstart,
            "n_weeks": n_weeks}
    return p, meta


def segment_products(sales, products, meta, k=4):
    """K-Means clustering of products by sales behaviour, with automatic naming."""
    end = meta["end"]
    act = products[products["units_window"] > 0].copy()
    if len(act) < 20:
        return None
    w = sales[sales["date"] > end - pd.Timedelta(days=365)]
    last13 = w[w["date"] > end - pd.Timedelta(weeks=13)].groupby("key")["qty"].sum()
    prev13 = w[(w["date"] <= end - pd.Timedelta(weeks=13)) &
               (w["date"] > end - pd.Timedelta(weeks=26))].groupby("key")["qty"].sum()
    monthly = w.groupby(["key", w["date"].dt.to_period("M")])["qty"].sum()
    peak = (monthly.groupby(level=0).max() / monthly.groupby(level=0).sum())
    f = pd.DataFrame(index=act.index)
    f["log_revenue"] = np.log1p(act["revenue_window"])
    f["active_week_share"] = act["active_weeks"] / meta["n_weeks"]
    f["cv"] = act["cv"].fillna(act["cv"].max()).clip(upper=10)
    f["trend"] = np.log((last13.reindex(f.index).fillna(0) + 1) / (prev13.reindex(f.index).fillna(0) + 1))
    f["recency_days"] = act["days_since_last_sale"]
    f["peak_month_share"] = peak.reindex(f.index).fillna(1)
    X = StandardScaler().fit_transform(f)
    km = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE).fit(X)
    f["cluster"] = km.labels_
    sample = np.random.default_rng(RANDOM_STATE).choice(len(X), min(len(X), 4000), replace=False)
    sil = float(silhouette_score(X[sample], km.labels_[sample]))

    # Name clusters from their average behaviour
    c = f.groupby("cluster").mean()
    names, left = {}, list(c.index)
    for label, col, fn in [("Dormant / fading", "recency_days", "idxmax"),
                           ("Steady sellers", "active_week_share", "idxmax"),
                           ("Seasonal / peaky", "peak_month_share", "idxmax")]:
        idx = getattr(c.loc[left, col], fn)()
        names[idx] = label
        left.remove(idx)
    for idx in left:
        names[idx] = "Occasional / slow movers"
    f["segment"] = f["cluster"].map(names)
    summary = f.groupby("segment").agg(
        products=("cluster", "size"), avg_active_week_share=("active_week_share", "mean"),
        avg_recency_days=("recency_days", "mean"), avg_peak_month_share=("peak_month_share", "mean"),
        avg_trend=("trend", "mean"))
    rev = act.groupby(f["segment"])["revenue_window"].sum()
    summary["revenue_share_pct"] = (rev / rev.sum() * 100).round(1)
    return {"features": f, "summary": summary.round(3), "silhouette": round(sil, 3)}


# ============================================================================
# 4. AI MODEL 1: DEAD-STOCK RISK PREDICTION
# ============================================================================
DEAD_FEATURES = ["days_since_last", "age_days", "qty_30d", "qty_90d", "qty_365d", "orders_90d",
                 "customers_365d", "active_weeks_52", "cv_52", "trend_90d", "avg_price"]


def dead_stock_snapshot(sales, cutoff, horizon, end, has_invoice, has_customer):
    """Features for every product using ONLY data up to `cutoff`; label = product
    sold nothing in the next `horizon` days (only when that period is observed)."""
    hist = sales[sales["date"] <= cutoff]
    g = hist.groupby("key")["date"]
    first, last = g.min(), g.max()
    elig = last.index[((cutoff - last).dt.days <= 365) & ((cutoff - first).dt.days >= 28)]
    h = hist[hist["key"].isin(elig)]
    f = pd.DataFrame(index=elig)
    f["days_since_last"] = (cutoff - last[elig]).dt.days
    f["age_days"] = (cutoff - first[elig]).dt.days

    def recent(days):
        return h[h["date"] > cutoff - pd.Timedelta(days=days)]
    for d in (30, 90, 365):
        f[f"qty_{d}d"] = recent(d).groupby("key")["qty_demand"].sum()
    r90 = recent(90)
    f["orders_90d"] = r90.groupby("key")["invoice"].nunique() if has_invoice else r90.groupby("key").size()
    f["customers_365d"] = recent(365).groupby("key")["customer"].nunique() if has_customer else 0
    prev90 = h[(h["date"] <= cutoff - pd.Timedelta(days=90)) &
               (h["date"] > cutoff - pd.Timedelta(days=180))].groupby("key")["qty_demand"].sum()
    wk = recent(364).groupby(["key", "week"])["qty_demand"].sum()
    s, ss = wk.groupby(level=0).sum(), (wk ** 2).groupby(level=0).sum()
    mean = s / 52
    f["active_weeks_52"] = wk.groupby(level=0).size()
    f["cv_52"] = np.sqrt(np.maximum(ss / 52 - mean ** 2, 0)) / mean
    r365 = recent(365)
    f["avg_price"] = r365.groupby("key")["revenue"].sum() / r365.groupby("key")["qty"].sum()
    f = f.fillna(0)
    f["trend_90d"] = np.log((f["qty_90d"] + 1) / (prev90.reindex(f.index).fillna(0) + 1))
    for c in ("qty_30d", "qty_90d", "qty_365d", "orders_90d", "customers_365d", "avg_price"):
        f[c] = np.log1p(f[c])
    if cutoff + pd.Timedelta(days=horizon) <= end:
        fut = sales[(sales["date"] > cutoff) & (sales["date"] <= cutoff + pd.Timedelta(days=horizon))]
        sold = fut.groupby("key")["qty"].sum().reindex(f.index).fillna(0)
        f["dead"] = (sold == 0).astype(int)
    f["cutoff"] = cutoff
    return f


def train_dead_stock_model(sales, extras, horizon, progress=None):
    end = sales["date"].max().normalize()
    start = sales["date"].min()
    val_cut = end - pd.Timedelta(days=horizon)
    train_cuts = [end - pd.Timedelta(days=2 * horizon + 28 * k) for k in range(6)]
    train_cuts = [c for c in train_cuts if c - start >= pd.Timedelta(days=120)]
    if not train_cuts:
        return None
    args = (end, extras["has_invoice"], extras["has_customer"])
    train = pd.concat([dead_stock_snapshot(sales, c, horizon, *args) for c in train_cuts])
    val = dead_stock_snapshot(sales, val_cut, horizon, *args)
    if train["dead"].nunique() < 2 or val["dead"].nunique() < 2:
        return None
    Xtr, ytr, Xva, yva = train[DEAD_FEATURES], train["dead"], val[DEAD_FEATURES], val["dead"]

    models = {
        "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(
            max_iter=2000, class_weight="balanced")),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, min_samples_leaf=5, class_weight="balanced_subsample",
            n_jobs=-1, random_state=RANDOM_STATE),
    }
    rows, probs = [], {}
    rule_pred = (Xva["days_since_last"] >= min(horizon, 90)).astype(int)
    rows.append({"Model": f"Rule: no sale in last {min(horizon, 90)} days",
                 "ROC-AUC": roc_auc_score(yva, Xva["days_since_last"]),
                 "Precision": precision_score(yva, rule_pred, zero_division=0),
                 "Recall": recall_score(yva, rule_pred, zero_division=0),
                 "F1": f1_score(yva, rule_pred, zero_division=0)})
    for name, m in models.items():
        if progress:
            progress(f"Training dead-stock model: {name}")
        m.fit(Xtr, ytr)
        pr = m.predict_proba(Xva)[:, 1]
        probs[name] = pr
        pred = (pr >= 0.5).astype(int)
        rows.append({"Model": name, "ROC-AUC": roc_auc_score(yva, pr),
                     "Precision": precision_score(yva, pred, zero_division=0),
                     "Recall": recall_score(yva, pred, zero_division=0),
                     "F1": f1_score(yva, pred, zero_division=0)})
    metrics = pd.DataFrame(rows)
    best = metrics.iloc[1:].sort_values("ROC-AUC", ascending=False).iloc[0]["Model"]

    imp = permutation_importance(models[best], Xva, yva, scoring="roc_auc", n_repeats=5,
                                 random_state=RANDOM_STATE, n_jobs=-1)
    importance = pd.DataFrame({"feature": DEAD_FEATURES, "importance": imp.importances_mean}) \
        .sort_values("importance", ascending=False)

    # Refit on all labelled snapshots and score products as of today
    final = models[best]
    both = pd.concat([train, val])
    final.fit(both[DEAD_FEATURES], both["dead"])
    now = dead_stock_snapshot(sales, end, horizon, *args)
    now["dead_risk"] = final.predict_proba(now[DEAD_FEATURES])[:, 1]
    now["risk_level"] = pd.cut(now["dead_risk"], [-0.01, 0.35, 0.6, 1.01],
                               labels=["Low", "Medium", "High"]).astype(str)
    return {"metrics": metrics.round(3), "best": best, "importance": importance,
            "scores": now[["dead_risk", "risk_level"]], "horizon": horizon,
            "n_train": len(train), "n_val": len(val), "val_cutoff": val_cut,
            "train_cutoffs": train_cuts, "base_rate": float(yva.mean())}


# ============================================================================
# 5. AI MODEL 2: DEMAND FORECAST (next 4 weeks per product)
# ============================================================================
FC_FEATURES = ["lag0", "lag1", "lag2", "lag3", "sum4", "mean12", "std12", "mean52", "active12",
               "ly_next4", "weeks_since_first", "woy", "month"]


def build_weekly_grid(sales, meta, min_active_weeks=8):
    lfw = meta["last_full_week"]
    s = sales[sales["week"] <= lfw]
    wk = s.groupby(["key", "week"])["qty_demand"].sum()
    recent = wk[wk.index.get_level_values("week") > lfw - pd.Timedelta(weeks=52)]
    keys = recent.groupby(level=0).size()
    keys = keys[keys >= min_active_weeks].index
    weeks = pd.date_range(s["week"].min(), lfw, freq="7D")
    idx = pd.MultiIndex.from_product([keys, weeks], names=["key", "week"])
    grid = wk.reindex(idx, fill_value=0).rename("qty").reset_index()
    first = s[s["key"].isin(keys)].groupby("key")["week"].min()
    grid = grid[grid["week"] >= grid["key"].map(first)].reset_index(drop=True)
    return grid


def forecast_features(grid):
    g = grid.groupby("key")["qty"]
    gd = grid.copy()
    gd["lag0"] = gd["qty"]
    for i in (1, 2, 3):
        gd[f"lag{i}"] = g.shift(i)

    def roll(w, fn):
        return getattr(g.rolling(w, min_periods=1), fn)().reset_index(level=0, drop=True)
    gd["sum4"], gd["mean12"], gd["std12"], gd["mean52"] = roll(4, "sum"), roll(12, "mean"), \
        roll(12, "std"), roll(52, "mean")
    gd["active12"] = (gd["qty"] > 0).astype(int).groupby(gd["key"]).rolling(12, min_periods=1) \
        .sum().reset_index(level=0, drop=True)
    gd["ly_next4"] = sum(g.shift(s) for s in (48, 49, 50, 51))
    gd["weeks_since_first"] = g.cumcount()
    nxt = gd["week"] + pd.Timedelta(weeks=1)
    gd["woy"] = nxt.dt.isocalendar().week.astype(int)
    gd["month"] = nxt.dt.month
    gd["target"] = sum(g.shift(-s) for s in (1, 2, 3, 4))
    return gd


def train_forecaster(sales, meta, progress=None):
    if progress:
        progress("Building weekly demand history")
    grid = build_weekly_grid(sales, meta)
    if grid["key"].nunique() < 10:
        return None
    gd = forecast_features(grid)
    T = meta["last_full_week"]
    test_origins = [T - pd.Timedelta(weeks=4), T - pd.Timedelta(weeks=8)]
    train = gd[(gd["week"] <= T - pd.Timedelta(weeks=12)) & gd["target"].notna()]
    test = gd[gd["week"].isin(test_origins)].copy()
    if len(train) < 500 or test.empty:
        return None
    if progress:
        progress("Training demand forecast model (Gradient Boosting)")
    model = HistGradientBoostingRegressor(loss="poisson", max_iter=400, learning_rate=0.05,
                                          random_state=RANDOM_STATE)
    model.fit(train[FC_FEATURES], train["target"])
    preds = {
        "Naive: last 4 weeks": test["sum4"],
        "Moving average (12 weeks)": test["mean12"] * 4,
        "Seasonal naive: same weeks last year": test["ly_next4"].fillna(test["sum4"]),
        "Gradient Boosting (AI)": pd.Series(model.predict(test[FC_FEATURES]), index=test.index),
    }
    rows = []
    for name, p in preds.items():
        err = p - test["target"]
        rows.append({"Method": name, "MAE (units / 4 weeks)": mean_absolute_error(test["target"], p),
                     "WAPE %": err.abs().sum() / test["target"].sum() * 100,
                     "Bias %": err.sum() / test["target"].sum() * 100})
    metrics = pd.DataFrame(rows).round(2)
    # Selection rule: methods within 1 percentage point of the best WAPE are treated as
    # equally accurate; among them the least biased wins, because consistent
    # under-forecasting causes stock-outs and over-forecasting causes dead stock.
    tied = metrics[metrics["WAPE %"] <= metrics["WAPE %"].min() + 1.0]
    best = tied.loc[tied["Bias %"].abs().idxmin(), "Method"]

    # Forecast the next 4 weeks from the latest week
    latest = gd[gd["week"] == T].set_index("key")
    if best == "Gradient Boosting (AI)":
        model.fit(gd.loc[gd["target"].notna(), FC_FEATURES], gd.loc[gd["target"].notna(), "target"])
        fc = pd.Series(model.predict(latest[FC_FEATURES]), index=latest.index)
    elif best.startswith("Naive"):
        fc = latest["sum4"]
    elif best.startswith("Moving"):
        fc = latest["mean12"] * 4
    else:
        fc = latest["ly_next4"].fillna(latest["sum4"])
    forecast = pd.DataFrame({"forecast_4w": fc.clip(lower=0), "method": best,
                             "weekly_std_12": latest["std12"].fillna(0)})
    return {"metrics": metrics, "best": best, "forecast": forecast, "grid": grid,
            "test_origins": test_origins, "n_products": grid["key"].nunique(),
            "n_train_rows": len(train)}


# ============================================================================
# 6. INVENTORY: SAFETY STOCK, REORDER POINT, STOCK / EXPIRY CHECKS
# ============================================================================
def build_plan(products, fc, dead, sales, meta, lead_time, review_days, service_level,
               dead_days, stock=None):
    """Reorder plan for every product sold in the last 12 weeks (+ stock checks)."""
    end, lfw = meta["end"], meta["last_full_week"]
    ws12 = weekly_stats(sales, lfw - pd.Timedelta(weeks=11), lfw, 12)
    plan = products[["name", "ABC", "XYZ", "class", "avg_price", "days_since_last_sale",
                     "units_window"]].join(ws12, rsuffix="_12")
    plan = plan[plan["weekly_mean"].notna() | products.index.isin(stock.index if stock is not None else [])]
    plan["weekly_mean"] = plan["weekly_mean"].fillna(0)
    plan["weekly_std"] = plan["weekly_std"].fillna(0)
    plan["forecast_4w"] = plan["weekly_mean"] * 4
    plan["forecast_method"] = "12-week average (little history)"
    if fc is not None:
        f = fc["forecast"].reindex(plan.index)
        has = f["forecast_4w"].notna()
        plan.loc[has, "forecast_4w"] = f.loc[has, "forecast_4w"]
        plan.loc[has, "forecast_method"] = f.loc[has, "method"]
    plan["daily_demand"] = plan["forecast_4w"] / 28
    z = NormalDist().inv_cdf(service_level)
    sigma_d = plan["weekly_std"] / np.sqrt(7)
    plan["safety_stock"] = np.ceil(z * sigma_d * np.sqrt(lead_time))
    plan["reorder_point"] = np.ceil(plan["daily_demand"] * lead_time + plan["safety_stock"])
    plan["order_up_to"] = np.ceil(plan["daily_demand"] * (lead_time + review_days) + plan["safety_stock"])
    if dead is not None:
        plan = plan.join(dead["scores"])
    plan["dead_risk"] = plan.get("dead_risk", pd.Series(np.nan, index=plan.index))
    plan["risk_level"] = plan.get("risk_level", pd.Series("n/a", index=plan.index)).fillna("n/a")

    if stock is not None:
        plan = plan.join(stock, how="left")
        plan["current_stock"] = plan["current_stock"].fillna(0)
        unit_value = plan["unit_cost"].fillna(plan["avg_price"]) if "unit_cost" in plan else plan["avg_price"]
        plan["stock_value"] = (plan["current_stock"] * unit_value).round(2)
        dd = plan["daily_demand"].replace(0, np.nan)
        plan["days_of_cover"] = (plan["current_stock"] / dd).round(0)
        plan["suggested_order_qty"] = np.maximum(0, plan["order_up_to"] - plan["current_stock"])
        is_dead = (plan["days_since_last_sale"] >= dead_days) | (plan["risk_level"] == "High")
        plan["status"] = np.select(
            [(plan["current_stock"] > 0) & is_dead,
             (plan["daily_demand"] > 0) & (plan["current_stock"] <= plan["reorder_point"]),
             plan["days_of_cover"] > 120],
            ["Dead / slow stock", "Reorder now", "Overstocked"], "OK")
        plan.loc[plan["status"] != "Reorder now", "suggested_order_qty"] = 0
        if "expiry_date" in plan:
            dte = (plan["expiry_date"] - end).dt.days
            plan["days_to_expiry"] = dte
            exp_sales = plan["daily_demand"] * dte.clip(lower=0)
            plan["units_at_expiry_risk"] = np.where(dte.notna(),
                                                    np.maximum(0, np.floor(plan["current_stock"] - exp_sales)), 0)
            plan["value_at_expiry_risk"] = (plan["units_at_expiry_risk"] * unit_value).round(2)
            share = plan["units_at_expiry_risk"] / plan["current_stock"].replace(0, np.nan)
            plan["expiry_action"] = np.select(
                [plan["units_at_expiry_risk"] <= 0, dte <= 0, dte <= 14, dte <= 45, share > 0.5],
                ["-", "Expired: remove from shelf", "Clearance 30-50% off / return to supplier",
                 "Discount 15-25% + front-shelf display", "Bundle offer / 10% off"],
                "Promote (combo, display)")
    return plan


def parse_stock_sheet(raw, mapping, products, key_is_code):
    s = pd.DataFrame({"product": raw[mapping["product"]].astype("string").str.strip()})
    s["current_stock"] = _to_number(raw[mapping["stock"]]).fillna(0)
    if mapping.get("cost"):
        s["unit_cost"] = _to_number(raw[mapping["cost"]])
    if mapping.get("expiry"):
        s["expiry_date"] = _parse_dates(raw[mapping["expiry"]], "Auto-detect")
    by_code = s["product"].str.upper()
    by_name = normalise_name(s["product"])
    name_to_key = pd.Series(products.index, index=normalise_name(products["name"])).groupby(level=0).first()
    key = by_code.where(by_code.isin(products.index))
    key = key.fillna(by_name.map(name_to_key))
    if not key_is_code:
        key = key.fillna(by_name.where(by_name.isin(products.index)))
    s["key"] = key
    unmatched = int(s["key"].isna().sum())
    s = s.dropna(subset=["key"]).groupby("key").agg(
        {c: ("sum" if c == "current_stock" else "first") for c in s.columns if c not in ("key", "product")})
    return s, unmatched


def simulate_demo_stock(products, meta, seed=RANDOM_STATE):
    """SIMULATED stock sheet for demonstrating stock features on the demo dataset.
    The UCI dataset has no stock or expiry data, so these numbers are NOT real."""
    rng = np.random.default_rng(seed)
    p = products[products["units_all_time"] > 0]
    daily = p["weekly_mean"] / 7
    days = rng.uniform(0, 120, len(p))
    stock = np.ceil(daily * days)
    dormant = (p["days_since_last_sale"] > 90).values
    stock[dormant] = np.ceil(p["units_all_time"].values[dormant] / 104 * rng.uniform(1, 12, dormant.sum()))
    out = pd.DataFrame({"current_stock": stock,
                        "unit_cost": (p["avg_price"] * rng.uniform(0.5, 0.7, len(p))).round(2)},
                       index=p.index)
    has_exp = rng.random(len(p)) < 0.35
    out["expiry_date"] = pd.NaT
    out.loc[has_exp, "expiry_date"] = meta["end"].normalize() + pd.to_timedelta(
        rng.integers(-5, 240, has_exp.sum()), unit="D")
    return out[out["current_stock"] > 0]


# ============================================================================
# 7. INSIGHTS, RECOMMENDATIONS & REPORTS
# ============================================================================
def money(x, cur):
    return f"{cur}{x:,.0f}"


def money_short(x, cur):
    """Compact money for dashboard tiles: 9865390 -> 9.87M."""
    for div, suffix in ((1e9, "B"), (1e6, "M"), (1e3, "K")):
        if abs(x) >= div:
            return f"{cur}{x / div:.2f}{suffix}"
    return f"{cur}{x:,.0f}"


def generate_insights(R, cur):
    p, ins = R["products"], []
    act = p[p["ABC"] != "-"]
    if len(act):
        a = act[act["ABC"] == "A"]
        ins.append(f"Only {len(a):,} of {len(act):,} products sold in the last "
                   f"{R['opts']['window_days']} days ({len(a) / len(act) * 100:.0f}%) earn 80% of revenue "
                   "(A-class). These must never go out of stock.")
        cz = act[act["class"] == "CZ"]
        if len(cz):
            ins.append(f"{len(cz):,} products are CZ (low value and erratic demand). They earn "
                       f"{cz['revenue_window'].sum() / act['revenue_window'].sum() * 100:.1f}% of revenue "
                       "and are the first candidates to reduce or delist.")
    stale = p[(p["days_since_last_sale"] >= R["opts"]["dead_days"]) &
              (p["days_since_last_sale"] <= 365)]
    ins.append(f"{len(stale):,} products sold before but have had no sale in the last "
               f"{R['opts']['dead_days']} days. Check whether they are still on the shelf.")
    d = R.get("dead")
    if d:
        hi = (d["scores"]["risk_level"] == "High").sum()
        m = d["metrics"].set_index("Model").loc[d["best"]]
        ins.append(f"The AI dead-stock model ({d['best']}, ROC-AUC {m['ROC-AUC']:.2f} on unseen data) flags "
                   f"{hi:,} currently active products as HIGH risk of selling nothing in the next "
                   f"{d['horizon']} days.")
    fc = R.get("forecast")
    if fc:
        m = fc["metrics"].set_index("Method")
        naive = m.loc["Naive: last 4 weeks", "WAPE %"]
        b = m.loc[fc["best"]]
        ins.append(f"Demand forecast method used: {fc['best']} (WAPE {b['WAPE %']:.1f}%, bias "
                   f"{b['Bias %']:+.1f}%). For comparison, the simple last-4-weeks guess has WAPE "
                   f"{naive:.1f}% and bias {m.loc['Naive: last 4 weeks', 'Bias %']:+.1f}%.")
    top_ret = p[(p["units_all_time"] >= 100)].sort_values("return_rate_pct", ascending=False).head(1)
    if len(top_ret) and top_ret["return_rate_pct"].iloc[0] > 0:
        r = top_ret.iloc[0]
        ins.append(f"Highest return rate among regular sellers: '{r['name']}' "
                   f"({r['return_rate_pct']:.1f}% of units returned). Check why: quality, wrong description or order mistakes.")
    mon = R["sales"].groupby(R["sales"]["date"].dt.month)["revenue"].sum()
    if len(mon) >= 6:
        pk = mon.idxmax()
        ins.append(f"Peak sales month is {pd.Timestamp(2000, pk, 1):%B}. Build stock of A-class items "
                   "4-6 weeks before it.")
    return ins


def plan_recommendations(plan, cur):
    recs = []
    if "status" not in plan:
        return ["Upload a current stock sheet (Reorder & Stock tab) to get item-level reorder, "
                "overstock and expiry actions."]
    ro = plan[plan["status"] == "Reorder now"]
    dead = plan[plan["status"] == "Dead / slow stock"]
    over = plan[plan["status"] == "Overstocked"]
    recs.append(f"Reorder now: {len(ro):,} items are at or below their reorder point "
                f"({(ro['ABC'] == 'A').sum()} of them are A-class). Suggested units: "
                f"{ro['suggested_order_qty'].sum():,.0f}.")
    recs.append(f"Dead / slow stock: {len(dead):,} items holding {money(dead['stock_value'].sum(), cur)}. "
                "Stop reordering them, run a clearance, or return them to the supplier.")
    recs.append(f"Overstocked: {len(over):,} items have more than 120 days of cover "
                f"({money(over['stock_value'].sum(), cur)}). Pause their reorders.")
    if "value_at_expiry_risk" in plan:
        ex = plan[plan["units_at_expiry_risk"] > 0]
        recs.append(f"Expiry risk: {len(ex):,} items will not sell out before expiry at the current rate "
                    f"({money(ex['value_at_expiry_risk'].sum(), cur)} at risk). See the suggested actions.")
    return recs


def build_excel(R, plan, cur):
    buf = io.BytesIO()
    p = R["products"].reset_index().rename(columns={"key": "product_code"})
    with pd.ExcelWriter(buf, engine="openpyxl") as xw:
        pd.DataFrame({"Insight": generate_insights(R, cur) + plan_recommendations(plan, cur)}) \
            .to_excel(xw, sheet_name="Summary", index=False)
        cols = ["name", "ABC", "XYZ", "risk_level", "forecast_4w", "safety_stock", "reorder_point",
                "order_up_to"] + [c for c in ("current_stock", "days_of_cover", "suggested_order_qty",
                                               "status", "stock_value", "days_to_expiry",
                                               "units_at_expiry_risk", "expiry_action") if c in plan]
        plan[cols].round(2).reset_index().rename(columns={"key": "product_code"}) \
            .to_excel(xw, sheet_name="Reorder plan", index=False)
        p.drop(columns=["first_sale", "last_sale"]).round(3).to_excel(xw, sheet_name="Products (ABC-XYZ)", index=False)
        R["log"].to_excel(xw, sheet_name="Cleaning log", index=False)
        R["quality"].to_excel(xw, sheet_name="Data quality", index=False)
        if R.get("dead"):
            R["dead"]["metrics"].to_excel(xw, sheet_name="Dead-stock model", index=False)
        if R.get("forecast"):
            R["forecast"]["metrics"].to_excel(xw, sheet_name="Forecast accuracy", index=False)
        if R.get("segments"):
            R["segments"]["summary"].reset_index().to_excel(xw, sheet_name="Segments", index=False)
    return buf.getvalue()


def build_html_report(R, plan, cur, figs):
    k = R["kpis"]
    kp = "".join(f"<div class='k'><div class='v'>{v}</div><div class='l'>{l}</div></div>" for l, v in k)
    ins = "".join(f"<li>{i}</li>" for i in generate_insights(R, cur))
    rec = "".join(f"<li>{i}</li>" for i in plan_recommendations(plan, cur))
    charts = "".join(f.to_html(full_html=False, include_plotlyjs=("cdn" if i == 0 else False))
                     for i, f in enumerate(figs))

    def tbl(df, n=15):
        return df.head(n).to_html(index=False, border=0, classes="t", float_format=lambda x: f"{x:,.2f}")
    sections = f"<h2>Data cleaning log</h2>{tbl(R['log'], 30)}"
    if R.get("dead"):
        sections += f"<h2>AI dead-stock model (validation on unseen period)</h2>{tbl(R['dead']['metrics'])}"
    if R.get("forecast"):
        sections += f"<h2>Demand forecast accuracy (last 8 weeks, unseen)</h2>{tbl(R['forecast']['metrics'])}"
    if "status" in plan:
        ro = plan[plan["status"] == "Reorder now"].sort_values("suggested_order_qty", ascending=False)
        sections += "<h2>Reorder now (top 15)</h2>" + tbl(ro.reset_index()[[
            "key", "name", "ABC", "current_stock", "reorder_point", "suggested_order_qty"]])
        dd = plan[plan["status"] == "Dead / slow stock"].sort_values("stock_value", ascending=False)
        sections += "<h2>Dead / slow stock (top 15 by value)</h2>" + tbl(dd.reset_index()[[
            "key", "name", "current_stock", "stock_value", "days_since_last_sale", "risk_level"]])
    return f"""<!DOCTYPE html><html><head><meta charset='utf-8'><title>ShelfSense Report</title>
<style>body{{font-family:Segoe UI,Arial,sans-serif;max-width:1100px;margin:24px auto;padding:0 16px;color:#1f2328}}
h1{{color:#2f6b4f;margin-bottom:0}}h2{{color:#2f6b4f;border-bottom:2px solid #e3ebe6;padding-bottom:4px;margin-top:28px}}
.kp{{display:flex;flex-wrap:wrap;gap:10px}}.k{{flex:1;min-width:150px;background:#f3f7f5;border-radius:8px;padding:10px 12px}}
.v{{font-size:20px;font-weight:700}}.l{{font-size:12px;color:#5b636e}}
table.t{{border-collapse:collapse;width:100%;font-size:13px}}table.t th{{background:#2f6b4f;color:#fff;text-align:left;padding:6px}}
table.t td{{border-bottom:1px solid #e3e3e3;padding:5px 6px}}li{{margin:4px 0}}.note{{color:#666;font-size:12px}}</style></head><body>
<h1>ShelfSense Report</h1><div class='note'>Source: {R['source']} &middot; Data up to {R['meta']['end']:%d %b %Y}
&middot; Generated {pd.Timestamp.now():%d %b %Y %H:%M}{' &middot; <b>Stock figures are SIMULATED for demonstration</b>' if R.get('stock_simulated') else ''}</div>
<h2>Key figures</h2><div class='kp'>{kp}</div>
<h2>Key insights</h2><ul>{ins}</ul><h2>Recommended actions</h2><ul>{rec}</ul>
{charts}{sections}
<p class='note'>Recommendations are data-driven suggestions based on past sales, not guarantees.
Generated by ShelfSense (IBM SkillsBuild Data Analytics with AI Internship project, Vanshika Solanki).</p></body></html>"""


# ============================================================================
# 8. FULL PIPELINE
# ============================================================================
def run_pipeline(raw, mapping, opts, source, progress=None):
    t0 = time.time()
    say = progress or (lambda m: None)
    say("Checking data quality")
    quality = quality_report(raw, mapping)
    say("Cleaning data")
    sales, returns, log, extras = clean_data(raw, mapping, opts)
    if len(sales) < 50:
        raise ValueError("Fewer than 50 valid sales lines remain after cleaning.")
    say("Analysing products (ABC-XYZ)")
    products, meta = product_table(sales, returns, extras, opts["window_days"])
    say("Segmenting products (K-Means)")
    segments = segment_products(sales, products, meta)
    if segments:
        products = products.join(segments["features"]["segment"])
    span_days = (sales["date"].max() - sales["date"].min()).days
    dead = fc = None
    if span_days >= 2 * opts["horizon"] + 150:
        say("Training dead-stock risk models")
        dead = train_dead_stock_model(sales, extras, opts["horizon"], say)
    if span_days >= 180:
        fc = train_forecaster(sales, meta, say)
    w = sales[sales["date"] > meta["end"] - pd.Timedelta(days=opts["window_days"])]
    cur = opts.get("currency", "")
    ret_units = returns["qty"].sum() / sales["qty"].sum() * 100 if len(returns) else 0.0
    kpis = [("Revenue (window)", money_short(w["revenue"].sum(), cur)),
            ("Units sold (window)", money_short(w["qty"].sum(), "")),
            ("Products sold (window)", f"{w['key'].nunique():,}"),
            ("Orders (window)", f"{w['invoice'].nunique():,}" if extras["has_invoice"] else f"{len(w):,} lines"),
            ("Returned vs sold", f"{ret_units:.1f}%")]
    return {"sales": sales, "returns": returns, "log": log, "extras": extras, "quality": quality,
            "products": products, "meta": meta, "segments": segments, "dead": dead, "forecast": fc,
            "kpis": kpis, "opts": opts, "source": source, "mapping": mapping,
            "runtime_s": round(time.time() - t0, 1)}


# ============================================================================
# 9. CHARTS
# ============================================================================
GREEN, ORANGE, GREY = "#2f6b4f", "#d9822b", "#8a8f98"
ABC_COLORS = {"A": GREEN, "B": ORANGE, "C": GREY}
RISK_COLORS = {"High": "#b23a48", "Medium": ORANGE, "Low": GREEN}


def _style(fig, h=380):
    fig.update_layout(height=h, margin=dict(l=10, r=10, t=50, b=10), template="plotly_white",
                      legend=dict(orientation="h", y=-0.15))
    return fig


def fig_monthly(sales, cur):
    m = sales.groupby(sales["date"].dt.to_period("M")).agg(revenue=("revenue", "sum"), units=("qty", "sum"))
    m.index = m.index.to_timestamp()
    fig = px.bar(m, y="revenue", title="Monthly revenue", labels={"revenue": f"Revenue ({cur})", "date": ""},
                 color_discrete_sequence=[GREEN])
    return _style(fig)


def fig_top_products(products, cur, n=15):
    t = products.nlargest(n, "revenue_window").iloc[::-1]
    fig = px.bar(t, x="revenue_window", y="name", orientation="h", color="ABC", color_discrete_map=ABC_COLORS,
                 title=f"Top {n} products by revenue (analysis window)",
                 labels={"revenue_window": f"Revenue ({cur})", "name": ""})
    return _style(fig, 460)


def fig_pareto(products):
    a = products[products["revenue_window"] > 0].sort_values("revenue_window", ascending=False)
    y = a["revenue_window"].cumsum() / a["revenue_window"].sum() * 100
    x = np.arange(1, len(a) + 1) / len(a) * 100
    fig = go.Figure(go.Scatter(x=x, y=y, mode="lines", line=dict(color=GREEN, width=3), name="Cumulative revenue"))
    fig.add_hline(y=80, line_dash="dash", line_color=ORANGE, annotation_text="80% of revenue")
    fig.update_layout(title="Pareto curve: % of products vs % of revenue",
                      xaxis_title="% of products (best sellers first)", yaxis_title="% of revenue")
    return _style(fig)


def fig_abc_matrix(products):
    a = products[products["ABC"] != "-"]
    m = a.pivot_table(index="ABC", columns="XYZ", values="name", aggfunc="count").reindex(
        index=["A", "B", "C"], columns=["X", "Y", "Z"]).fillna(0)
    fig = px.imshow(m, text_auto=True, color_continuous_scale="Greens", aspect="auto",
                    title="ABC-XYZ matrix (number of products)",
                    labels=dict(x="Demand variability (X steady to Z erratic)", y="Value class", color="Products"))
    return _style(fig, 360)


def fig_weekday_hour(sales, cur):
    s = sales.assign(Day=sales["date"].dt.day_name().str[:3], Hour=sales["date"].dt.hour)
    m = s.pivot_table(index="Day", columns="Hour", values="revenue", aggfunc="sum").reindex(
        ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]).dropna(how="all")
    fig = px.imshow(m, color_continuous_scale="Greens", aspect="auto",
                    title="When do sales happen? (revenue by weekday and hour)",
                    labels=dict(color=f"Revenue ({cur})"))
    return _style(fig, 340)


# ============================================================================
# 10. STREAMLIT APP
# ============================================================================
def _file_key(data: bytes) -> str:
    return hashlib.md5(data).hexdigest()


@st.cache_resource(show_spinner=False, max_entries=2)
def _cached_read(key, name, _data):
    return read_table(_data, name)


@st.cache_resource(show_spinner=False, max_entries=1)
def _cached_demo():
    return load_demo_data()


def _mapping_form(columns, fields, detected, key_prefix):
    mapping = {}
    cols = st.columns(3)
    options = ["(not available)"] + list(columns)
    for i, (f, (label, req, _)) in enumerate(fields.items()):
        default = options.index(detected[f]) if f in detected else 0
        choice = cols[i % 3].selectbox(label + (" *" if req else ""), options, index=default,
                                       key=f"{key_prefix}_{f}")
        if choice != "(not available)":
            mapping[f] = choice
    return mapping


def app():
    st.set_page_config(page_title="ShelfSense", page_icon="📦", layout="wide")
    st.markdown("""<style>
      .block-container{padding-top:1.4rem}
      div[data-testid="stMetric"]{background:#f3f7f5;border-radius:10px;padding:10px 14px}
      .hero{background:linear-gradient(90deg,#2f6b4f,#3f8a66);color:#fff;border-radius:12px;padding:16px 22px;margin-bottom:10px}
      .hero h1{color:#fff;margin:0;font-size:2rem}.hero p{margin:2px 0 0;opacity:.92}
    </style>""", unsafe_allow_html=True)
    st.markdown("<div class='hero'><h1>📦 ShelfSense</h1><p>Dead-stock, expiry &amp; reorder analyzer "
                "for small shops. Upload your sales, get cleaning, analysis, AI predictions and an "
                "action report.</p></div>", unsafe_allow_html=True)

    # ---------------- Sidebar ----------------
    with st.sidebar:
        st.header("1. Data")
        src = st.radio("Sales data source", ["Upload my file", "Demo: UCI Online Retail II"],
                       help="Demo = 1 million real transactions of a UK gift wholesaler (2009-2011).")
        raw, file_key, source = None, None, None
        if src == "Upload my file":
            up = st.file_uploader("Sales file (CSV or Excel)", type=["csv", "xlsx", "xls"])
            if up is not None:
                data = up.getvalue()
                file_key, source = _file_key(data), up.name
                try:
                    with st.spinner("Reading file..."):
                        raw = _cached_read(file_key, up.name, data)
                except Exception as e:
                    st.error(f"Could not read the file: {e}")
        else:
            try:
                with st.spinner("Loading demo dataset (first time: about 1-2 minutes)..."):
                    raw = _cached_demo()
                file_key, source = "demo-online-retail-ii", "UCI Online Retail II (demo)"
            except FileNotFoundError as e:
                st.error(str(e))
        st.header("2. Settings")
        currency = st.selectbox("Currency", ["₹", "£", "$", "€"],
                                index=1 if src.startswith("Demo") else 0)
        window_days = st.select_slider("Analysis window", [90, 180, 365], value=365,
                                       format_func=lambda d: f"Last {d} days")
        dead_days = st.slider("Dead stock = no sale for (days)", 30, 180, 90, 15)
        horizon = st.select_slider("AI: predict dead stock over next", [30, 60, 90], value=60,
                                   format_func=lambda d: f"{d} days")
        lead_time = st.slider("Supplier lead time (days)", 1, 60, 7)
        review_days = st.slider("Order review cycle (days)", 1, 30, 7)
        service = st.select_slider("Target service level", [0.80, 0.85, 0.90, 0.95, 0.98, 0.99],
                                   value=0.95, format_func=lambda v: f"{v:.0%}")
        date_format = st.selectbox("Date format", ["Auto-detect", "Day-first (DD/MM/YYYY)",
                                                   "Month-first (MM/DD/YYYY)"])
        with st.expander("Cleaning options"):
            rm_dup = st.checkbox("Remove exact duplicate lines", True)
            rm_np = st.checkbox("Remove non-product lines (postage, fees)", True)
            cap = st.checkbox("Cap extreme bulk orders for demand planning", True)
        st.caption("Built by **Vanshika Solanki** · IBM SkillsBuild Data Analytics with AI "
                   "Internship 2026 (BharatCares × AICTE) · Supports UN SDG 12: Responsible "
                   "consumption & production.")

    if raw is None:
        st.info("⬅️ Upload a sales file (CSV/Excel) or choose the demo dataset in the sidebar to begin.")
        c1, c2, c3 = st.columns(3)
        c1.markdown("**What you need**\n- One row per item sold\n- Date, product, quantity\n"
                    "- Optional: price, bill no., customer, branch")
        c2.markdown("**What you get**\n- Data quality check & cleaning log\n- ABC-XYZ analysis\n"
                    "- AI dead-stock risk & demand forecast\n- Reorder / discount / delist lists")
        c3.markdown("**Privacy**\n- Everything runs on this computer\n- No data is uploaded to any "
                    "external service")
        return

    tabs = st.tabs(["📥 Data", "🧹 Quality & Cleaning", "📊 Dashboard", "🔤 ABC-XYZ",
                    "🤖 AI Insights", "📦 Reorder & Stock", "🔮 What-if", "📄 Report"])

    # ---------------- Tab 1: data & mapping ----------------
    with tabs[0]:
        st.subheader("Your data")
        c1, c2 = st.columns(2)
        c1.metric("Rows", f"{len(raw):,}")
        c2.metric("Columns", f"{raw.shape[1]}")
        st.dataframe(raw.head(200), width="stretch", height=240)
        st.subheader("Column mapping")
        st.caption("ShelfSense guessed which column is which. Correct it if needed. "
                   "* = required. You need a product code **or** a product name.")
        mapping = _mapping_form(raw.columns, FIELDS, auto_map(raw.columns), "map")
        ok = "date" in mapping and "quantity" in mapping and \
            ("product_code" in mapping or "product_name" in mapping)
        if len(set(mapping.values())) < len(mapping):
            st.error("The same column is selected for two fields.")
            ok = False
        opts = {"window_days": window_days, "dead_days": dead_days, "horizon": horizon,
                "date_format": date_format, "remove_duplicates": rm_dup,
                "remove_non_products": rm_np, "cap_outliers": cap, "currency": currency}
        run_key = (file_key, json.dumps(mapping, sort_keys=True), json.dumps(opts, sort_keys=True))
        if len(raw) > 200_000:
            st.caption(f"⏱️ {len(raw):,} rows: the first analysis takes about 1-3 minutes "
                       "(training the AI models). After that, every tab and slider responds instantly.")
        if st.button("🚀 Clean & analyse", type="primary", disabled=not ok):
            st.session_state["run_key"] = run_key
        if not ok:
            st.warning("Map the required fields to continue.")

    R = None
    if st.session_state.get("run_key") and st.session_state["run_key"][0] == file_key:
        rk = st.session_state["run_key"]
        with tabs[0]:
            # Results are kept in the session, so changing a tab or a slider does not
            # re-run the whole analysis.
            if st.session_state.get("results_key") == rk:
                R = st.session_state["results"]
                st.success(f"Analysis ready (took {R['runtime_s']} s).")
            else:
                with st.status("Running analysis...", expanded=True) as status:
                    try:
                        R = run_pipeline(raw, json.loads(rk[1]), json.loads(rk[2]), source, status.write)
                        st.session_state["results"], st.session_state["results_key"] = R, rk
                        status.update(label=f"Analysis ready ({R['runtime_s']} s)", state="complete",
                                      expanded=False)
                    except Exception as e:
                        status.update(label="Analysis failed", state="error")
                        st.error(f"Analysis failed: {e}")
            if rk != run_key:
                st.info("Settings or mapping changed. Press **Clean & analyse** again to update.")
    if R is None:
        for t in tabs[1:]:
            with t:
                st.info("Press **🚀 Clean & analyse** on the Data tab first.")
        return

    cur = R["opts"]["currency"]
    products, sales, meta = R["products"], R["sales"], R["meta"]

    # ---------------- Tab 2: quality & cleaning ----------------
    with tabs[1]:
        ex = R["extras"]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Rows before", f"{ex['rows_before']:,}")
        c2.metric("Clean sales rows", f"{ex['rows_after']:,}",
                  f"{ex['rows_after'] - ex['rows_before']:,}")
        c3.metric("Return lines kept aside", f"{ex['returns_rows']:,}")
        c4.metric("Bulk orders flagged", f"{ex['outliers']:,}")
        st.subheader("Issues found before cleaning")
        st.dataframe(R["quality"], width="stretch", hide_index=True)
        st.subheader("Cleaning log: what was done and why")
        st.dataframe(R["log"], width="stretch", hide_index=True)
        if len(ex["dupe_names"]):
            with st.expander(f"⚠️ {len(ex['dupe_names'])} product names appear under more than one code "
                             "(possible duplicates: check manually)"):
                st.dataframe(ex["dupe_names"].head(200), width="stretch", hide_index=True)
        st.download_button("⬇️ Download cleaned sales data (CSV)",
                           sales.drop(columns=["week"]).to_csv(index=False).encode(),
                           "shelfsense_cleaned_sales.csv", "text/csv")

    # ---------------- Tab 3: dashboard ----------------
    with tabs[2]:
        cols = st.columns(3)
        for i, (label, val) in enumerate(R["kpis"]):
            cols[i % 3].metric(label, val)
        view = sales
        if ex["has_location"]:
            locs = sales["location"].value_counts().index.tolist()
            pick = st.multiselect("Filter by country / branch", locs, placeholder="All")
            if pick:
                view = sales[sales["location"].isin(pick)]
        st.plotly_chart(fig_monthly(view, cur), width="stretch")
        c1, c2 = st.columns(2)
        with c1:
            vp = products if view is sales else product_table(view, R["returns"], ex, R["opts"]["window_days"])[0]
            st.plotly_chart(fig_top_products(vp, cur), width="stretch")
        with c2:
            if ex["has_time"]:
                st.plotly_chart(fig_weekday_hour(view, cur), width="stretch")
            if ex["has_location"]:
                loc = view.groupby("location")["revenue"].sum().nlargest(10).iloc[::-1]
                st.plotly_chart(_style(px.bar(loc, orientation="h", title="Top locations by revenue",
                                              color_discrete_sequence=[GREEN],
                                              labels={"value": f"Revenue ({cur})", "location": ""}), 320),
                                width="stretch")
        rt = products[products["units_all_time"] >= 50].nlargest(10, "return_rate_pct")
        if len(rt) and rt["return_rate_pct"].max() > 0:
            st.subheader("Most returned products (min. 50 units sold)")
            st.dataframe(rt[["name", "units_all_time", "units_returned", "return_rate_pct"]],
                         width="stretch")

    # ---------------- Tab 4: ABC-XYZ ----------------
    with tabs[3]:
        st.caption("**ABC** ranks products by revenue: A = the few that earn 80%, B = next 15%, C = last 5%. "
                   "**XYZ** measures how predictable weekly demand is: X steady (CV < 0.5), "
                   "Y variable (0.5-1.0), Z erratic (> 1.0).")
        c1, c2 = st.columns(2)
        c1.plotly_chart(fig_pareto(products), width="stretch")
        c2.plotly_chart(fig_abc_matrix(products), width="stretch")
        a = products[products["ABC"] != "-"]
        summ = a.groupby("class").agg(products=("name", "size"), revenue=("revenue_window", "sum"))
        summ["revenue_share_%"] = (summ["revenue"] / summ["revenue"].sum() * 100).round(1)
        summ["strategy"] = summ.index.map(ABC_STRATEGY)
        st.dataframe(summ.drop(columns="revenue"), width="stretch")
        cls = st.multiselect("Show products in classes", sorted(summ.index), default=["AX"])
        show = products[products["class"].isin(cls)].sort_values("revenue_window", ascending=False)
        st.dataframe(show[["name", "class", "revenue_window", "units_window", "weekly_mean", "cv",
                           "days_since_last_sale"]].round(2), width="stretch")

    # ---------------- Tab 5: AI insights ----------------
    with tabs[4]:
        sub = st.tabs(["☠️ Dead-stock risk", "📈 Demand forecast", "🧩 Product segments"])
        with sub[0]:
            d = R["dead"]
            if not d:
                st.warning("Not enough history for the dead-stock model (needs about "
                           f"{2 * R['opts']['horizon'] + 150} days of sales).")
            else:
                st.markdown(f"**Question:** will an active product sell **nothing** in the next "
                            f"**{d['horizon']} days**? The models learn from past snapshots "
                            f"({d['n_train']:,} examples) and are tested on a later, unseen period "
                            f"(snapshot {d['val_cutoff']:%d %b %Y}, {d['n_val']:,} products; "
                            f"{d['base_rate']:.0%} actually went dead).")
                st.dataframe(d["metrics"], width="stretch", hide_index=True)
                st.success(f"Selected model: **{d['best']}** (highest ROC-AUC on the unseen period).")
                c1, c2 = st.columns(2)
                imp = d["importance"].iloc[::-1]
                c1.plotly_chart(_style(px.bar(imp, x="importance", y="feature", orientation="h",
                                              title="What drives the prediction (permutation importance)",
                                              color_discrete_sequence=[GREEN]), 400), width="stretch")
                sc = d["scores"].join(products[["name", "ABC", "days_since_last_sale", "units_window"]])
                c2.plotly_chart(_style(px.histogram(sc, x="dead_risk", color="risk_level", nbins=30,
                                                    color_discrete_map=RISK_COLORS,
                                                    title="Current dead-stock risk of active products"), 400),
                                width="stretch")
                lvl = st.radio("Show", ["High", "Medium", "Low"], horizontal=True)
                st.dataframe(sc[sc["risk_level"] == lvl].sort_values("dead_risk", ascending=False)
                             .round(3).head(300), width="stretch")
        with sub[1]:
            f = R["forecast"]
            if not f:
                st.warning("Not enough weekly history to train a forecast (needs about 6 months).")
            else:
                st.markdown(f"Forecasts **next-4-week demand** for {f['n_products']:,} regularly sold "
                            "products. Four methods were tested on the last 8 weeks, which the models "
                            "never saw. The most accurate one is used.")
                st.dataframe(f["metrics"], width="stretch", hide_index=True)
                st.success(f"Method used for planning: **{f['best']}**. Rule: methods within 1 point of the "
                           "best WAPE count as equally accurate, and the least biased one is chosen. "
                           "WAPE = total absolute error ÷ total actual demand; Bias = over (+) or under (−) "
                           "forecasting.")
                keys = f["forecast"].join(products["revenue_window"]).sort_values(
                    "revenue_window", ascending=False).index
                label = {k: f"{k} · {products.at[k, 'name']}" for k in keys[:500]}
                k = st.selectbox("Product", keys[:500], format_func=label.get)
                hist = f["grid"][f["grid"]["key"] == k].set_index("week")["qty"]
                fut = pd.date_range(hist.index.max() + pd.Timedelta(weeks=1), periods=4, freq="7D")
                wk_fc = f["forecast"].at[k, "forecast_4w"] / 4
                fig = go.Figure([go.Scatter(x=hist.index, y=hist.values, name="Actual weekly units",
                                            line=dict(color=GREEN)),
                                 go.Scatter(x=fut, y=[wk_fc] * 4, name="Forecast (weekly avg)",
                                            line=dict(color=ORANGE, dash="dash", width=3))])
                fig.update_layout(title=f"{products.at[k, 'name']}: weekly demand and forecast")
                st.plotly_chart(_style(fig), width="stretch")
                st.metric("Forecast demand, next 4 weeks", f"{f['forecast'].at[k, 'forecast_4w']:,.0f} units")
        with sub[2]:
            sg = R["segments"]
            if not sg:
                st.warning("Too few products to segment.")
            else:
                st.markdown(f"K-Means grouped active products into 4 behaviour segments "
                            f"(silhouette score {sg['silhouette']}: higher means better separated).")
                st.dataframe(sg["summary"], width="stretch")
                ff = sg["features"].join(products[["name"]])
                st.plotly_chart(_style(px.scatter(ff, x="active_week_share", y="log_revenue", color="segment",
                                                  hover_name="name", opacity=0.6,
                                                  title="Product segments: how often vs how much they sell",
                                                  labels={"active_week_share": "Share of weeks with a sale",
                                                          "log_revenue": "log(revenue)"}), 460),
                                width="stretch")

    # ---------------- Tab 6: reorder & stock ----------------
    with tabs[5]:
        st.caption(f"Lead time **{lead_time} days** · review cycle **{review_days} days** · service level "
                   f"**{service:.0%}** (change these in the sidebar). Safety stock = z × σ(daily demand) × "
                   "√lead time; Reorder point = daily demand × lead time + safety stock.")
        st.markdown("#### Current stock (optional, unlocks reorder quantities, overstock and expiry checks)")
        sfile = st.file_uploader("Stock sheet (CSV/Excel): product code or name, current stock, "
                                 "optional unit cost and expiry date", type=["csv", "xlsx", "xls"],
                                 key="stock_up")
        stock, simulated = None, False
        if sfile is not None:
            sraw = read_table(sfile.getvalue(), sfile.name)
            smap = _mapping_form(sraw.columns, STOCK_FIELDS, auto_map(sraw.columns, STOCK_FIELDS), "smap")
            if "product" in smap and "stock" in smap:
                stock, unmatched = parse_stock_sheet(sraw, smap, products, "product_code" in R["mapping"])
                st.caption(f"Matched {len(stock):,} products · {unmatched:,} stock rows did not match any "
                           "product in the sales data.")
        elif source and source.startswith("UCI"):
            if st.toggle("Use a SIMULATED stock sheet for the demo (the UCI dataset has no stock data)"):
                stock, simulated = simulate_demo_stock(products, meta), True
                st.warning("⚠️ Stock, cost and expiry values are **randomly simulated** to demonstrate "
                           "the features. They are not real data.")
        R["stock_simulated"] = simulated
        plan = build_plan(products, R["forecast"], R["dead"], sales, meta, lead_time, review_days,
                          service, R["opts"]["dead_days"], stock)
        st.session_state["plan"] = plan
        tmpl = plan[["name"]].reset_index().rename(columns={"key": "product_code"}).assign(
            current_stock="", unit_cost="", expiry_date="")
        st.download_button("⬇️ Download a stock-sheet template with your products",
                           tmpl.to_csv(index=False).encode(), "stock_template.csv", "text/csv")
        if "status" in plan:
            counts = plan["status"].value_counts()
            c = st.columns(4)
            for i, s in enumerate(["Reorder now", "Dead / slow stock", "Overstocked", "OK"]):
                c[i].metric(s, f"{counts.get(s, 0):,}")
            c = st.columns(3)
            c[0].metric("Stock value", money_short(plan["stock_value"].sum(), cur))
            c[1].metric("Locked in dead / slow stock",
                        money_short(plan.loc[plan["status"] == "Dead / slow stock", "stock_value"].sum(), cur))
            if "value_at_expiry_risk" in plan:
                c[2].metric("Value at expiry risk", money_short(plan["value_at_expiry_risk"].sum(), cur))
            view = st.radio("List", ["Reorder now", "Dead / slow stock", "Overstocked", "Expiry risk", "All"],
                            horizontal=True)
            if view == "Expiry risk" and "units_at_expiry_risk" in plan:
                t = plan[plan["units_at_expiry_risk"] > 0].sort_values("value_at_expiry_risk", ascending=False)
                cols = ["name", "current_stock", "days_to_expiry", "daily_demand", "units_at_expiry_risk",
                        "value_at_expiry_risk", "expiry_action"]
            else:
                t = plan if view in ("All", "Expiry risk") else plan[plan["status"] == view]
                t = t.sort_values("stock_value", ascending=False)
                cols = ["name", "ABC", "risk_level", "current_stock", "daily_demand", "days_of_cover",
                        "reorder_point", "suggested_order_qty", "stock_value", "status"]
            st.dataframe(t[cols].round(2), width="stretch")
        else:
            st.info("No stock sheet yet. Showing reorder levels computed from sales alone.")
            st.dataframe(plan[["name", "ABC", "XYZ", "risk_level", "forecast_4w", "daily_demand",
                               "safety_stock", "reorder_point", "order_up_to", "forecast_method"]]
                         .sort_values("forecast_4w", ascending=False).round(2), width="stretch")

    # ---------------- Tab 7: what-if ----------------
    with tabs[6]:
        st.caption("Model-based scenario estimates to support decisions. They are not guaranteed outcomes.")
        pl = st.session_state["plan"]
        cand = pl[pl["forecast_4w"] > 0].sort_values("forecast_4w", ascending=False)
        if cand.empty:
            st.info("No products with recent demand.")
        else:
            k = st.selectbox("Product", cand.index[:500],
                             format_func=lambda x: f"{x} · {cand.at[x, 'name']}", key="wi_prod")
            r = cand.loc[k]
            c1, c2, c3 = st.columns(3)
            dchg = c1.slider("Demand change (%)", -50, 100, 0, 5)
            lt = c2.slider("Lead time (days)", 1, 60, lead_time, key="wi_lt")
            sl = c3.select_slider("Service level", [0.80, 0.85, 0.90, 0.95, 0.98, 0.99], value=service,
                                  format_func=lambda v: f"{v:.0%}", key="wi_sl")

            def calc(dem, L, s):
                zz = NormalDist().inv_cdf(s)
                sd = r["weekly_std"] / np.sqrt(7) * max(dem / max(r["daily_demand"], 1e-9), 0)
                ss = np.ceil(zz * sd * np.sqrt(L))
                return ss, np.ceil(dem * L + ss), np.ceil(dem * (L + review_days) + ss)
            base = calc(r["daily_demand"], lead_time, service)
            new = calc(r["daily_demand"] * (1 + dchg / 100), lt, sl)
            st.table(pd.DataFrame({"Current settings": base, "Scenario": new},
                                  index=["Safety stock", "Reorder point", "Order-up-to level"]).astype(int))
            if "current_stock" in pl:
                ndd = r["daily_demand"] * (1 + dchg / 100)
                st.metric("Days of cover in scenario", f"{r['current_stock'] / ndd:,.0f} days" if ndd > 0 else "∞",
                          f"{r['current_stock'] / ndd - (r['days_of_cover'] if pd.notna(r['days_of_cover']) else 0):+,.0f} days"
                          if ndd > 0 else None)
            extra = (new[0] - base[0]) * r["avg_price"]
            st.caption(f"Extra safety stock value in the scenario: about {money(extra, cur)} "
                       "(at selling price).")
        if "status" in pl:
            st.divider()
            st.markdown("#### Clearance sale for dead / slow stock")
            dead_v = pl.loc[pl["status"] == "Dead / slow stock", "stock_value"].sum()
            c1, c2 = st.columns(2)
            disc = c1.slider("Discount (%)", 0, 70, 30, 5)
            sell = c2.slider("Share of this stock you expect to sell (%)", 0, 100, 60, 5)
            cash = dead_v * (1 - disc / 100) * sell / 100
            st.metric("Cash released (estimate)", money_short(cash, cur),
                      f"from {money_short(dead_v, cur)} locked in dead / slow stock")

    # ---------------- Tab 8: report ----------------
    with tabs[7]:
        pl = st.session_state["plan"]
        st.subheader("Key insights")
        for i in generate_insights(R, cur):
            st.markdown(f"- {i}")
        st.subheader("Recommended actions")
        for i in plan_recommendations(pl, cur):
            st.markdown(f"- {i}")
        if R.get("stock_simulated"):
            st.warning("Stock-based figures use SIMULATED demo stock.")
        figs = [fig_monthly(sales, cur), fig_pareto(products), fig_abc_matrix(products)]
        c1, c2, c3 = st.columns(3)
        c1.download_button("⬇️ HTML report (open in browser → Print → Save as PDF)",
                           build_html_report(R, pl, cur, figs).encode("utf-8"),
                           "ShelfSense_Report.html", "text/html", type="primary")
        c2.download_button("⬇️ Excel workbook (all tables)", build_excel(R, pl, cur),
                           "ShelfSense_Results.xlsx",
                           "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        c3.download_button("⬇️ Cleaned sales (CSV)", sales.drop(columns=["week"]).to_csv(index=False).encode(),
                           "shelfsense_cleaned_sales.csv", "text/csv", key="dl2")


# ============================================================================
# 11. COMMAND-LINE MODE (python VanshikaSolanki_ShelfSense.py)
# ============================================================================
def cli():
    ap = argparse.ArgumentParser(description="ShelfSense command-line analysis")
    ap.add_argument("--data", help="sales CSV/Excel (default: demo dataset in data/)")
    ap.add_argument("--currency", default=None)
    ap.add_argument("--simulate-stock", action="store_true",
                    help="demo only: add a SIMULATED stock sheet to show stock features")
    a = ap.parse_args()
    print("ShelfSense - command-line mode")
    print("Tip: for the interactive app run:  streamlit run VanshikaSolanki_ShelfSense.py\n")
    try:
        if a.data:
            raw = read_table(Path(a.data).read_bytes(), a.data)
            source = Path(a.data).name
        else:
            raw = load_demo_data()
            source = "UCI Online Retail II (demo)"
    except (FileNotFoundError, ValueError) as e:
        sys.exit(f"ERROR: {e}")
    cur = a.currency or ("£" if not a.data else "₹")
    mapping = auto_map(raw.columns)
    print(f"Loaded {len(raw):,} rows from {source}\nColumn mapping: {mapping}")
    opts = {"window_days": 365, "dead_days": 90, "horizon": 60, "date_format": "Auto-detect",
            "remove_duplicates": True, "remove_non_products": True, "cap_outliers": True,
            "currency": cur}
    R = run_pipeline(raw, mapping, opts, source, progress=lambda m: print(f"  ... {m}"))
    stock = simulate_demo_stock(R["products"], R["meta"]) if a.simulate_stock else None
    R["stock_simulated"] = stock is not None
    plan = build_plan(R["products"], R["forecast"], R["dead"], R["sales"], R["meta"], 7, 7, 0.95, 90, stock)

    print("\n=== DATA QUALITY (before cleaning) ===\n" + R["quality"].to_string(index=False))
    print("\n=== CLEANING LOG ===\n" + R["log"][["Step", "Issue", "Rows affected", "Action"]].to_string(index=False))
    print(f"\nRows: {R['extras']['rows_before']:,} -> {R['extras']['rows_after']:,} clean sales lines")
    print("\n=== KPIs ===")
    for l, v in R["kpis"]:
        print(f"  {l:<36}{v}")
    abc = R["products"][R["products"]["ABC"] != "-"]["class"].value_counts().sort_index()
    print("\n=== ABC-XYZ (products) ===\n" + abc.to_string())
    if R["segments"]:
        print(f"\n=== SEGMENTS (silhouette {R['segments']['silhouette']}) ===\n" + R["segments"]["summary"].to_string())
    if R["dead"]:
        print(f"\n=== DEAD-STOCK MODEL (horizon {R['dead']['horizon']} days, train {R['dead']['n_train']:,}, "
              f"validation {R['dead']['n_val']:,}, base rate {R['dead']['base_rate']:.1%}) ===")
        print(R["dead"]["metrics"].to_string(index=False) + f"\nSelected: {R['dead']['best']}")
        print(R["dead"]["importance"].round(4).to_string(index=False))
        print(R["dead"]["scores"]["risk_level"].value_counts().to_string())
    if R["forecast"]:
        print(f"\n=== DEMAND FORECAST ({R['forecast']['n_products']:,} products) ===\n"
              + R["forecast"]["metrics"].to_string(index=False) + f"\nSelected: {R['forecast']['best']}")
    print("\n=== INSIGHTS ===")
    for i in generate_insights(R, cur) + plan_recommendations(plan, cur):
        print(" - " + i)

    OUTPUT_DIR.mkdir(exist_ok=True)
    figs = [fig_monthly(R["sales"], cur), fig_pareto(R["products"]), fig_abc_matrix(R["products"])]
    (OUTPUT_DIR / "ShelfSense_Report.html").write_text(build_html_report(R, plan, cur, figs), encoding="utf-8")
    (OUTPUT_DIR / "ShelfSense_Results.xlsx").write_bytes(build_excel(R, plan, cur))
    summary = {
        "source": source, "rows_before": R["extras"]["rows_before"], "rows_after": R["extras"]["rows_after"],
        "returns_rows": R["extras"]["returns_rows"], "outliers": R["extras"]["outliers"],
        "date_range": [str(R["sales"]["date"].min()), str(R["sales"]["date"].max())],
        "quality": R["quality"].to_dict("records"), "cleaning_log": R["log"].to_dict("records"),
        "kpis": dict(R["kpis"]), "abc_xyz_counts": abc.to_dict(),
        "segments": R["segments"]["summary"].reset_index().to_dict("records") if R["segments"] else None,
        "silhouette": R["segments"]["silhouette"] if R["segments"] else None,
        "dead_stock": {k: (v.to_dict("records") if isinstance(v, pd.DataFrame) else v)
                       for k, v in R["dead"].items() if k not in ("scores",)} if R["dead"] else None,
        "dead_risk_counts": R["dead"]["scores"]["risk_level"].value_counts().to_dict() if R["dead"] else None,
        "forecast": {"metrics": R["forecast"]["metrics"].to_dict("records"), "best": R["forecast"]["best"],
                     "n_products": R["forecast"]["n_products"], "n_train_rows": R["forecast"]["n_train_rows"],
                     "test_origins": [str(t) for t in R["forecast"]["test_origins"]]} if R["forecast"] else None,
        "insights": generate_insights(R, cur), "recommendations": plan_recommendations(plan, cur),
        "runtime_s": R["runtime_s"],
    }
    (OUTPUT_DIR / "results_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(f"\nSaved: outputs/ShelfSense_Report.html, outputs/ShelfSense_Results.xlsx, "
          f"outputs/results_summary.json  (pipeline {R['runtime_s']} s)")


def _running_in_streamlit():
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        return get_script_run_ctx(suppress_warning=True) is not None
    except Exception:
        return False


if _running_in_streamlit():
    app()
elif __name__ == "__main__":
    cli()
