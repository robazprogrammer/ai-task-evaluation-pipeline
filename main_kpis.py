import os
import math
import argparse
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -------------------------
# Config / constants
# -------------------------
SALES_FILE = "2025 YTD.csv"
FREDDIE_FILE = "freddiemac.csv"
OUT_DIR = "out"

# Only lenient for a few common cases
COLMAP = {
    "coe": "COE",
    "close of escrow date": "COE",
    "soldprice": "Sold Price",
    "zip": "Zip Code",
}

PRICE_BANDS = [
    ("500k and Under", 0, 500_000),
    ("500,001 to 800k", 500_001, 800_000),
    ("801k and Above", 800_001, np.inf),
]

# KPI requirements (used for validation + messaging)
KPI_REQUIREMENTS = {
    "sales_counts_monthly": {"COE"},
    "median_price_by_city_month": {"COE", "City", "Sold Price"},
    "sale_to_list_by_city_month": {"COE", "City", "Sold Price", "OLP"},
    "top10_zip_sales_by_month": {"COE", "Zip Code"},
    "price_band_share_and_p66_by_month": {"COE", "Sold Price"},
    "avg_dom_by_price_reduction_flag": {"OLP", "FLP", "DOM"},
    "avg_dom_by_price_reduction_bins": {"OLP", "FLP", "DOM"},

    # YTD items
    "top10_zip_sales_ytd": {"COE", "Zip Code"},
    "top10_zip_appreciation_ytd": {"COE", "Zip Code", "Sold Price"},
    "top10_city_sales_ytd": {"COE", "City"},
    "top10_city_appreciation_ytd": {"COE", "City", "Sold Price"},
    "top10_city_median_price_ytd": {"COE", "City", "Sold Price"},
    "ytd_sale_to_list_overall": {"COE", "Sold Price", "OLP"},
    "ytd_sale_to_list_by_city": {"COE", "City", "Sold Price", "OLP"},
}

FOOTER_SOURCE = f"Source: ARMLS, {datetime.now():%Y-%m-%d} · Graphic: NhanceData"

# -------------------------
# IO helpers
# -------------------------
def read_csv_fallback(path: str) -> pd.DataFrame:
    for enc in ("utf-8", "ISO-8859-1", "Windows-1252", "latin1", "cp1252"):
        try:
            return pd.read_csv(path, encoding=enc, low_memory=False)
        except Exception:
            pass
    raise RuntimeError(f"Failed to read {path} with common encodings")

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    mapping = {}
    for c in df.columns:
        key = c.strip().lower()
        mapping[c] = COLMAP.get(key, c.strip())
    df = df.rename(columns=mapping)
    for c in df.select_dtypes(include=["object"]).columns:
        df[c] = df[c].astype(str).str.strip()
    return df

def ts_stamp_include_time(with_time: bool) -> str:
    return datetime.now().strftime("%Y-%m-%d_%H%M%S") if with_time else datetime.now().strftime("%Y-%m-%d")

def safe_make_outdir(path: str):
    os.makedirs(path, exist_ok=True)

# -------------------------
# Feature engineering
# -------------------------
def synthesize_pool_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create unified 'Pool' from 'Private Pool Y/N' and 'Community Pool Y/N' when present.
    Values: 'Both', 'None', 'Private', 'Community'.
    If those columns are absent, preserve legacy 'Pool' if it exists; otherwise add a NaN placeholder.
    """
    lower_map = {c.strip().lower(): c for c in df.columns}
    priv_col = lower_map.get("private pool y/n")
    comm_col = lower_map.get("community pool y/n")

    def _as_yes(s: pd.Series) -> pd.Series:
        if s is None:
            return pd.Series([False] * len(df))
        v = s.astype(str).str.strip().str.upper()
        return v.isin({"Y", "YES", "TRUE", "1"})

    if priv_col and comm_col:
        p = _as_yes(df[priv_col])
        c = _as_yes(df[comm_col])
        df["Pool"] = np.select(
            [p & c, (~p) & (~c), p & (~c), (~p) & c],
            ["Both", "None", "Private", "Community"],
            default=np.nan,
        )
    else:
        if "Pool" not in df.columns:
            df["Pool"] = np.nan
    return df

def extract_loan_type_from_features(df: pd.DataFrame) -> pd.DataFrame:
    if "Features" not in df.columns:
        df["LoanType"] = np.nan
        return df

    def _extract(val: str):
        if val is None or (isinstance(val, float) and math.isnan(val)):
            return None
        for p in str(val).split(";"):
            p = p.strip()
            if p.startswith("Sold Info|Loan Type"):
                parts = p.split("|")
                return parts[-1].strip() if parts else None
        return None

    df["LoanType"] = df["Features"].apply(_extract)
    return df

def compute_interest_from_freddie(df: pd.DataFrame, freddie_path: str) -> pd.DataFrame:
    """Assign weekly Freddie Mac rate by as-of merge (backward)."""
    if "COE" not in df.columns:
        df["Int"] = np.nan
        return df

    freddie = read_csv_fallback(freddie_path)
    if "Week" not in freddie.columns or "Rate" not in freddie.columns:
        raise ValueError("freddiemac.csv must contain 'Week' and 'Rate' columns")

    freddie = freddie.copy()
    freddie["Week"] = pd.to_datetime(freddie["Week"], errors="coerce")
    freddie = freddie.dropna(subset=["Week", "Rate"]).sort_values("Week")

    sales = df.sort_values("COE")
    merged = pd.merge_asof(
        sales,
        freddie[["Week", "Rate"]].rename(columns={"Week": "WeekRef", "Rate": "BaseRate"}),
        left_on="COE",
        right_on="WeekRef",
        direction="backward",
        tolerance=pd.Timedelta(days=14),  # reasonable max distance to prior Freddie print
    )

    def _rate_adjust(row):
        r = row.get("BaseRate")
        lt = (row.get("LoanType") or "").strip()
        if pd.isna(r):
            return np.nan
        if lt == "Cash":
            return 0.0
        if lt == "Conventional":
            return r
        if lt in ("FHA", "VA"):
            return r - 0.50
        return r + 4.00  # other/unknown
    merged["Int"] = merged.apply(_rate_adjust, axis=1)
    merged = merged.drop(columns=["WeekRef"], errors="ignore")
    return merged

# -------------------------
# QC
# -------------------------
def qc_checks(df: pd.DataFrame) -> dict:
    out = {
        "rows": len(df),
        "missing_coe": int(df["COE"].isna().sum()) if "COE" in df.columns else None,
        "missing_soldprice": int(df["Sold Price"].isna().sum()) if "Sold Price" in df.columns else None,
    }
    if "APN" in df.columns:
        out["duplicate_apn_count"] = int(df["APN"].duplicated().sum())
    return out

# -------------------------
# Common helpers
# -------------------------
def ytd_filter(df: pd.DataFrame) -> pd.DataFrame:
    """Restrict to YTD for the *max year* present in COE."""
    tmp = df.dropna(subset=["COE"]).copy()
    if tmp.empty:
        return tmp
    max_year = int(tmp["COE"].dt.year.max())
    start = pd.Timestamp(f"{max_year}-01-01")
    end = tmp["COE"].max()
    return tmp[(tmp["COE"] >= start) & (tmp["COE"] <= end)]

def jan_and_latest_month_groups(df: pd.DataFrame, group_col: str):
    """Return (jan_grouped, latest_grouped, latest_month_number)."""
    tmp = ytd_filter(df)
    if tmp.empty:
        return pd.DataFrame(), pd.DataFrame(), None
    tmp["Month"] = tmp["COE"].dt.month
    latest_month = int(tmp["Month"].max())
    jan = tmp[tmp["Month"] == 1]
    latest = tmp[tmp["Month"] == latest_month]
    jan_med = jan.groupby(group_col)["Sold Price"].median().rename("JanMedian")
    latest_med = latest.groupby(group_col)["Sold Price"].median().rename("LatestMedian")
    jan_cnt = jan.groupby(group_col).size().rename("JanCount")
    latest_cnt = latest.groupby(group_col).size().rename("LatestCount")
    ytd_cnt = tmp.groupby(group_col).size().rename("YTDSales")
    out = pd.concat([jan_med, latest_med, jan_cnt, latest_cnt, ytd_cnt], axis=1).reset_index()
    return out, latest, latest_month

# -------------------------
# KPIs (monthly)
# -------------------------
def kpi_sales_counts(df: pd.DataFrame) -> pd.DataFrame:
    tmp = df.dropna(subset=["COE"]).copy()
    tmp["YearMonth"] = tmp["COE"].dt.to_period("M").dt.to_timestamp()
    return tmp.groupby("YearMonth").size().reset_index(name="SalesCount")

def kpi_median_price_by_city_month(df: pd.DataFrame) -> pd.DataFrame:
    need = {"COE", "City", "Sold Price"}
    if not need.issubset(df.columns):
        return pd.DataFrame()
    tmp = df.dropna(subset=list(need)).copy()
    tmp["YearMonth"] = tmp["COE"].dt.to_period("M").dt.to_timestamp()
    return tmp.groupby(["City", "YearMonth"])["Sold Price"].median().reset_index(name="MedianSoldPrice")

def _filter_ratio_bounds(df: pd.DataFrame, ratio_col: str, low: float, high: float):
    """Keep rows with low <= ratio_col <= high. Returns (filtered_df, outliers_df)."""
    if df is None or df.empty or ratio_col not in df.columns:
        return df, pd.DataFrame()
    mask = df[ratio_col].between(low, high, inclusive="both")
    return df[mask].copy(), df[~mask].copy()

def kpi_sale_to_list_by_city_month(df: pd.DataFrame, ratio_min: float, ratio_max: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    need = {"COE", "City", "Sold Price", "OLP"}
    if not need.issubset(df.columns):
        return pd.DataFrame(), pd.DataFrame()
    tmp = df.dropna(subset=list(need)).copy()
    tmp = tmp[tmp["OLP"] > 0]
    tmp["YearMonth"] = tmp["COE"].dt.to_period("M").dt.to_timestamp()
    tmp["SaleToListRatio"] = (tmp["Sold Price"] / tmp["OLP"]) * 100
    tmp, outliers = _filter_ratio_bounds(tmp, "SaleToListRatio", ratio_min, ratio_max)
    out = tmp.groupby(["City", "YearMonth"])["SaleToListRatio"].mean().reset_index()
    out["SaleToListRatio"] = out["SaleToListRatio"].round(2)
    return out, outliers

def kpi_top_zip_sales(df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    need = {"Zip Code", "COE"}
    if not need.issubset(df.columns):
        return pd.DataFrame()
    tmp = df.dropna(subset=list(need)).copy()
    tmp["YearMonth"] = tmp["COE"].dt.to_period("M").dt.to_timestamp()
    out = (
        tmp.groupby(["YearMonth", "Zip Code"])
        .size()
        .reset_index(name="SalesCount")
        .sort_values(["YearMonth", "SalesCount"], ascending=[True, False])
    )
    return out.groupby("YearMonth").head(top_n).reset_index(drop=True)

def kpi_price_bands_and_p66(df: pd.DataFrame) -> pd.DataFrame:
    need = {"COE", "Sold Price"}
    if not need.issubset(df.columns):
        return pd.DataFrame()
    tmp = df.dropna(subset=list(need)).copy()
    tmp["YearMonth"] = tmp["COE"].dt.to_period("M").dt.to_timestamp()
    rows = []
    for ym, g in tmp.groupby("YearMonth"):
        total = len(g)
        if total == 0:
            continue
        row = {"YearMonth": ym, "TotalSales": total}
        for label, lo, hi in PRICE_BANDS:
            cnt = ((g["Sold Price"] >= lo) & (g["Sold Price"] <= hi)).sum()
            row[label] = round((cnt / total) * 100, 2)
        row["P66_SoldPrice"] = float(g["Sold Price"].quantile(0.66))
        rows.append(row)
    return pd.DataFrame(rows).sort_values("YearMonth")

def kpi_price_reduction_metrics(df: pd.DataFrame, min_bin_size: int = 10):
    """Computes (flag table with Avg/Median/N) and reduction-bin table (mean/median/N)."""
    need = {"OLP", "FLP", "DOM"}
    if not need.issubset(df.columns):
        return pd.DataFrame(), pd.DataFrame()
    tmp = df.dropna(subset=["OLP", "FLP"]).copy()

    # Guard weird flips and clip tails to 0–50%
    tmp = tmp[tmp["FLP"] <= tmp["OLP"]]
    tmp = tmp[tmp["OLP"] > 0]
    tmp["PriceReduction"] = tmp["OLP"] - tmp["FLP"]
    tmp["PriceReductionPct"] = (tmp["PriceReduction"] / tmp["OLP"]) * 100
    tmp["PriceReductionPct"] = tmp["PriceReductionPct"].clip(0, 50)
    tmp["HadReduction"] = tmp["PriceReduction"] > 0

    # With vs without reduction — keep both mean & median in table
    dom_by_flag = (
        tmp.dropna(subset=["DOM"])
          .groupby("HadReduction")["DOM"]
          .agg(AvgDOM="mean", MedianDOM="median", N="size")
          .reset_index()
          .replace({True: "Yes", False: "No"})
          .rename(columns={"HadReduction": "HadPriceReduction"})
    )

    # Reduction bins 0–50 by 5s
    bins = list(range(0, 55, 5))
    labels = [f"{i}-{i+5}%" for i in bins[:-1]]
    tmp["ReductionBin"] = pd.cut(tmp["PriceReductionPct"], bins=bins, labels=labels, right=False)

    dom_by_bin = (
        tmp.dropna(subset=["DOM"])
          .groupby("ReductionBin")["DOM"]
          .agg(DOM_mean="mean", DOM_median="median", N="size")
          .reset_index()
    )
    dom_by_bin = dom_by_bin[dom_by_bin["N"] >= min_bin_size]
    return dom_by_flag, dom_by_bin

# -------------------------
# KPIs (YTD)
# -------------------------
def kpi_top_zip_sales_ytd(df: pd.DataFrame, top_n: int = 10, min_ytd_sales: int = 30) -> pd.DataFrame:
    need = {"COE", "Zip Code"}
    if not need.issubset(df.columns):
        return pd.DataFrame()
    tmp = ytd_filter(df)
    out = tmp.groupby("Zip Code").size().rename("SalesCount").reset_index()
    out = out[out["SalesCount"] >= min_ytd_sales]
    return out.sort_values("SalesCount", ascending=False).head(top_n)

def _top_appreciation_generic_ytd(df: pd.DataFrame, group_col: str, min_ytd_sales: int, min_endmonth: int):
    need = {"COE", group_col, "Sold Price"}
    if not need.issubset(df.columns):
        return pd.DataFrame(), None
    combo, latest_df, latest_month = jan_and_latest_month_groups(df, group_col)
    if combo.empty:
        return pd.DataFrame(), None
    combo = combo.fillna(0)
    combo = combo[
        (combo["YTDSales"] >= min_ytd_sales) &
        (combo["JanCount"] >= min_endmonth) &
        (combo["LatestCount"] >= min_endmonth)
    ]
    if combo.empty:
        return pd.DataFrame(), latest_month
    combo["AppreciationPct"] = np.where(
        combo["JanMedian"] > 0,
        (combo["LatestMedian"] - combo["JanMedian"]) / combo["JanMedian"] * 100,
        np.nan
    )
    out = combo.dropna(subset=["AppreciationPct"]).copy()
    out = out.sort_values("AppreciationPct", ascending=False).head(10)
    return out.reset_index(drop=True), latest_month

def kpi_top_zip_appreciation_ytd(df: pd.DataFrame, min_ytd_sales: int = 30, min_endmonth: int = 10):
    out, latest_month = _top_appreciation_generic_ytd(df, "Zip Code", min_ytd_sales, min_endmonth)
    return out, latest_month

def kpi_top_city_sales_ytd(df: pd.DataFrame, top_n: int = 10, min_ytd_sales: int = 600) -> pd.DataFrame:
    need = {"COE", "City"}
    if not need.issubset(df.columns):
        return pd.DataFrame()
    tmp = ytd_filter(df)
    out = tmp.groupby("City").size().rename("SalesCount").reset_index()
    out = out[out["SalesCount"] >= min_ytd_sales]
    return out.sort_values("SalesCount", ascending=False).head(top_n)

def kpi_top_city_median_price_ytd(df: pd.DataFrame, top_n: int = 10, min_ytd_sales: int = 600) -> pd.DataFrame:
    need = {"COE", "City", "Sold Price"}
    if not need.issubset(df.columns):
        return pd.DataFrame()
    tmp = ytd_filter(df)
    if tmp.empty:
        return pd.DataFrame()
    g = tmp.groupby("City").agg(
        MedianSoldPrice=("Sold Price", "median"),
        YTDSales=("Sold Price", "size")
    ).reset_index()
    g = g[g["YTDSales"] >= min_ytd_sales]
    return g.sort_values("MedianSoldPrice", ascending=False).head(top_n)

def kpi_top_city_appreciation_ytd(df: pd.DataFrame, min_ytd_sales: int = 600, min_endmonth: int = 10):
    out, latest_month = _top_appreciation_generic_ytd(df, "City", min_ytd_sales, min_endmonth)
    return out, latest_month

def kpi_sale_to_list_ytd(df: pd.DataFrame, ratio_min: float, ratio_max: float):
    need = {"COE", "Sold Price", "OLP"}
    if not need.issubset(df.columns):
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    tmp = ytd_filter(df)
    tmp = tmp.dropna(subset=["Sold Price", "OLP"])
    tmp = tmp[tmp["OLP"] > 0]
    tmp["SaleToListPct"] = (tmp["Sold Price"] / tmp["OLP"]) * 100
    tmp_fair, outliers = _filter_ratio_bounds(tmp, "SaleToListPct", ratio_min, ratio_max)

    overall = pd.DataFrame({
        "Metric": ["Avg % of OLP Achieved (YTD)"],
        "Value": [round(tmp_fair["SaleToListPct"].mean(), 2)]
    })

    by_city = (
        tmp_fair.dropna(subset=["City"])
        .groupby("City")["SaleToListPct"]
        .mean()
        .round(2)
        .reset_index(name="AvgSaleToListPct")
        .sort_values("AvgSaleToListPct", ascending=False)
    )
    return overall, by_city, outliers

# -------------------------
# Excel report
# -------------------------
def write_excel_bundle(tables: dict, out_dir: str, stamp: str, reasons: dict) -> str:
    """
    Always write every expected sheet.
    If a table is empty, write a one-row note listing missing columns / reason.
    """
    xlsx_path = os.path.join(out_dir, f"report_{stamp}.xlsx")
    with pd.ExcelWriter(xlsx_path, engine="xlsxwriter") as xw:
        for sheet, df in tables.items():
            sheet_name = sheet[:31]  # Excel limit
            if df is not None and not df.empty:
                df.to_excel(xw, sheet_name=sheet_name, index=False)
            else:
                msg = reasons.get(sheet, "No data.")
                placeholder = pd.DataFrame({"Note": [msg]})
                placeholder.to_excel(xw, sheet_name=sheet_name, index=False)
    return xlsx_path

# -------------------------
# Validation helpers
# -------------------------
def missing_requirements(df: pd.DataFrame, need: set) -> list:
    return sorted([c for c in need if c not in df.columns])

# -------------------------
# Charts
# -------------------------
def _fig_save(path: str):
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()

def rainbow_palette(n: int, cmap_name: str = "rainbow"):
    cmap = plt.get_cmap(cmap_name)
    if n <= 1:
        return [cmap(0.5)]
    return [cmap(i / (n - 1)) for i in range(n)]

def plot_sales_counts_monthly_line(df: pd.DataFrame, out_dir: str, stamp: str):
    if df is None or df.empty:
        return None
    p = os.path.join(out_dir, f"sales_counts_monthly_{stamp}.png")
    plot_df = df.copy().sort_values("YearMonth")
    labels = pd.to_datetime(plot_df["YearMonth"]).dt.strftime("%b")
    plt.figure(figsize=(12,6))
    plt.plot(labels, plot_df["SalesCount"], marker="o", color="red")
    for x, y in zip(range(len(labels)), plot_df["SalesCount"].values):
        plt.text(x, y, f"{int(y):,}", va="bottom", ha="center", fontsize=9)
    plt.title("Closed Sales by Month")
    plt.xlabel(f"Month\n\n{FOOTER_SOURCE}")
    plt.ylabel("Closed Sales (count)")
    _fig_save(p)
    return p

def plot_dom_by_price_reduction_flag(df: pd.DataFrame, out_dir: str, stamp: str):
    if df is None or df.empty:
        return None
    p = os.path.join(out_dir, f"avg_dom_by_price_reduction_flag_{stamp}.png")
    plot_df = df.copy()
    # Use median for the graphic
    vals = plot_df["MedianDOM"] if "MedianDOM" in plot_df.columns else plot_df["AvgDOM"]
    plt.figure(figsize=(8,6))
    plt.bar(plot_df["HadPriceReduction"].astype(str), vals, color=rainbow_palette(len(plot_df)))
    for i, v in enumerate(vals.values):
        try:
            plt.text(i, v, f" {round(float(v))}", va="bottom")
        except Exception:
            pass
    plt.title("Median DOM — With vs Without Price Reduction (YTD)")
    plt.xlabel(f"Had Price Reduction?\n\n{FOOTER_SOURCE}")
    plt.ylabel("Median Days on Market")
    _fig_save(p)
    return p

def plot_dom_by_price_reduction_bins(df: pd.DataFrame, out_dir: str, stamp: str):
    if df is None or df.empty:
        return None
    p = os.path.join(out_dir, f"avg_dom_by_price_reduction_bins_{stamp}.png")
    plot_df = df.copy()
    yvals = plot_df["DOM_median"] if "DOM_median" in plot_df.columns else plot_df["DOM_mean"]
    plt.figure(figsize=(12,6))
    xlabels = plot_df["ReductionBin"].astype(str)
    plt.bar(xlabels, yvals, color=rainbow_palette(len(plot_df)))
    for i, v in enumerate(yvals.values):
        try:
            plt.text(i, v, f" {round(float(v))}", va="bottom", rotation=90)
        except Exception:
            pass
    plt.title("Median DOM by Price Reduction (%) Bin (YTD)")
    plt.xlabel(f"Reduction % Bin\n\n{FOOTER_SOURCE}")
    plt.ylabel("Median Days on Market")
    plt.xticks(rotation=90)
    _fig_save(p)
    return p

def _barh_with_labels(df: pd.DataFrame, label_col: str, value_col: str, title: str, xlabel: str, out_path: str, percent: bool=False):
    if df is None or df.empty:
        return None
    plt.figure(figsize=(12, 8))
    plot_df = df.copy().sort_values(value_col, ascending=True)
    y = np.arange(len(plot_df))
    vals = plot_df[value_col].values
    labels = plot_df[label_col].astype(str).values
    plt.barh(y, vals, color=rainbow_palette(len(vals)))
    for i, v in enumerate(vals):
        text = f"{v:,.0f}" if not percent else f"{v:.1f}%"
        try:
            if np.isfinite(float(v)):
                plt.text(v, i, f"  {text}", va="center")
        except Exception:
            pass
    plt.yticks(y, labels)
    plt.title(title)
    plt.xlabel(f"{xlabel}\n\n{FOOTER_SOURCE}")
    _fig_save(out_path)
    return out_path

def plot_top10_zip_sales_ytd(df: pd.DataFrame, out_dir: str, stamp: str):
    return _barh_with_labels(df, "Zip Code", "SalesCount",
                             "Top 10 ZIPs by Sales (YTD)",
                             "Closed Sales (count)",
                             os.path.join(out_dir, f"top10_zip_sales_ytd_{stamp}.png"))

def plot_top10_zip_appreciation_ytd(df: pd.DataFrame, latest_month: int, out_dir: str, stamp: str):
    title = f"Top 10 ZIPs by Appreciation (Jan → {latest_month:02d})"
    p = os.path.join(out_dir, f"top10_zip_appreciation_ytd_{stamp}.png")
    plot_df = df.copy()
    plot_df["AppreciationPct"] = plot_df["AppreciationPct"].astype(float)
    return _barh_with_labels(plot_df, "Zip Code", "AppreciationPct", title, "Appreciation (%)", p, percent=True)

def plot_top10_city_sales_ytd(df: pd.DataFrame, out_dir: str, stamp: str):
    return _barh_with_labels(df, "City", "SalesCount",
                             "Top 10 Cities by Sales (YTD)",
                             "Closed Sales (count)",
                             os.path.join(out_dir, f"top10_city_sales_ytd_{stamp}.png"))

def plot_top10_city_appreciation_ytd(df: pd.DataFrame, latest_month: int, out_dir: str, stamp: str):
    title = f"Top 10 Cities by Appreciation (Jan → {latest_month:02d})"
    p = os.path.join(out_dir, f"top10_city_appreciation_ytd_{stamp}.png")
    plot_df = df.copy()
    plot_df["AppreciationPct"] = plot_df["AppreciationPct"].astype(float)
    return _barh_with_labels(plot_df, "City", "AppreciationPct", title, "Appreciation (%)", p, percent=True)

def plot_top10_city_median_price_ytd(df: pd.DataFrame, out_dir: str, stamp: str):
    if df is None or df.empty:
        return None
    p = os.path.join(out_dir, f"top10_city_median_price_ytd_{stamp}.png")
    plot_df = df.copy().sort_values("MedianSoldPrice", ascending=True)
    plt.figure(figsize=(12,8))
    plt.barh(plot_df["City"].astype(str), plot_df["MedianSoldPrice"], color=rainbow_palette(len(plot_df)))
    for i, v in enumerate(plot_df["MedianSoldPrice"].values):
        try:
            plt.text(v, i, f" ${float(v):,.0f}", va="center")
        except Exception:
            pass
    plt.title("Top 10 Cities — YTD Median Sold Price")
    plt.xlabel(f"Median Sold Price\n\n{FOOTER_SOURCE}")
    plt.ylabel("City")
    _fig_save(p)
    return p

def plot_sale_to_list_by_city_ytd(df: pd.DataFrame, out_dir: str, stamp: str):
    return _barh_with_labels(df, "City", "AvgSaleToListPct",
                             "Avg % of Original List Price Achieved by City (YTD)",
                             "% of OLP",
                             os.path.join(out_dir, f"ytd_sale_to_list_city_{stamp}.png"),
                             percent=True)

def plot_sale_to_list_overall_kpi(overall_df: pd.DataFrame, out_dir: str, stamp: str):
    if overall_df is None or overall_df.empty:
        return None
    v = overall_df.iloc[0]["Value"]
    p = os.path.join(out_dir, f"ytd_sale_to_list_overall_{stamp}.png")
    plt.figure(figsize=(6, 4))
    plt.axis("off")
    plt.text(0.5, 0.6, "Avg % of OLP Achieved (YTD)", ha="center", va="center", fontsize=14, weight="bold")
    plt.text(0.5, 0.35, f"{v:.2f}%", ha="center", va="center", fontsize=36)
    plt.text(0.5, 0.08, FOOTER_SOURCE, ha="center", va="center", fontsize=9)
    _fig_save(p)
    return p

def plot_price_bands_p66(df: pd.DataFrame, out_dir: str, stamp: str):
    """
    Stacked bars of monthly price-band shares + optional P66 text.
    Expects columns: YearMonth, '500k and Under', '500,001 to 800k', '801k and Above', P66_SoldPrice.
    """
    if df is None or df.empty:
        return None
    p = os.path.join(out_dir, f"price_bands_p66_{stamp}.png")

    months = pd.to_datetime(df["YearMonth"])
    labels = [m.strftime("%b") for m in months]

    series = ["500k and Under", "500,001 to 800k", "801k and Above"]
    bottoms = np.zeros(len(df))
    cols = rainbow_palette(len(series))

    plt.figure(figsize=(12, 7))
    for s, col in zip(series, cols):
        vals = df[s].values if s in df.columns else np.zeros(len(df))
        plt.bar(labels, vals, bottom=bottoms, label=s, color=col)
        bottoms += vals

    if "P66_SoldPrice" in df.columns:
        latest_p66 = float(df["P66_SoldPrice"].iloc[-1])
        plt.text(
            0.01, -0.17,
            f"P66 (latest): ${latest_p66:,.0f}",
            transform=plt.gca().transAxes, ha="left", va="top", fontsize=10
        )

    plt.title("Monthly Price-Band Shares (% of Sales)")
    plt.ylabel("Share (%)")
    plt.yticks([0, 25, 50, 75, 100])
    plt.xlabel(f"Month\n\n{FOOTER_SOURCE}")
    plt.legend(title="Price Bands", loc="upper left")
    _fig_save(p)
    return p

def render_charts(charts: dict, out_dir: str, stamp: str,
                  city_plot_ratio_min: float = 70.0, city_plot_ratio_max: float = 105.0):
    """Render all charts, skipping gracefully if a chart fails."""
    def _abs(p): return os.path.abspath(p) if p else p
    paths = []

    try:
        item = charts.get("price_bands_p66")
        if item is not None and (not hasattr(item, 'empty') or not item.empty):
            paths.append(_abs(plot_price_bands_p66(item, out_dir, stamp)))
    except Exception as e:
        print(f"[chart:price_bands_p66] skipped: {e}")

    try:
        item = charts.get("top10_zip_sales_ytd")
        if item is not None and (not hasattr(item, 'empty') or not item.empty):
            paths.append(_abs(plot_top10_zip_sales_ytd(item, out_dir, stamp)))
    except Exception as e:
        print(f"[chart:top10_zip_sales_ytd] skipped: {e}")

    try:
        item = charts.get("top10_zip_appreciation_ytd")
        if item is not None and (not hasattr(item, 'empty') or not item.empty):
            paths.append(_abs(plot_top10_zip_appreciation_ytd(item, charts.get('top10_zip_appreciation_latest_month'), out_dir, stamp)))
    except Exception as e:
        print(f"[chart:top10_zip_appreciation_ytd] skipped: {e}")

    try:
        item = charts.get("top10_city_sales_ytd")
        if item is not None and (not hasattr(item, 'empty') or not item.empty):
            paths.append(_abs(plot_top10_city_sales_ytd(item, out_dir, stamp)))
    except Exception as e:
        print(f"[chart:top10_city_sales_ytd] skipped: {e}")

    try:
        item = charts.get("top10_city_appreciation_ytd")
        if item is not None and (not hasattr(item, 'empty') or not item.empty):
            paths.append(_abs(plot_top10_city_appreciation_ytd(item, charts.get('top10_city_appreciation_latest_month'), out_dir, stamp)))
    except Exception as e:
        print(f"[chart:top10_city_appreciation_ytd] skipped: {e}")

    try:
        item = charts.get("top10_city_median_price_ytd")
        if item is not None and (not hasattr(item, 'empty') or not item.empty):
            paths.append(_abs(plot_top10_city_median_price_ytd(item, out_dir, stamp)))
    except Exception as e:
        print(f"[chart:top10_city_median_price_ytd] skipped: {e}")

    try:
        item = charts.get("ytd_sale_to_list_overall")
        if item is not None and (not hasattr(item, 'empty') or not item.empty):
            paths.append(_abs(plot_sale_to_list_overall_kpi(item, out_dir, stamp)))
    except Exception as e:
        print(f"[chart:ytd_sale_to_list_overall] skipped: {e}")

    try:
        # Apply stricter 70–105% window just for the city rank PLOT
        by_city_table = charts.get("ytd_sale_to_list_by_city_raw")
        if by_city_table is not None and not by_city_table.empty:
            paths.append(_abs(plot_sale_to_list_by_city_ytd(by_city_table, out_dir, stamp)))
        else:
            item = charts.get("ytd_sale_to_list_by_city")
            if item is not None and (not hasattr(item, 'empty') or not item.empty):
                paths.append(_abs(plot_sale_to_list_by_city_ytd(item, out_dir, stamp)))
    except Exception as e:
        print(f"[chart:ytd_sale_to_list_by_city] skipped: {e}")

    try:
        item = charts.get("sales_counts_monthly")
        if item is not None and (not hasattr(item, 'empty') or not item.empty):
            paths.append(_abs(plot_sales_counts_monthly_line(item, out_dir, stamp)))
    except Exception as e:
        print(f"[chart:sales_counts_monthly] skipped: {e}")

    try:
        item = charts.get("avg_dom_by_price_reduction_flag")
        if item is not None and (not hasattr(item, 'empty') or not item.empty):
            paths.append(_abs(plot_dom_by_price_reduction_flag(item, out_dir, stamp)))
    except Exception as e:
        print(f"[chart:avg_dom_by_price_reduction_flag] skipped: {e}")

    try:
        item = charts.get("avg_dom_by_price_reduction_bins")
        if item is not None and (not hasattr(item, 'empty') or not item.empty):
            paths.append(_abs(plot_dom_by_price_reduction_bins(item, out_dir, stamp)))
    except Exception as e:
        print(f"[chart:avg_dom_by_price_reduction_bins] skipped: {e}")

    return [p for p in paths if p]

# -------------------------
# Main pipeline
# -------------------------
def run_pipeline(sales_path: str, freddie_path: str, out_dir: str, csv_exports: bool,
                 make_charts: bool,
                 min_zip_ytd: int, min_city_ytd: int, min_zip_endmonth: int, min_city_endmonth: int,
                 ratio_min: float, ratio_max: float,
                 min_bin_size: int,
                 run_subdir: bool,
                 stamp_with_time: bool,
                 city_plot_ratio_min: float = 70.0,
                 city_plot_ratio_max: float = 105.0):
    # Decide stamp and run directory
    stamp = ts_stamp_include_time(stamp_with_time)
    if run_subdir:
        out_dir = os.path.join(out_dir, f"run_{stamp}")
    safe_make_outdir(out_dir)

    # Load + normalize
    sales = read_csv_fallback(sales_path)
    sales = normalize_columns(sales)

    # ZIP portability (keep AZ the same, protect any leading zeros elsewhere)
    if "Zip Code" in sales.columns:
        sales["Zip Code"] = sales["Zip Code"].astype(str).str.extract(r"(\d+)")[0].str.zfill(5)

    # NEW: synthesize 'Pool' from Private/Community Pool Y/N pair (when present)
    sales = synthesize_pool_column(sales)

    if "COE" not in sales.columns:
        raise ValueError("Expected a 'COE' column in the sales file.")
    if "Sold Price" not in sales.columns:
        raise ValueError("Expected a 'Sold Price' column in the sales file.")

    sales["COE"] = pd.to_datetime(sales["COE"], errors="coerce")
    for c in ("Sold Price", "OLP", "FLP", "DOM"):
        if c in sales.columns:
            sales[c] = pd.to_numeric(sales[c], errors="coerce")

    # Optional conservative de-dupe (APN + COE + Sold Price)
    pre_dupes = None
    if "APN" in sales.columns:
        pre_dupes = int(sales.duplicated(subset=["APN", "COE", "Sold Price"]).sum())
        sales = sales.drop_duplicates(subset=["APN", "COE", "Sold Price"])

    # Enrichment
    sales = extract_loan_type_from_features(sales)
    sales = compute_interest_from_freddie(sales, freddie_path)

    # Drop Features after enrichment
    if "Features" in sales.columns:
        sales = sales.drop(columns=["Features"])

    # KPIs (monthly)
    kpi1 = kpi_sales_counts(sales)
    kpi2 = kpi_median_price_by_city_month(sales)
    kpi3, kpi3_outliers = kpi_sale_to_list_by_city_month(sales, ratio_min, ratio_max)
    kpi4 = kpi_top_zip_sales(sales, top_n=10)
    kpi5 = kpi_price_bands_and_p66(sales)
    dom_flag, dom_bins = kpi_price_reduction_metrics(sales, min_bin_size=min_bin_size)

    # KPIs (YTD)
    zip_sales_ytd = kpi_top_zip_sales_ytd(sales, top_n=10, min_ytd_sales=min_zip_ytd)
    zip_app_ytd, zip_latest = kpi_top_zip_appreciation_ytd(sales, min_ytd_sales=min_zip_ytd, min_endmonth=min_zip_endmonth)
    city_sales_ytd = kpi_top_city_sales_ytd(sales, top_n=10, min_ytd_sales=min_city_ytd)
    city_app_ytd, city_latest = kpi_top_city_appreciation_ytd(sales, min_ytd_sales=min_city_ytd, min_endmonth=min_city_endmonth)
    city_median_ytd = kpi_top_city_median_price_ytd(sales, top_n=10, min_ytd_sales=min_city_ytd)
    olp_overall_ytd, by_city_ytd, olp_outliers_ytd = kpi_sale_to_list_ytd(sales, ratio_min, ratio_max)

    # For plotting the city sale-to-list ranks with a tighter window (70–105%)
    # Recreate just for the PLOT, not the table:
    _, by_city_for_plot, _ = kpi_sale_to_list_ytd(sales, city_plot_ratio_min, city_plot_ratio_max)

    # Requirements report (why a KPI might be empty)
    reasons = {}
    for name, need in KPI_REQUIREMENTS.items():
        miss = missing_requirements(sales, need)
        if miss:
            reasons[name] = f"No data: missing required columns {miss}"

    # Date coverage check
    min_coe = pd.to_datetime(sales["COE"]).min()
    max_coe = pd.to_datetime(sales["COE"]).max()
    print(f"\nData covers COE from {min_coe:%Y-%m-%d} to {max_coe:%Y-%m-%d}")

    # QC summary
    qc = qc_checks(sales)
    if pre_dupes is not None:
        qc["duplicate_apn_count_pre_dedupe"] = pre_dupes
        qc["deduped_rows_on_apn_coe_price"] = int(pre_dupes)  # number removed equals pre_dupes

    # Save enriched
    enriched_path = os.path.join(out_dir, f"2025_YTD_enriched_{stamp}.csv")
    sales.to_csv(enriched_path, index=False)

    # Bundle to Excel (always include every sheet)
    all_tables = {
        # QC + monthly
        "qc_summary": pd.DataFrame([qc]),
        "sales_counts_monthly": kpi1,
        "median_price_by_city_month": kpi2,
        "sale_to_list_by_city_month": kpi3,
        "top10_zip_sales_by_month": kpi4,
        "price_band_share_and_p66_by_month": kpi5,
        "avg_dom_by_price_reduction_flag": dom_flag,
        "avg_dom_by_price_reduction_bins": dom_bins,
        "sale_to_list_by_city_outliers": kpi3_outliers,

        # YTD tables
        "top10_zip_sales_ytd": zip_sales_ytd,
        "top10_zip_appreciation_ytd": zip_app_ytd,
        "top10_city_sales_ytd": city_sales_ytd,
        "top10_city_appreciation_ytd": city_app_ytd,
        "top10_city_median_price_ytd": city_median_ytd,
        "ytd_sale_to_list_overall": olp_overall_ytd,
        "ytd_sale_to_list_by_city": by_city_ytd,
        "ytd_sale_to_list_outliers": olp_outliers_ytd,
    }
    excel_path = write_excel_bundle(all_tables, out_dir, stamp, reasons)

    # Optional: also write per-table CSVs (only for non-empty tables)
    if csv_exports:
        for name, df in all_tables.items():
            if df is not None and not df.empty:
                df.to_csv(os.path.join(out_dir, f"{name}_{stamp}.csv"), index=False)

    # Console preview helpers
    def preview(title, df, n=5, tail_n=3):
        if df is not None and not df.empty:
            print(f"\n{title} — first {n} & last {tail_n}")
            print(df.head(n).to_string(index=False))
            if len(df) > n:
                print("...")
                print(df.tail(tail_n).to_string(index=False))
        else:
            reason = reasons.get(title, "No data.")
            print(f"\n{title} — {reason}")

    # Run summary + previews
    print("\n=== Run Summary ===")
    print(f"Rows: {qc.get('rows')} | Missing COE: {qc.get('missing_coe')} | Missing Sold Price: {qc.get('missing_soldprice')}")
    if "duplicate_apn_count_pre_dedupe" in qc:
        print(f"Duplicate APNs (pre-dedupe): {qc['duplicate_apn_count_pre_dedupe']}")
        print(f"De-duped rows on APN+COE+SoldPrice: {qc['deduped_rows_on_apn_coe_price']}")
    elif qc.get("duplicate_apn_count") is not None:
        print(f"Duplicate APNs: {qc['duplicate_apn_count']}")
    print(f"\nEnriched sales saved: {os.path.abspath(enriched_path)}")
    print(f"Excel bundle saved:  {os.path.abspath(excel_path)}")

    print("\n=== KPI Quick Preview (monthly) ===")
    preview("sales_counts_monthly", kpi1)
    preview("median_price_by_city_month", kpi2)
    preview("sale_to_list_by_city_month", kpi3)
    preview("top10_zip_sales_by_month", kpi4)
    preview("price_band_share_and_p66_by_month", kpi5)

    print("\n=== KPI Quick Preview (YTD) ===")
    preview("top10_zip_sales_ytd", zip_sales_ytd)
    preview("top10_zip_appreciation_ytd", zip_app_ytd)
    preview("top10_city_sales_ytd", city_sales_ytd)
    preview("top10_city_appreciation_ytd", city_app_ytd)
    preview("ytd_sale_to_list_overall", olp_overall_ytd)
    preview("ytd_sale_to_list_by_city", by_city_ytd)

    # Charts
    if make_charts:
        chart_inputs = {
            "price_bands_p66": kpi5,
            "top10_zip_sales_ytd": zip_sales_ytd,
            "top10_zip_appreciation_ytd": zip_app_ytd,
            "top10_zip_appreciation_latest_month": zip_latest,
            "top10_city_sales_ytd": city_sales_ytd,
            "top10_city_appreciation_ytd": city_app_ytd,
            "top10_city_appreciation_latest_month": city_latest,
            "top10_city_median_price_ytd": city_median_ytd,
            "ytd_sale_to_list_overall": olp_overall_ytd,
            # Provide the stricter-by-city table just for the plot, but keep normal table in Excel
            "ytd_sale_to_list_by_city_raw": by_city_for_plot,
            "sales_counts_monthly": kpi1,
            "avg_dom_by_price_reduction_flag": dom_flag,
            "avg_dom_by_price_reduction_bins": dom_bins,
        }
        paths = render_charts(chart_inputs, out_dir, stamp,
                              city_plot_ratio_min=city_plot_ratio_min,
                              city_plot_ratio_max=city_plot_ratio_max)
        if paths:
            print("\nCharts written:")
            for p in paths:
                print(" -", os.path.abspath(p))
        else:
            print("\nCharts: no eligible tables to render.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Daily/Monthly Real Estate KPI Pipeline (always writes all sheets)")
    parser.add_argument("--sales", default=SALES_FILE, help="Path to sales CSV")
    parser.add_argument("--freddie", default=FREDDIE_FILE, help="Path to freddiemac CSV")
    parser.add_argument("--outdir", default=OUT_DIR, help="Output directory")
    parser.add_argument("--csv-exports", action="store_true", help="Also write individual KPI CSVs (non-empty only)")
    # Charts on by default; use --no-charts to disable
    parser.add_argument("--charts", action="store_true", default=True, help="Render PNG charts to the output directory (default: on)")
    parser.add_argument("--no-charts", action="store_true", help="Disable chart rendering")
    # Ratio bounds for sale-to-list (tables/overall)
    parser.add_argument("--ratio-min", type=float, default=60.0, help="Min acceptable Sale-to-List % (inclusive)")
    parser.add_argument("--ratio-max", type=float, default=110.0, help="Max acceptable Sale-to-List % (inclusive)")
    # City plot specific tighter bounds (only affects the graphic)
    parser.add_argument("--city-plot-ratio-min", type=float, default=70.0, help="Min % for city rank plot (inclusive)")
    parser.add_argument("--city-plot-ratio-max", type=float, default=105.0, help="Max % for city rank plot (inclusive)")
    # YTD thresholds
    parser.add_argument("--min-zip-ytd", type=int, default=30, help="Minimum YTD sales for a ZIP to be ranked")
    parser.add_argument("--min-city-ytd", type=int, default=600, help="Minimum YTD sales for a City to be ranked")
    parser.add_argument("--min-zip-endmonth", type=int, default=10, help="Minimum Jan and latest-month sales for a ZIP to be included in appreciation calc")
    parser.add_argument("--min-city-endmonth", type=int, default=10, help="Minimum Jan and latest-month sales for a City to be included in appreciation calc")
    # DOM bins minimum sample size
    parser.add_argument("--min-bin-size", type=int, default=10, help="Min sample size per DOM reduction bin to include in chart/table")
    # Output organization
    parser.add_argument("--run-subdir", action="store_true", default=True, help="Put outputs in a unique subfolder per run (default: on)")
    parser.add_argument("--stamp-with-time", action="store_true", help="Append _HHMMSS to the date stamp in filenames")

    args = parser.parse_args()
    if args.no_charts:
        args.charts = False

    run_pipeline(
        args.sales, args.freddie, args.outdir, args.csv_exports, args.charts,
        args.min_zip_ytd, args.min_city_ytd, args.min_zip_endmonth, args.min_city_endmonth,
        args.ratio_min, args.ratio_max,
        args.min_bin_size,
        args.run_subdir,
        args.stamp_with_time,
        city_plot_ratio_min=args.city_plot_ratio_min,
        city_plot_ratio_max=args.city_plot_ratio_max
    )






