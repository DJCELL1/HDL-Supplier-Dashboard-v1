
import os
import io
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio
import streamlit as st

# -------------------- Page setup --------------------
st.set_page_config(page_title="Hardware Direct | CIN7 Purchase Order Dashboard", page_icon="📦", layout="wide")

# -------------------- Header --------------------
logo_path = Path(__file__).parent / "logo.png"
header_cols = st.columns([1, 4])
with header_cols[0]:
    if logo_path.exists():
        st.image(str(logo_path), width=150)
with header_cols[1]:
    st.markdown("<h2 style='margin-bottom:0'>Hardware Direct | CIN7 Purchase Order Dashboard</h2>", unsafe_allow_html=True)
st.markdown("<hr style='margin-top:6px;margin-bottom:10px' />", unsafe_allow_html=True)

# -------------------- Sidebar --------------------
st.sidebar.title("📦 CIN7 PO Dashboard v5.1")
st.sidebar.caption("Upload a CIN7 Purchase Orders CSV")

uploaded = st.sidebar.file_uploader("Upload CSV", type=["csv"])
watch_path = st.sidebar.text_input("Or load from a path (optional)", value="")

st.sidebar.markdown("### Options")
use_ordered = st.sidebar.checkbox("Use Ordered Qty (potential spend)", value=False)
auto_refresh = st.sidebar.checkbox("Auto-refresh", value=False)
refresh_sec = st.sidebar.slider("Every (seconds)", 10, 300, 60)

st.sidebar.markdown("---")
st.sidebar.write("Date range (by month)")

# -------------------- Load data --------------------
@st.cache_data(ttl=5, show_spinner=False)
def load_csv(file_bytes=None, path=None):
    if file_bytes is not None:
        df = pd.read_csv(file_bytes)
    elif path and os.path.exists(path):
        df = pd.read_csv(path)
    else:
        return pd.DataFrame()

    df.columns = [c.strip() for c in df.columns]

    # Dates
    for col in ["Created Date", "Invoice Date", "ETD", "Fully Received", "Supplier Acceptance"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    # Numerics
    for col in ["Item Qty", "Item Qty Moved", "Item Price (Local Currency)", "Item Price",
                "Freight Cost (Local Currency)", "Freight Cost"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Fallbacks
    if "Item Price (Local Currency)" not in df.columns and "Item Price" in df.columns:
        df["Item Price (Local Currency)"] = df["Item Price"]
    if "Freight Cost (Local Currency)" not in df.columns and "Freight Cost" in df.columns:
        df["Freight Cost (Local Currency)"] = df["Freight Cost"]

    # Month helper
    if "Created Date" in df.columns:
        df["Created Month"] = df["Created Date"].dt.to_period("M").dt.to_timestamp()

    # Backorder detection + Base PO
    if "Order Ref" in df.columns:
        df["Is Backorder Suffix"] = df["Order Ref"].str.match(r".*[A-Z]$", na=False)
        df["Base PO"] = df["Order Ref"].str.replace(r"[A-Z]$", "", regex=True)
    else:
        df["Is Backorder Suffix"] = False
        df["Base PO"] = df.get("Order Id", pd.Series(range(len(df))))

    # Item values
    df["Item Value (Moved)"] = np.where(
        df.get("Item Qty Moved").notna() if "Item Qty Moved" in df.columns else False,
        df["Item Qty Moved"].fillna(0) * df["Item Price (Local Currency)"].fillna(0),
        np.nan
    )
    df["Item Value (Ordered)"] = np.where(
        df.get("Item Qty").notna() if "Item Qty" in df.columns else False,
        df["Item Qty"].fillna(0) * df["Item Price (Local Currency)"].fillna(0),
        np.nan
    )

    return df


df = load_csv(uploaded, watch_path)

if df.empty:
    st.info("⬆️ Upload a CIN7 Purchase Orders CSV (or provide a path) to begin.", icon="👋")
    st.stop()

st.sidebar.success(f"Loaded rows: {len(df):,}")

# -------------------- Date range (month-year) --------------------
if "Created Date" in df.columns and not df["Created Date"].dropna().empty:
    min_m = df["Created Date"].min().to_period("M")
    max_m = df["Created Date"].max().to_period("M")
    months = pd.period_range(min_m, max_m, freq="M")
    start_month = st.sidebar.selectbox("Start month", months.astype(str), index=0)
    end_month = st.sidebar.selectbox("End month", months.astype(str), index=len(months) - 1)
    start_ts = pd.Period(start_month).to_timestamp()
    end_ts = (pd.Period(end_month) + 1).to_timestamp()
else:
    start_ts, end_ts = None, None

# -------------------- Filters --------------------
supplier_col = "Company" if "Company" in df.columns else None
branch_col = "Branch" if "Branch" in df.columns else None
project_col = "Project Name" if "Project Name" in df.columns else None

suppliers = sorted(df[supplier_col].dropna().unique().tolist()) if supplier_col else []
branches = sorted(df[branch_col].dropna().unique().tolist()) if branch_col else []
projects = sorted(df[project_col].dropna().unique().tolist()) if project_col else []

sel_suppliers = st.sidebar.multiselect("Suppliers", suppliers, default=[])
sel_branches = st.sidebar.multiselect("Branches", branches)
sel_projects = st.sidebar.multiselect("Projects", projects)

# Apply filters
fdf = df.copy()
if start_ts is not None and end_ts is not None and "Created Date" in fdf.columns:
    fdf = fdf[(fdf["Created Date"] >= start_ts) & (fdf["Created Date"] < end_ts)]
if supplier_col and sel_suppliers:
    fdf = fdf[fdf[supplier_col].isin(sel_suppliers)]
if branch_col and sel_branches:
    fdf = fdf[fdf[branch_col].isin(sel_branches)]
if project_col and sel_projects:
    fdf = fdf[fdf[project_col].isin(sel_projects)]

value_col = "Item Value (Ordered)" if use_ordered else "Item Value (Moved)"

# -------------------- Helpers --------------------
def classify_supply_status(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=["Base PO", "Status"])
    has_suffix = frame.groupby("Base PO")["Is Backorder Suffix"].any().rename("HasSuffix")
    status = has_suffix.apply(lambda x: "Backordered" if x else "Supplied in Full").reset_index(name="Status")
    return status

# -------------------- GLOBAL KPIs --------------------
status_df = classify_supply_status(fdf)
total_spend = float(fdf[value_col].sum()) if value_col in fdf.columns else 0.0
sup_full_pct = (status_df["Status"].eq("Supplied in Full").mean() * 100.0) if not status_df.empty else np.nan
backord_pct = (status_df["Status"].eq("Backordered").mean() * 100.0) if not status_df.empty else np.nan

g1, g2, g3 = st.columns(3)
g1.metric("Total Spend", f"${total_spend:,.0f}")
g2.metric("Delivered in Full %", f"{sup_full_pct:,.1f}%" if sup_full_pct == sup_full_pct else "—")
g3.metric("Backordered %", f"{backord_pct:,.1f}%" if backord_pct == backord_pct else "—")

# -------------------- OVERALL MONTHLY SPEND --------------------
st.markdown("### Overall Monthly Spend")
df_date = df.copy()
if start_ts is not None and end_ts is not None and "Created Date" in df_date.columns:
    df_date = df_date[(df_date["Created Date"] >= start_ts) & (df_date["Created Date"] < end_ts)]
df_date["Value"] = df_date[value_col]

if "Created Month" in df_date.columns and "Value" in df_date.columns:
    trend_all = df_date.groupby("Created Month", as_index=False)["Value"].sum()
    fig_global = px.line(trend_all, x="Created Month", y="Value")
    fig_global.update_traces(mode="lines+markers", line=dict(shape="spline", color="#0ea5e9", width=3),
                             fill="tozeroy", fillcolor="rgba(14,165,233,0.2)")
    fig_global.update_layout(margin=dict(l=10, r=10, t=10, b=10), height=320)
    st.plotly_chart(fig_global, use_container_width=True)

# -------------------- SUPPLIER FOCUS --------------------
st.markdown("---")
st.markdown("## Supplier Focus")

if supplier_col and (sel_suppliers or not suppliers):
    suppliers_to_show = sel_suppliers if sel_suppliers else suppliers

    for sup in suppliers_to_show:
        sframe = fdf[fdf[supplier_col] == sup].copy()
        if sframe.empty:
            continue
        sframe["Value"] = sframe[value_col]

        st.markdown(f"### {sup}")
        s_k1, s_k2, s_k3 = st.columns(3)

        s_status = classify_supply_status(sframe)
        s_total = float(sframe["Value"].sum())
        s_sup_pct = (s_status["Status"].eq("Supplied in Full").mean() * 100.0) if not s_status.empty else np.nan
        s_back_pct = (s_status["Status"].eq("Backordered").mean() * 100.0) if not s_status.empty else np.nan

        s_k1.metric("Supplier Spend", f"${s_total:,.0f}")
        s_k2.metric("Delivered in Full %", f"{s_sup_pct:,.1f}%" if s_sup_pct == s_sup_pct else "—")
        s_k3.metric("Backordered %", f"{s_back_pct:,.1f}%" if s_back_pct == s_back_pct else "—")

        # Charts row
        c1, c2 = st.columns(2)

        if "Created Month" in sframe.columns and not sframe.empty:
            s_month = sframe.groupby("Created Month", as_index=False)["Value"].sum()
            fig_s_trend = px.line(s_month, x="Created Month", y="Value", title="Supplier Spend Trend")
            fig_s_trend.update_traces(mode="lines+markers", line=dict(shape="spline", color="#0ea5e9", width=3),
                                      fill="tozeroy", fillcolor="rgba(14,165,233,0.2)")
            fig_s_trend.update_layout(margin=dict(l=10, r=10, t=40, b=10), height=360)
            c1.plotly_chart(fig_s_trend, use_container_width=True)

        pie_df = s_status.groupby("Status").size().reset_index(name="Count")
        if not pie_df.empty:
            fig_pie = px.pie(pie_df, values="Count", names="Status", title="Supply Performance", hole=0.3)
            fig_pie.update_traces(textposition='inside', textinfo='percent+label', hovertemplate="%{label}: %{value} POs")
            c2.plotly_chart(fig_pie, use_container_width=True)
        else:
            c2.info("No PO supply status available.")

        # Most popular items table
        st.markdown("#### Most Popular Items Ordered")
        table_df = sframe.copy()
        if "Item Name" in table_df.columns and "Item Code" in table_df.columns:
            agg = (table_df.groupby(["Item Code", "Item Name"], as_index=False)
                   .agg(Qty_Ordered=("Item Qty", "sum"),
                        Spend=("Value", "sum"))
                   .sort_values("Spend", ascending=False)
                   .head(15))
            st.dataframe(agg, use_container_width=True, height=320)
        else:
            st.info("Item details not available in this dataset.")
else:
    st.info("Use the sidebar to select suppliers to analyze.")

# -------------------- PDF EXPORT --------------------
st.markdown("---")
st.markdown("### 📤 Generate Management PDF")
# (PDF export section unchanged)

