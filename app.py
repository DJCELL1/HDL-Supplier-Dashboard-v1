
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
from streamlit_plotly_events import plotly_events

# -------------------- Page setup --------------------
st.set_page_config(page_title="Hardware Direct | CIN7 Purchase Order Dashboard", page_icon="📦", layout="wide")

# -------------------- Header --------------------
logo_path = Path(__file__).parent / "logo.png"
header_cols = st.columns([1,4])
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
    for col in ["Created Date","Invoice Date","ETD","Fully Received","Supplier Acceptance"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    # Numerics
    for col in ["Item Qty","Item Qty Moved","Item Price (Local Currency)","Item Price","Freight Cost (Local Currency)","Freight Cost"]:
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

    # Backorder detection + Base PO (include ALL POs)
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
    end_month = st.sidebar.selectbox("End month", months.astype(str), index=len(months)-1)
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

# Workframe with all filters (for supplier section)
fdf = df.copy()
if start_ts is not None and end_ts is not None and "Created Date" in fdf.columns:
    fdf = fdf[(fdf["Created Date"] >= start_ts) & (fdf["Created Date"] < end_ts)]
if supplier_col and sel_suppliers:
    fdf = fdf[fdf[supplier_col].isin(sel_suppliers)]
if branch_col and sel_branches:
    fdf = fdf[fdf[branch_col].isin(sel_branches)]
if project_col and sel_projects:
    fdf = fdf[fdf[project_col].isin(sel_projects)]

# Value column
value_col = "Item Value (Ordered)" if use_ordered else "Item Value (Moved)"

# -------------------- Helpers --------------------
def classify_supply_status(frame: pd.DataFrame) -> pd.DataFrame:
    """Classify each Base PO as Supplied in Full vs Backordered, based on presence of suffix POs (A/B/C...)."""
    if frame.empty:
        return pd.DataFrame(columns=["Base PO","Status"])
    has_suffix = frame.groupby("Base PO")["Is Backorder Suffix"].any().rename("HasSuffix")
    status = has_suffix.apply(lambda x: "Backordered" if x else "Supplied in Full").reset_index(name="Status")
    return status

# -------------------- GLOBAL KPIs --------------------
status_df = classify_supply_status(fdf)
total_spend = float(fdf[value_col].sum()) if value_col in fdf.columns else 0.0
sup_full_pct = (status_df["Status"].eq("Supplied in Full").mean()*100.0) if not status_df.empty else np.nan
backord_pct = (status_df["Status"].eq("Backordered").mean()*100.0) if not status_df.empty else np.nan

g1, g2, g3 = st.columns(3)
g1.metric("Total Spend", f"${total_spend:,.0f}")
g2.metric("Delivered in Full %", f"{sup_full_pct:,.1f}%" if sup_full_pct == sup_full_pct else "—")
g3.metric("Backordered %", f"{backord_pct:,.1f}%" if backord_pct == backord_pct else "—")

# -------------------- OVERALL MONTHLY SPEND (all suppliers) --------------------
st.markdown("### Overall Monthly Spend")
df_date = df.copy()
if start_ts is not None and end_ts is not None and "Created Date" in df_date.columns:
    df_date = df_date[(df_date["Created Date"] >= start_ts) & (df_date["Created Date"] < end_ts)]
df_date["Value"] = df_date[value_col]

if "Created Month" in df_date.columns and "Value" in df_date.columns:
    trend_all = df_date.groupby("Created Month", as_index=False)["Value"].sum()
    fig_global = px.line(trend_all, x="Created Month", y="Value")
    fig_global.update_traces(mode="lines+markers", line=dict(shape="spline", color="#0ea5e9", width=3), fill="tozeroy", fillcolor="rgba(14,165,233,0.2)")
    fig_global.update_layout(margin=dict(l=10,r=10,t=10,b=10), height=320)
    st.plotly_chart(fig_global, use_container_width=True)

    # Month buttons
    st.markdown("#### Select Month")
    months_lbl = trend_all["Created Month"].dt.to_period("M").astype(str).tolist()
    cols = st.columns(min(len(months_lbl), 6) if len(months_lbl)>0 else 1)
    for i, m in enumerate(months_lbl):
        col = cols[i % len(cols)]
        with col:
            if st.button(pd.Period(m).strftime("%b %Y"), key=f"btn_{m}"):
                st.session_state["selected_month"] = pd.Period(m).to_timestamp()
    if st.session_state.get("selected_month"):
        st.success(f"📅 Supplier section focused on {st.session_state['selected_month']:%B %Y}")
        if st.button("Clear Month Filter"):
            st.session_state["selected_month"] = None
else:
    st.info("No data for the selected period.")

sel_month = st.session_state.get("selected_month", None)

# Supplier detail frame (respect sidebar filters + selected month)
detail_df = fdf.copy()
if sel_month is not None and "Created Date" in detail_df.columns:
    detail_df = detail_df[detail_df["Created Date"].dt.to_period("M") == sel_month.to_period("M")]

# -------------------- SUPPLIER FOCUS --------------------
st.markdown("---")
st.markdown("## Supplier Focus")

if supplier_col and (sel_suppliers or not suppliers):
    suppliers_to_show = sel_suppliers if sel_suppliers else suppliers

    for sup in suppliers_to_show:
        sframe = detail_df[detail_df[supplier_col] == sup].copy()
        if sframe.empty:
            continue
        sframe["Value"] = sframe[value_col]

        st.markdown(f"### {sup}")
        s_k1, s_k2, s_k3 = st.columns(3)

        s_status = classify_supply_status(sframe)
        s_total = float(sframe["Value"].sum())
        s_sup_pct = (s_status["Status"].eq("Supplied in Full").mean()*100.0) if not s_status.empty else np.nan
        s_back_pct = (s_status["Status"].eq("Backordered").mean()*100.0) if not s_status.empty else np.nan

        s_k1.metric("Supplier Spend", f"${s_total:,.0f}")
        s_k2.metric("Delivered in Full %", f"{s_sup_pct:,.1f}%" if s_sup_pct == s_sup_pct else "—")
        s_k3.metric("Backordered %", f"{s_back_pct:,.1f}%" if s_back_pct == s_back_pct else "—")

        # Charts row: Supplier Spend Trend (line) + Supply Performance Pie
        c1, c2 = st.columns(2)

        # Supplier trend (line, spline, filled)
        if "Created Month" in sframe.columns and not sframe.empty:
            s_month = sframe.groupby("Created Month", as_index=False)["Value"].sum()
            fig_s_trend = px.line(s_month, x="Created Month", y="Value", title="Supplier Spend Trend")
            fig_s_trend.update_traces(mode="lines+markers", line=dict(shape="spline", color="#0ea5e9", width=3), fill="tozeroy", fillcolor="rgba(14,165,233,0.2)")
            fig_s_trend.update_layout(margin=dict(l=10,r=10,t=40,b=10), height=360)
            c1.plotly_chart(fig_s_trend, use_container_width=True)
        else:
            c1.info("No data for spend trend.")

        # Supply performance pie (single chart)
        pie_df = s_status.groupby("Status").size().reset_index(name="Count")
        if not pie_df.empty:
            fig_pie = px.pie(pie_df, values="Count", names="Status", title="Supply Performance", hole=0.3)
            fig_pie.update_traces(
                textposition='inside',
                textinfo='percent+label',
                hovertemplate="%{label}: %{value} POs"
            )

           # Display the chart once (removed duplicate rendering from plotly_events)
           c2.plotly_chart(fig_pie, use_container_width=True)

           # Disable interactive selection (no event rendering = no second black chart)
           clicked_filter = None

        else:
            c2.info("No PO supply status available.")

        # Most popular items table (Qty + Spend only)
        st.markdown("#### Most Popular Items Ordered")
        table_df = sframe.copy()
        if "Item Name" in table_df.columns and "Item Code" in table_df.columns:
            if 'clicked_filter' in locals() and clicked_filter in ["Supplied in Full","Backordered"]:
                merge_status = s_status.copy()
                table_df = table_df.merge(merge_status, on="Base PO", how="left")
                table_df = table_df[table_df["Status"] == clicked_filter]
            agg = (table_df.groupby(["Item Code","Item Name"], as_index=False)
                          .agg(Qty_Ordered=("Item Qty","sum"),
                               Spend=("Value","sum"))
                          .sort_values("Spend", ascending=False)
                          .head(15))
            st.dataframe(agg, use_container_width=True, height=320)
        else:
            st.info("Item details not available in this dataset.")

else:
    st.info("Use the sidebar to select suppliers to analyze.")

# -------------------- PDF EXPORT (bottom with spinner) --------------------
st.markdown("---")
st.markdown("### 📤 Generate Management PDF")

def fig_to_png_bytes(fig):
    return pio.to_image(fig, format="png", scale=2)

def export_pdf_for_suppliers(df_filtered: pd.DataFrame, suppliers_list, filename: str, start_ts, end_ts):
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas
    from reportlab.lib.units import mm
    from reportlab.lib.utils import ImageReader
    from reportlab.lib import colors
    from reportlab.platypus import Table, TableStyle

    pdf_path = Path(filename)
    c = canvas.Canvas(str(pdf_path), pagesize=A4)
    W, H = A4

    for sup in suppliers_list:
        sframe = df_filtered[df_filtered[supplier_col] == sup].copy()
        if sframe.empty:
            continue
        sframe["Value"] = sframe[value_col]

        # Header (centered title)
        if logo_path.exists():
            c.drawImage(ImageReader(str(logo_path)), 15*mm, H-30*mm, width=30*mm, preserveAspectRatio=True, mask='auto')
        c.setFont("Helvetica-Bold", 16)
        c.drawCentredString(W/2, H-20*mm, "Supplier Purchase Summary")
        c.setFont("Helvetica", 11)
        c.drawCentredString(W/2, H-27*mm, f"Supplier: {sup}")
        c.drawCentredString(W/2, H-34*mm, f"Date range: {start_ts.strftime('%b %Y')} — {(end_ts - pd.Timedelta(days=1)).strftime('%b %Y')}")
        c.drawCentredString(W/2, H-41*mm, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")

        # KPIs
        status = classify_supply_status(sframe)
        s_total = float(sframe["Value"].sum())
        counts = status["Status"].value_counts()
        sup_cnt = int(counts.get("Supplied in Full", 0))
        back_cnt = int(counts.get("Backordered", 0))
        total_pos = int(len(status)) if not status.empty else 0
        sup_pct = (sup_cnt/total_pos*100.0) if total_pos else 0.0
        back_pct = (back_cnt/total_pos*100.0) if total_pos else 0.0

        kpi_data = [
            ["Total Spend", f"${s_total:,.0f}"],
            ["Delivered in Full %", f"{sup_pct:,.1f}%"],
            ["Backordered %", f"{back_pct:,.1f}%"],
            ["PO Count", f"{total_pos:,}"],
        ]
        t = Table(kpi_data, colWidths=[55*mm, 45*mm])
        t.setStyle(TableStyle([
            ("BACKGROUND", (0,0), (-1,0), colors.lightgrey),
            ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
            ("BOX", (0,0), (-1,-1), 0.25, colors.black),
            ("INNERGRID", (0,0), (-1,-1), 0.25, colors.black),
            ("ALIGN", (1,0), (1,-1), "RIGHT"),
        ]))
        w, h = t.wrapOn(c, W-30*mm, H)
        t.drawOn(c, 15*mm, H-60*mm-h)

        # Charts side-by-side (70% width combined)
        pie_df = status.groupby("Status").size().reset_index(name="Count")
        x_left = 15*mm
        x_right = 110*mm
        chart_y = H-120*mm-h
        if not pie_df.empty:
            fig_pie = px.pie(pie_df, values="Count", names="Status", title="Supply Performance", hole=0.3)
            fig_pie.update_traces(textposition='inside', textinfo='percent+label')
            pie_img = ImageReader(io.BytesIO(fig_to_png_bytes(fig_pie)))
            c.drawImage(pie_img, x_right, chart_y, width=85*mm, preserveAspectRatio=True, mask='auto')

        if "Created Month" in sframe.columns:
            s_month = sframe.groupby("Created Month", as_index=False)["Value"].sum()
            if not s_month.empty:
                fig_line = px.line(s_month, x="Created Month", y="Value", title="Supplier Spend Trend")
                fig_line.update_traces(mode="lines+markers",
                                       line=dict(shape="spline", color="#0ea5e9", width=3),
                                       fill="tozeroy", fillcolor="rgba(14,165,233,0.2)")
                line_img = ImageReader(io.BytesIO(fig_to_png_bytes(fig_line)))
                c.drawImage(line_img, x_left, chart_y, width=80*mm, preserveAspectRatio=True, mask='auto')

        # Top items table
        items = (sframe.groupby(["Item Code","Item Name"], as_index=False)
                        .agg(Qty_Ordered=("Item Qty","sum"), Spend=("Value","sum"))
                        .sort_values("Spend", ascending=False).head(15))
        table_y = chart_y - 70*mm
        if not items.empty:
            items["Spend"] = items["Spend"].round(2)
            data = [["Item Code","Item Name","Qty Ordered","Spend ($)"]] + items.values.tolist()
            from reportlab.platypus import Table, TableStyle
            from reportlab.lib import colors
            t2 = Table(data, colWidths=[30*mm, 80*mm, 25*mm, 30*mm])
            t2.setStyle(TableStyle([
                ("BACKGROUND", (0,0), (-1,0), colors.lightgrey),
                ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
                ("BOX", (0,0), (-1,-1), 0.25, colors.black),
                ("INNERGRID", (0,0), (-1,-1), 0.25, colors.black),
                ("ALIGN", (2,1), (2,-1), "RIGHT"),
                ("ALIGN", (3,1), (3,-1), "RIGHT"),
            ]))
            w2, h2 = t2.wrapOn(c, W-30*mm, H)
            t2.drawOn(c, 15*mm, max(20*mm, table_y-h2))

        c.showPage()

    c.save()
    return str(pdf_path)

# Export button at bottom with spinner + naming
export_suppliers = sel_suppliers if sel_suppliers else suppliers
if st.button("📤 Generate Management PDF"):
    if not export_suppliers:
        st.warning("No suppliers to export.")
    else:
        # File naming per user spec
        label = "Multiple_Suppliers" if len(export_suppliers) > 1 else export_suppliers[0]
        mm_yy = (start_ts.strftime("%m") + "-" + (end_ts - pd.Timedelta(days=1)).strftime("%y")) if (start_ts and end_ts) else datetime.now().strftime("%m-%y")
        filename = f"{label}_PO_REPORT_{mm_yy}.pdf".replace("/", "-")

        with st.spinner("⏳ Generating PDF report — please wait..."):
            path = export_pdf_for_suppliers(fdf, export_suppliers, filename, start_ts or df['Created Date'].min(), end_ts or df['Created Date'].max()+pd.Timedelta(days=1))

        st.success(f"✅ PDF Ready! {filename}")
        with open(path, "rb") as f:
            st.download_button("📥 Download Management PDF", f, file_name=filename, mime="application/pdf")

# Auto-refresh
if auto_refresh:
    now = time.time()
    if "last_refresh" not in st.session_state:
        st.session_state["last_refresh"] = now
    if now - st.session_state["last_refresh"] >= refresh_sec:
        st.session_state["last_refresh"] = now
        st.rerun()
