"""
Regional Sales ML App  -  Streamlit
Run:  streamlit run app.py
Tabs: Data Analysis | Data Visualization | Predictive Analysis | Forecasting
"""
import io
import os

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import core

st.set_page_config(page_title="Regional Sales ML App", page_icon="📈", layout="wide")

# ----------------------------------------------------------------------------- helpers
_VER = tuple(int(p) for p in st.__version__.split(".")[:2] if p.isdigit())
_NEW = _VER >= (1, 50)


def plot(fig, key=None):
    """Version-safe st.plotly_chart (use_container_width was replaced by width='stretch')."""
    fig.update_layout(template="plotly_white", margin=dict(l=10, r=10, t=50, b=10))
    if _NEW:
        st.plotly_chart(fig, width="stretch", key=key)
    else:
        st.plotly_chart(fig, use_container_width=True, key=key)


def table(data, **kw):
    if _NEW:
        st.dataframe(data, width="stretch", **kw)
    else:
        st.dataframe(data, use_container_width=True, **kw)


def human(x, prefix=""):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "–"
    a = abs(x)
    for div, suf in ((1e9, "B"), (1e6, "M"), (1e3, "K")):
        if a >= div:
            return f"{prefix}{x / div:,.2f}{suf}"
    return f"{prefix}{x:,.0f}"


def csv_button(data: pd.DataFrame, label: str, name: str, index=True, key=None):
    st.download_button(label, data.to_csv(index=index).encode("utf-8"), file_name=name, mime="text/csv", key=key)


DEFAULT_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sales_data.csv")


@st.cache_data(show_spinner="Loading & cleaning dataset…")
def load_data(file_bytes, name, path):
    if file_bytes is not None:
        return core.load_dataset(io.BytesIO(file_bytes), name)
    return core.load_dataset(path)


@st.cache_data(show_spinner="Running forecasting models…")
def cached_forecast(y, horizon, test_months, models, selected, ci):
    return core.run_forecast(y, horizon, test_months, list(models), selected, ci)


# ----------------------------------------------------------------------------- sidebar: data
st.sidebar.title("📈 Sales ML App")
src = st.sidebar.radio("Data source", ["Bundled dataset", "Upload file"], horizontal=True)
try:
    if src == "Upload file":
        up = st.sidebar.file_uploader("Excel / CSV (same schema)", type=["xlsx", "xls", "csv"])
        if up is None:
            st.info("⬅️ Upload an Excel/CSV file with the regional sales schema, or switch to the bundled dataset.")
            st.stop()
        df = load_data(up.getvalue(), up.name, "")
    else:
        if not os.path.exists(DEFAULT_FILE):
            st.error(f"Bundled file not found: {DEFAULT_FILE}. Use 'Upload file' instead.")
            st.stop()
        df = load_data(None, "", DEFAULT_FILE)
except ValueError as e:
    st.error(str(e))
    st.stop()

# ----------------------------------------------------------------------------- sidebar: filters
st.sidebar.markdown("---")
st.sidebar.subheader("🔎 Filters")
st.sidebar.caption("Apply to **Data Analysis** and **Data Visualization** tabs. "
                   "Predictive & Forecasting tabs use their own settings.")
dmin, dmax = df["Date"].min().date(), df["Date"].max().date()
rng = st.sidebar.date_input("Date range", (dmin, dmax), min_value=dmin, max_value=dmax)
if not (isinstance(rng, (tuple, list)) and len(rng) == 2):
    rng = (dmin, dmax)
f_scen = st.sidebar.multiselect("Scenario", sorted(df["Scenario Name"].unique()), default=sorted(df["Scenario Name"].unique()))
f_city = st.sidebar.multiselect("City", sorted(df["Cities"].unique()), default=sorted(df["Cities"].unique()))
f_cat = st.sidebar.multiselect("Category", sorted(df["Category"].unique()), default=sorted(df["Category"].unique()))
f_brand = st.sidebar.multiselect("Brand", sorted(df["Brand"].unique()), default=sorted(df["Brand"].unique()))

mask = (
    (df["Date"].dt.date >= rng[0]) & (df["Date"].dt.date <= rng[1])
    & df["Scenario Name"].isin(f_scen) & df["Cities"].isin(f_city)
    & df["Category"].isin(f_cat) & df["Brand"].isin(f_brand)
)
fdf = df[mask]
st.sidebar.success(f"{len(fdf):,} of {len(df):,} rows selected")

DIMS = [c for c in ["Year", "Quarter", "Month", "YearMonth", "Category", "Brand", "Warranty", "Cities",
                    "Product Name", "SalesPerson Name", "Designation", "Scenario Name"] if c in df.columns]
METRICS = [c for c in ["Sales", "Profit", "Units Sold", "Cost of sales", "Margin %", "Sale price", "Cost price"] if c in df.columns]

st.title("📈 Regional Sales – Analytics, Prediction & Forecasting")
tab1, tab2, tab3, tab4 = st.tabs(["🔬 Data Analysis", "📊 Data Visualization", "🤖 Predictive Analysis", "🔮 Forecasting"])


def summarize(d, by):
    g = d.groupby(by, observed=True).agg(Sales=("Sales", "sum"), Profit=("Profit", "sum"), Units=("Units Sold", "sum"),
                                         Orders=("Sales", "size"), Avg_Sale=("Sales", "mean"))
    g["Margin %"] = g["Profit"] / g["Sales"] * 100
    return g.sort_values("Sales", ascending=False)


# =============================================================================
# TAB 1 - DATA ANALYSIS
# =============================================================================
with tab1:
    if fdf.empty:
        st.warning("No rows match the current filters.")
    else:
        k = st.columns(6)
        k[0].metric("Total Sales", human(fdf["Sales"].sum()))
        k[1].metric("Total Profit", human(fdf["Profit"].sum()))
        k[2].metric("Profit Margin", f"{fdf['Profit'].sum() / fdf['Sales'].sum() * 100:.1f}%")
        k[3].metric("Units Sold", f"{int(fdf['Units Sold'].sum()):,}")
        k[4].metric("Transactions", f"{len(fdf):,}")
        k[5].metric("Avg Sale / Txn", human(fdf["Sales"].mean()))

        st.subheader("1 · Dataset overview")
        c1, c2 = st.columns([2, 1])
        with c1:
            st.caption("First 100 filtered rows")
            table(fdf.drop(columns=["YearMonth", "t"], errors="ignore").head(100))
        with c2:
            info = pd.DataFrame({"dtype": fdf.dtypes.astype(str), "non-null": fdf.notna().sum(),
                                 "unique": fdf.nunique()})
            st.caption(f"Shape: **{fdf.shape[0]:,} rows × {fdf.shape[1]} cols**  ·  "
                       f"{fdf['Date'].min():%d %b %Y} → {fdf['Date'].max():%d %b %Y}")
            table(info, height=330)

        st.subheader("2 · Data quality checks")
        q1, q2, q3, q4 = st.columns(4)
        q1.metric("Missing cells", f"{int(fdf.isna().sum().sum()):,}")
        q2.metric("Duplicate rows", f"{int(fdf.drop(columns=['Index'], errors='ignore').duplicated().sum()):,}")
        if "Sale price" in fdf.columns:
            bad = (fdf["Sales"] != fdf["Units Sold"] * fdf["Sale price"]).mean() * 100
            q3.metric("Sales ≠ Units × Price", f"{bad:.2f}% of rows")
        Q1, Q3 = fdf["Sales"].quantile([0.25, 0.75])
        iqr = Q3 - Q1
        out = fdf[(fdf["Sales"] > Q3 + 1.5 * iqr) | (fdf["Sales"] < Q1 - 1.5 * iqr)]
        q4.metric("Sales outliers (IQR)", f"{len(out):,} ({len(out) / len(fdf) * 100:.1f}%)")
        if fdf.isna().sum().sum() == 0:
            st.success("No missing values detected.")
        with st.expander("Outlier details – largest sales"):
            table(out.nlargest(50, "Sales")[["Date", "Product Name", "Cities", "Units Sold", "Sales", "Scenario Name"]])
            st.caption("High-value outliers here are mostly expensive items (laptops / graphic cards) – "
                       "legitimate sales rather than errors.")

        st.subheader("3 · Descriptive statistics")
        num_cols = [c for c in ["Units Sold", "Sales", "Cost of sales", "Profit", "Margin %", "Sale price", "Cost price", "Age"]
                    if c in fdf.columns]
        table(fdf[num_cols].describe().T.round(2))
        cc1, cc2 = st.columns([1, 2])
        cat_col = cc1.selectbox("Frequency of…", ["Category", "Brand", "Warranty", "Cities", "Product Name",
                                                  "SalesPerson Name", "Designation", "Scenario Name"])
        vc = fdf[cat_col].value_counts().rename("Transactions").to_frame()
        vc["Share %"] = (vc["Transactions"] / vc["Transactions"].sum() * 100).round(2)
        with cc2:
            table(vc, height=250)

        st.subheader("4 · Group-by analysis")
        g1, g2 = st.columns([2, 1])
        by = g1.multiselect("Group by", [d for d in DIMS if d != "YearMonth"], default=["Category"])
        sort_by = g2.selectbox("Sort by", ["Sales", "Profit", "Units", "Orders", "Avg_Sale", "Margin %"])
        if by:
            gt = summarize(fdf, by).sort_values(sort_by, ascending=False)
            table(gt.round(2))
            csv_button(gt.round(2), "⬇️ Download table", "group_analysis.csv", key="dl_group")

        st.subheader("5 · Actual vs Budget")
        if {"Actual", "Budget"} <= set(fdf["Scenario Name"].unique()):
            av_dim = st.selectbox("Compare by", [d for d in DIMS if d not in ("YearMonth", "Scenario Name")], index=0, key="av_dim")
            pv = fdf.pivot_table(index=av_dim, columns="Scenario Name", values="Sales", aggfunc="sum")
            pv["Variance"] = pv["Actual"] - pv["Budget"]
            pv["Variance %"] = pv["Variance"] / pv["Budget"] * 100
            table(pv.round(1))
            st.caption("Positive variance = actual sales above budget.")
        else:
            st.info("Select both 'Actual' and 'Budget' in the Scenario filter to compare them.")

        st.subheader("6 · Correlation matrix")
        corr_cols = [c for c in ["Units Sold", "Sales", "Cost of sales", "Profit", "Margin %", "Sale price", "Cost price",
                                 "Age", "Year", "Month"] if c in fdf.columns]
        cm = fdf[corr_cols].corr()
        plot(px.imshow(cm, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1, aspect="auto",
                       title="Pearson correlation"), key="corr")

        st.subheader("7 · Pivot-table builder")
        p1, p2, p3, p4 = st.columns(4)
        pidx = p1.multiselect("Rows", [d for d in DIMS if d != "YearMonth"], default=["Category"], key="pv_i")
        pcol = p2.selectbox("Columns", ["(none)"] + [d for d in DIMS if d != "YearMonth"], index=1, key="pv_c")
        pval = p3.selectbox("Values", METRICS, key="pv_v")
        pagg = p4.selectbox("Aggregation", ["sum", "mean", "median", "count", "max", "min"], key="pv_a")
        if pidx:
            pt = pd.pivot_table(fdf, index=pidx, columns=None if pcol == "(none)" else pcol, values=pval,
                                aggfunc=pagg, observed=True)
            table(pt.round(2))
            csv_button(pt.round(2), "⬇️ Download pivot", "pivot.csv", key="dl_pivot")

# =============================================================================
# TAB 2 - DATA VISUALIZATION
# =============================================================================
with tab2:
    if fdf.empty:
        st.warning("No rows match the current filters.")
    else:
        vm = st.selectbox("Metric to visualise", [m for m in ["Sales", "Profit", "Units Sold", "Cost of sales"] if m in fdf.columns], key="vm")

        st.markdown("#### Trends")
        mt = fdf.groupby(["YearMonth", "Scenario Name"])[vm].sum().reset_index()
        plot(px.line(mt, x="YearMonth", y=vm, color="Scenario Name", title=f"Monthly {vm}"), key="v_month")
        yt = fdf.groupby(["Year", "Scenario Name"])[vm].sum().reset_index()
        yfig = px.bar(yt, x="Year", y=vm, color="Scenario Name", barmode="group", title=f"Yearly {vm}")
        a, b = st.columns(2)
        with a:
            plot(yfig, key="v_year")
        with b:
            qt = fdf.groupby(["Year", "Quarter"])[vm].sum().reset_index()
            qt["Period"] = qt["Year"].astype(str) + " Q" + qt["Quarter"].astype(str)
            plot(px.bar(qt, x="Period", y=vm, title=f"Quarterly {vm}"), key="v_qtr")

        st.markdown("#### Mix: products, categories, cities")
        a, b = st.columns(2)
        with a:
            cat = fdf.groupby(["Category", "Brand"])[vm].sum().reset_index()
            plot(px.treemap(cat, path=["Category", "Brand"], values=vm, title=f"{vm} by Category → Brand"), key="v_tree")
        with b:
            ct = fdf.groupby(["Cities", "Scenario Name"])[vm].sum().reset_index()
            plot(px.bar(ct, x="Cities", y=vm, color="Scenario Name", barmode="group", title=f"{vm} by City"), key="v_city")

        a, b = st.columns(2)
        with a:
            topn = st.slider("Top N products", 5, 30, 10, key="topn")
            tp = fdf.groupby("Product Name")[vm].sum().nlargest(topn).sort_values().reset_index()
            plot(px.bar(tp, x=vm, y="Product Name", orientation="h", title=f"Top {topn} products by {vm}"), key="v_top")
        with b:
            lb = fdf.groupby(["SalesPerson Name", "Designation"])[vm].sum().reset_index().sort_values(vm)
            plot(px.bar(lb, x=vm, y="SalesPerson Name", color="Designation", orientation="h",
                        title=f"Salesperson leaderboard ({vm})"), key="v_rep")

        st.markdown("#### Seasonality & geography")
        a, b = st.columns(2)
        with a:
            hm = fdf.pivot_table(index="Year", columns="Month", values=vm, aggfunc="sum")
            plot(px.imshow(hm, aspect="auto", color_continuous_scale="YlGnBu", labels=dict(color=vm),
                           title=f"{vm}: Year × Month heat-map"), key="v_heat")
        with b:
            ym = fdf.groupby(["Year", "Month"])[vm].sum().reset_index()
            sea = ym.groupby("Month")[vm].mean().reset_index()
            plot(px.bar(sea, x="Month", y=vm, title=f"Average {vm} by calendar month (seasonality)"), key="v_season")

        if {"Latitude", "Longitude"} <= set(fdf.columns):
            geo = fdf.groupby(["Cities", "Latitude", "Longitude"])[[vm]].sum().reset_index()
            gfig = px.scatter_geo(geo, lat="Latitude", lon="Longitude", size=vm, color=vm, hover_name="Cities",
                                  color_continuous_scale="Viridis", size_max=45, title=f"{vm} by city (map)")
            gfig.update_geos(fitbounds="locations", showcountries=True, showland=True, landcolor="#f1f1f1")
            plot(gfig, key="v_map")

        st.markdown("#### Profitability & pricing")
        a, b = st.columns(2)
        with a:
            mg = fdf.groupby("Category").agg(Sales=("Sales", "sum"), Profit=("Profit", "sum")).reset_index()
            mg["Margin %"] = mg["Profit"] / mg["Sales"] * 100
            plot(px.bar(mg.sort_values("Margin %"), x="Margin %", y="Category", orientation="h",
                        color="Margin %", color_continuous_scale="Greens", title="Profit margin by category"), key="v_margin")
        with b:
            if "Sale price" in fdf.columns:
                pr = fdf.groupby(["Year", "Category"])["Sale price"].mean().reset_index()
                plot(px.line(pr, x="Year", y="Sale price", color="Category", markers=True,
                             title="Average selling price trend by category"), key="v_price")

        st.markdown("#### Distribution")
        a, b = st.columns(2)
        sample = fdf.sample(min(20000, len(fdf)), random_state=1)
        with a:
            plot(px.histogram(sample, x="Sales", nbins=60, color="Scenario Name", barmode="overlay", opacity=0.6,
                              title="Sales per transaction (20k sample)"), key="v_hist")
        with b:
            plot(px.box(sample, x="Category", y="Sales", color="Category", title="Sales distribution by category"), key="v_box")

        st.markdown("#### 🛠️ Build your own chart")
        b1, b2, b3, b4, b5 = st.columns(5)
        ctype = b1.selectbox("Chart", ["Bar", "Line", "Area", "Pie", "Box", "Histogram"])
        cx = b2.selectbox("X / group", DIMS, index=DIMS.index("Category") if "Category" in DIMS else 0, key="cx")
        cy = b3.selectbox("Y metric", METRICS, key="cy")
        cagg = b4.selectbox("Aggregate", ["sum", "mean", "median", "count", "max", "min"], key="cagg")
        ccol = b5.selectbox("Colour by", ["(none)"] + DIMS, key="ccol")
        col = None if ccol == "(none)" or ccol == cx else ccol
        if ctype in ("Box", "Histogram"):
            sm = fdf.sample(min(20000, len(fdf)), random_state=2)
            if ctype == "Box":
                cfig = px.box(sm, x=cx, y=cy, color=col)
            else:
                cfig = px.histogram(sm, x=cy, color=col, nbins=60, barmode="overlay", opacity=0.7)
        else:
            keys = [cx] + ([col] if col else [])
            g = fdf.groupby(keys, observed=True)[cy].agg(cagg).reset_index()
            if ctype == "Bar":
                cfig = px.bar(g, x=cx, y=cy, color=col, barmode="group")
            elif ctype == "Line":
                cfig = px.line(g, x=cx, y=cy, color=col, markers=True)
            elif ctype == "Area":
                cfig = px.area(g, x=cx, y=cy, color=col)
            else:
                cfig = px.pie(g, names=cx, values=cy)
        cfig.update_layout(title=f"{ctype}: {cagg} of {cy} by {cx}")
        plot(cfig, key="v_custom")

# =============================================================================
# TAB 3 - PREDICTIVE ANALYSIS
# =============================================================================
with tab3:
    st.subheader("1 · Train & compare regression models")
    st.caption("Models learn from **all** rows of the dataset (sidebar filters are ignored here). "
               "Price / cost columns are deliberately excluded because `Sales = Units × Sale price` (that would be target leakage).")
    c1, c2, c3 = st.columns(3)
    target = c1.selectbox("Target to predict", core.TARGETS, help="Units Sold is intrinsically noisy – expect a low R².")
    split_mode = c2.radio("Validation split", ["time", "random"], horizontal=True,
                          format_func=lambda x: "Time-based (train on past → test on recent)" if x == "time" else "Random")
    test_frac = c3.slider("Test share", 0.10, 0.40, 0.20, 0.05)
    feats = st.multiselect("Features", core.ALL_FEATURES, default=core.ALL_FEATURES)
    models = st.multiselect("Models", core.MODEL_NAMES, default=core.MODEL_NAMES)

    if st.button("🚀 Train models", type="primary"):
        if not feats or not models:
            st.warning("Pick at least one feature and one model.")
        else:
            with st.spinner("Training… (Random Forest takes the longest)"):
                st.session_state["res"] = core.train_and_evaluate(df, target, feats, models, split_mode, test_frac)
                st.session_state["fi"] = {}
                st.session_state["full"] = {}

    res = st.session_state.get("res")
    if res is None:
        st.info("Configure the options above and click **Train models**.")
    else:
        st.success(f"Target: **{res.target}** · {res.split_desc} · train rows: {res.train_size:,} · test rows: {len(res.test):,}")
        met = res.metrics.copy()
        table(met.style.format({"R²": "{:.4f}", "MAE": "{:,.1f}", "RMSE": "{:,.1f}", "MAPE %": "{:.2f}"}))
        mc1, mc2 = st.columns(2)
        with mc1:
            plot(px.bar(met, x="Model", y="R²", color="Model", title="R² (higher is better)", text_auto=".3f"), key="m_r2")
        with mc2:
            plot(px.bar(met, x="Model", y="RMSE", color="Model", title="RMSE (lower is better)", text_auto=".3s"), key="m_rmse")
        st.caption(f"🏆 Best model on test data: **{res.best_model}**")

        st.subheader("2 · Model diagnostics")
        dm = st.selectbox("Inspect model", list(res.pipelines), index=list(res.pipelines).index(res.best_model))
        pred = res.predictions[dm]
        tdf = res.test.assign(Predicted=pred)
        tdf["Residual"] = tdf[res.target] - tdf["Predicted"]
        d1, d2 = st.columns(2)
        with d1:
            s = tdf.sample(min(4000, len(tdf)), random_state=0)
            fig = px.scatter(s, x=res.target, y="Predicted", color="Category", opacity=0.5, title="Actual vs Predicted (4k sample)")
            lim = [0, float(max(s[res.target].max(), s["Predicted"].max()))]
            fig.add_trace(go.Scatter(x=lim, y=lim, mode="lines", line=dict(dash="dash", color="black"), name="Perfect"))
            plot(fig, key="d_scatter")
        with d2:
            plot(px.histogram(tdf, x="Residual", nbins=80, title="Residuals (Actual − Predicted)"), key="d_resid")
        d3, d4 = st.columns(2)
        with d3:
            ts = tdf.groupby("YearMonth")[[res.target, "Predicted"]].sum().reset_index()
            plot(px.line(ts.melt("YearMonth"), x="YearMonth", y="value", color="variable",
                         title=f"Monthly total {res.target}: actual vs predicted (test period)"), key="d_ts")
        with d4:
            if dm not in st.session_state["fi"]:
                with st.spinner("Computing permutation importance…"):
                    st.session_state["fi"][dm] = core.feature_importance(res.pipelines[dm], res.test, res.features, res.target)
            fi = st.session_state["fi"][dm]
            plot(px.bar(fi, x="Importance", y="Feature", orientation="h", error_x="Std",
                        title="Permutation importance (drop in R² when shuffled)"), key="d_fi")
        csv_button(tdf[["Date", "Product Name", "Cities", "SalesPerson Name", "Scenario Name", res.target, "Predicted"]],
                   "⬇️ Download test predictions", "test_predictions.csv", index=False, key="dl_pred")

        # ---------------------------------------------------------------- what-if
        st.subheader("3 · 🎯 Predict a sale (what-if)")
        tb = core.lookup_tables(df)
        with st.form("whatif"):
            w1, w2, w3 = st.columns(3)
            prod = w1.selectbox("Product", tb["product"]["Product Name"])
            city = w2.selectbox("City", tb["city"]["Cities"])
            rep = w3.selectbox("Salesperson", tb["rep"]["SalesPerson Name"])
            w4, w5, w6 = st.columns(3)
            scen = w4.selectbox("Scenario", tb["scenario"]["Scenario Name"])
            when = w5.date_input("Sale date", value=(df["Date"].max() + pd.Timedelta(days=30)).date())
            units = w6.slider("Units sold", 1, int(df["Units Sold"].max()), 3,
                              disabled=("Units Sold" not in res.features))
            wm = st.selectbox("Model", list(res.pipelines), index=list(res.pipelines).index(res.best_model))
            go_pred = st.form_submit_button("Predict", type="primary")

        if go_pred:
            pid = int(tb["product"].loc[tb["product"]["Product Name"] == prod, "ProductID"].iloc[0])
            cid = int(tb["city"].loc[tb["city"]["Cities"] == city, "CityID"].iloc[0])
            rid = int(tb["rep"].loc[tb["rep"]["SalesPerson Name"] == rep, "SalesrepID"].iloc[0])
            sid = int(tb["scenario"].loc[tb["scenario"]["Scenario Name"] == scen, "ScenarioID"].iloc[0])
            key = (wm, res.target, tuple(res.features))
            if key not in st.session_state["full"]:
                with st.spinner("Refitting the model on all data…"):
                    st.session_state["full"][key] = core.fit_full(wm, df, res.target, res.features)
            fm = st.session_state["full"][key]
            row = core.build_input_row(when, pid, cid, rid, sid, units, tb, df["Date"].min())
            p = float(fm.predict(row[res.features])[0])
            if res.target != "Profit":
                p = max(p, 0.0)
            lo_q, hi_q = res.rel_err_q[wm]
            r1, r2, r3 = st.columns(3)
            r1.metric(f"Predicted {res.target}", human(p))
            r2.metric("Likely range (10–90%)", f"{human(p * lo_q)} – {human(p * hi_q)}")
            if res.target == "Sales" and "Units Sold" in res.features:
                r3.metric("Implied price per unit", human(p / units))
            if when > df["Date"].max().date():
                st.warning("Date is beyond the training period. Tree models hold prices at the latest level they have seen; "
                           "Ridge extrapolates the trend. Treat as an estimate.")
            if res.target != "Units Sold" and "Units Sold" in res.features:
                rows = []
                for u in range(1, int(df["Units Sold"].max()) + 1):
                    rr = core.build_input_row(when, pid, cid, rid, sid, u, tb, df["Date"].min())
                    rows.append({"Units": u, res.target: max(float(fm.predict(rr[res.features])[0]), 0.0)})
                plot(px.bar(pd.DataFrame(rows), x="Units", y=res.target, text_auto=".3s",
                            title=f"Predicted {res.target} by units sold (same product/city/rep/date)"), key="w_units")

# =============================================================================
# TAB 4 - FORECASTING
# =============================================================================
with tab4:
    st.subheader("Monthly time-series forecasting")
    st.caption("Models are back-tested on the most recent months, ranked, then re-fitted on all history to forecast the future.")
    f1, f2, f3, f4 = st.columns(4)
    fm_metric = f1.selectbox("Metric", core.SERIES_METRICS, key="fm")
    fm_scen = f2.selectbox("Scenario", ["Actual", "Budget", "All"], key="fs")
    scope_dim = f3.selectbox("Scope", ["Total company", "Category", "Cities", "Brand", "Product Name", "SalesPerson Name", "Warranty"], key="fd")
    scope_val = None
    if scope_dim != "Total company":
        scope_val = f4.selectbox(scope_dim, sorted(df[scope_dim].unique()), key="fv")

    g1, g2, g3, g4 = st.columns(4)
    horizon = g1.slider("Forecast horizon (months)", 1, 24, 12)
    test_m = g2.slider("Back-test window (months)", 3, 18, 12)
    ci = g3.select_slider("Interval", options=[0.80, 0.90, 0.95], value=0.95, format_func=lambda x: f"{int(x * 100)}%")
    pick = g4.selectbox("Forecast model", ["Auto (best on back-test)"] + list(core.FORECAST_MODELS) + [core.ENSEMBLE])
    fmodels = st.multiselect("Candidate models", list(core.FORECAST_MODELS), default=list(core.FORECAST_MODELS))

    y = core.monthly_series(df, fm_metric, fm_scen, None if scope_dim == "Total company" else scope_dim, scope_val)
    if not fmodels:
        st.warning("Select at least one candidate model.")
    elif len(y) == 0:
        st.warning("No data for this selection.")
    else:
        try:
            sel = None if pick.startswith("Auto") else pick
            if sel == core.ENSEMBLE and len(fmodels) < 2:
                sel = None
            fr = cached_forecast(y, horizon, test_m, tuple(fmodels), sel, ci)
        except ValueError as e:
            st.error(str(e))
            st.stop()

        best_row = fr.scores.set_index("Model").loc[fr.best_model]
        h_cmp = min(horizon, 12)
        nxt = fr.future["Forecast"].iloc[:h_cmp].sum()
        prev = y.iloc[len(y) - 12: len(y) - 12 + h_cmp].sum()
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Selected model", fr.best_model)
        k2.metric("Back-test MAPE", f"{best_row['MAPE %']:.1f}%")
        k3.metric(f"Forecast next {h_cmp} mo", human(nxt), f"{(nxt / prev - 1) * 100:+.1f}% vs same months last year" if prev else None)
        k4.metric(f"Forecast month {horizon}", human(fr.future['Forecast'].iloc[-1]))

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=y.index, y=y.values, name="History", mode="lines", line=dict(color="#1f77b4")))
        bt = fr.backtest_preds[fr.best_model]
        fig.add_trace(go.Scatter(x=bt.index, y=bt.values, name="Back-test prediction", mode="lines",
                                 line=dict(color="#ff7f0e", dash="dot")))
        fx = pd.concat([pd.Series([y.iloc[-1]], index=[y.index[-1]]), fr.future["Forecast"]])
        fig.add_trace(go.Scatter(x=fr.future.index, y=fr.future["Upper"], mode="lines", line=dict(width=0),
                                 showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=fr.future.index, y=fr.future["Lower"], mode="lines", line=dict(width=0), fill="tonexty",
                                 fillcolor="rgba(44,160,44,0.2)", name=f"{int(ci * 100)}% interval", hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=fx.index, y=fx.values, name="Forecast", mode="lines+markers", line=dict(color="#2ca02c")))
        fig.update_layout(title=f"{fm_metric} – {scope_val or 'Total company'} ({fm_scen}) · {fr.best_model}",
                          hovermode="x unified", yaxis_title=fm_metric)
        plot(fig, key="f_main")

        a, b = st.columns(2)
        with a:
            st.markdown("**Model leaderboard (back-test)**")
            table(fr.scores.style.format({"MAE": "{:,.0f}", "RMSE": "{:,.0f}", "MAPE %": "{:.2f}", "Bias": "{:,.0f}"}))
            st.caption("Bias > 0 means the model over-forecasts on average.")
        with b:
            cmp_fig = go.Figure()
            cmp_fig.add_trace(go.Scatter(x=fr.backtest_actual.index, y=fr.backtest_actual.values, name="Actual",
                                         line=dict(color="black", width=3)))
            for m, s in fr.backtest_preds.items():
                cmp_fig.add_trace(go.Scatter(x=s.index, y=s.values, name=m, mode="lines", line=dict(dash="dot")))
            cmp_fig.update_layout(title="Back-test: every model vs actual", hovermode="x unified")
            plot(cmp_fig, key="f_cmp")

        a, b = st.columns(2)
        with a:
            si = (y.groupby(y.index.month).mean() / y.mean()).reset_index()
            si.columns = ["Month", "Seasonal index"]
            plot(px.bar(si, x="Month", y="Seasonal index", title="Seasonal index (1.0 = average month)"), key="f_seas")
        with b:
            yy = y.groupby(y.index.year).sum().to_frame("Total")
            yy["YoY %"] = yy["Total"].pct_change() * 100
            st.markdown("**Annual totals & growth**")
            table(yy.style.format({"Total": "{:,.0f}", "YoY %": "{:+.1f}"}, na_rep="–"))

        st.markdown("**Forecast table**")
        out = fr.future[["Forecast", "Lower", "Upper"] + [c for c in fr.future.columns if c not in ("Forecast", "Lower", "Upper")]].round(0)
        out.index = out.index.strftime("%b %Y")
        table(out)
        csv_button(out, "⬇️ Download forecast", "forecast.csv", key="dl_fc")

        with st.expander("🧩 Forecast breakdown by segment (fast OLS trend + seasonality)"):
            bd = st.selectbox("Break down by", ["Category", "Cities", "Brand", "Warranty"], key="bd")
            bdf = core.forecast_breakdown(df, fm_metric, fm_scen, bd, horizon)
            if bdf.empty:
                st.info("Not enough history per segment.")
            else:
                long = bdf.reset_index().melt("index", var_name=bd, value_name=fm_metric).rename(columns={"index": "Month"})
                plot(px.bar(long, x="Month", y=fm_metric, color=bd, title=f"Forecast {fm_metric} by {bd}"), key="f_break")
                tot = bdf.sum().rename("Forecast total").to_frame()
                tot["Share %"] = tot["Forecast total"] / tot["Forecast total"].sum() * 100
                table(tot.sort_values("Forecast total", ascending=False).round(1))

st.markdown("---")
st.caption("Built with Streamlit · scikit-learn · Plotly.  Forecast intervals are approximate (based on back-test error).")
