"""
core.py  -  data loading, ML and forecasting logic for the Regional Sales ML app.
Pure pandas / numpy / scikit-learn / scipy (no Streamlit import) so it can be unit-tested.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

warnings.filterwarnings("ignore")

# --------------------------------------------------------------------------------------
# 1. DATA LOADING & CLEANING
# --------------------------------------------------------------------------------------
REQUIRED_COLS = ["Date", "ProductID", "SalesrepID", "CityID", "ScenarioID", "Units Sold", "Sales",
                 "Cost of sales", "Product Name", "Category", "Brand", "Warranty", "Cities",
                 "SalesPerson Name", "Scenario Name"]
NUMERIC_COLS = ["Index", "ProductID", "SalesrepID", "CityID", "ScenarioID", "Units Sold", "Sales",
                "Cost of sales", "Sale price", "Cost price", "Latitude", "Longitude", "Age"]


def _find_header_row(raw: pd.DataFrame) -> int:
    """The Excel file has blank rows above the header. Find the row that contains 'Date'."""
    for i in range(min(len(raw), 20)):
        vals = raw.iloc[i].astype(str).str.strip().str.lower().tolist()
        if "date" in vals and ("sales" in vals or "units sold" in vals):
            return i
    return 0


def load_dataset(source, filename: str | None = None) -> pd.DataFrame:
    """Load .xlsx / .csv (path or file-like), locate header row, fix dtypes, add derived columns."""
    name = (filename or (source if isinstance(source, str) else getattr(source, "name", ""))).lower()
    if name.endswith(".csv"):
        raw = pd.read_csv(source, header=None, low_memory=False)
    else:
        raw = pd.read_excel(source, header=None)
    raw = raw.dropna(how="all").reset_index(drop=True)
    h = _find_header_row(raw)
    df = raw.iloc[h + 1:].copy()
    df.columns = [str(c).strip() for c in raw.iloc[h]]
    df = df.dropna(axis=1, how="all").reset_index(drop=True)

    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")

    for c in NUMERIC_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in ["Date", "DOJ"]:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")
    df = df.dropna(subset=["Date", "Sales", "Units Sold"]).reset_index(drop=True)
    return add_features(df)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Year"] = df["Date"].dt.year
    df["Quarter"] = df["Date"].dt.quarter
    df["Month"] = df["Date"].dt.month
    df["MonthName"] = df["Date"].dt.strftime("%b")
    df["DayOfWeek"] = df["Date"].dt.dayofweek
    df["YearMonth"] = df["Date"].dt.to_period("M").dt.to_timestamp()
    df["t"] = (df["Date"] - df["Date"].min()).dt.days
    df["Profit"] = df["Sales"] - df["Cost of sales"]
    df["Margin %"] = np.where(df["Sales"] != 0, df["Profit"] / df["Sales"] * 100, np.nan)
    if "DOJ" in df.columns:
        df["Tenure (yrs)"] = ((df["Date"] - df["DOJ"]).dt.days / 365.25).clip(lower=0)
    return df


# --------------------------------------------------------------------------------------
# 2. PREDICTIVE MODELLING (row-level regression)
# --------------------------------------------------------------------------------------
CAT_FEATURES = ["ProductID", "CityID", "SalesrepID", "ScenarioID", "Category", "Brand", "Warranty"]
NUM_FEATURES = ["Units Sold", "Year", "Month", "Quarter", "DayOfWeek", "t"]
ALL_FEATURES = CAT_FEATURES + NUM_FEATURES
TARGETS = ["Sales", "Profit", "Units Sold"]
MODEL_NAMES = ["Ridge Regression", "Random Forest", "Gradient Boosting (HistGB)"]


def _make_model(name: str):
    if name == "Ridge Regression":
        return Ridge(alpha=1.0), True
    if name == "Random Forest":
        return RandomForestRegressor(n_estimators=80, max_depth=18, min_samples_leaf=3, max_samples=0.5,
                                     n_jobs=-1, random_state=42), False
    return HistGradientBoostingRegressor(max_iter=300, learning_rate=0.1, random_state=42), False


def make_pipeline(name: str, features: list[str]) -> Pipeline:
    cats = [f for f in features if f in CAT_FEATURES]
    nums = [f for f in features if f not in CAT_FEATURES]
    model, linear = _make_model(name)
    if linear:  # one-hot + scaling for linear model
        ct = ColumnTransformer([("c", OneHotEncoder(handle_unknown="ignore"), cats),
                                ("n", StandardScaler(), nums)])
    else:       # ordinal codes are enough (and much faster) for tree models
        ct = ColumnTransformer([("c", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), cats),
                                ("n", "passthrough", nums)])
    return Pipeline([("prep", ct), ("model", model)])


def regression_metrics(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    nz = y_true != 0
    return {
        "R²": r2_score(y_true, y_pred),
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAPE %": float(np.mean(np.abs((y_true[nz] - y_pred[nz]) / y_true[nz])) * 100) if nz.any() else np.nan,
    }


def split_data(df: pd.DataFrame, mode: str, test_frac: float, seed: int = 42):
    """mode: 'time' (train on past, test on most recent rows) or 'random'."""
    if mode == "time":
        d = df.sort_values("Date")
        cut = int(len(d) * (1 - test_frac))
        return d.iloc[:cut], d.iloc[cut:]
    d = df.sample(frac=1.0, random_state=seed)
    cut = int(len(d) * (1 - test_frac))
    return d.iloc[:cut], d.iloc[cut:]


@dataclass
class TrainResult:
    target: str
    features: list
    metrics: pd.DataFrame
    pipelines: dict
    predictions: dict            # model name -> np.ndarray on the test set
    test: pd.DataFrame
    train_size: int
    split_desc: str
    best_model: str = ""
    rel_err_q: dict = field(default_factory=dict)   # model -> (q10, q90) of actual/predicted


def train_and_evaluate(df: pd.DataFrame, target: str, features: list[str], models: list[str],
                       split_mode: str = "time", test_frac: float = 0.2) -> TrainResult:
    feats = [f for f in features if f != target]
    if target == "Units Sold":
        feats = [f for f in feats if f != "Units Sold"]
    train, test = split_data(df, split_mode, test_frac)
    rows, pipes, preds, qs = [], {}, {}, {}
    for m in models:
        pipe = make_pipeline(m, feats)
        pipe.fit(train[feats], train[target])
        p = pipe.predict(test[feats])
        if target in ("Sales", "Units Sold"):
            p = np.clip(p, 0, None)
        pipes[m], preds[m] = pipe, p
        rows.append({"Model": m, **regression_metrics(test[target], p)})
        ok = p > 0
        ratio = (test[target].values[ok] / p[ok]) if ok.any() else np.array([1.0])
        qs[m] = (float(np.quantile(ratio, 0.10)), float(np.quantile(ratio, 0.90)))
    met = pd.DataFrame(rows).sort_values("RMSE").reset_index(drop=True)
    if split_mode == "time":
        desc = (f"Time-based split: trained on {train['Date'].min():%Y-%m-%d} → {train['Date'].max():%Y-%m-%d}, "
                f"tested on {test['Date'].min():%Y-%m-%d} → {test['Date'].max():%Y-%m-%d}")
    else:
        desc = "Random split"
    return TrainResult(target, feats, met, pipes, preds, test, len(train), desc, met.iloc[0]["Model"], qs)


def feature_importance(pipe: Pipeline, test: pd.DataFrame, features: list[str], target: str,
                       n: int = 3000, repeats: int = 3) -> pd.DataFrame:
    s = test.sample(min(n, len(test)), random_state=0)
    r = permutation_importance(pipe, s[features], s[target], n_repeats=repeats, random_state=0,
                               scoring="r2", n_jobs=1)
    out = pd.DataFrame({"Feature": features, "Importance": r.importances_mean, "Std": r.importances_std})
    return out.sort_values("Importance", ascending=True).reset_index(drop=True)


def fit_full(model_name: str, df: pd.DataFrame, target: str, features: list[str]) -> Pipeline:
    pipe = make_pipeline(model_name, features)
    pipe.fit(df[features], df[target])
    return pipe


def lookup_tables(df: pd.DataFrame) -> dict:
    prod = df.drop_duplicates("ProductID")[["ProductID", "Product Name", "Category", "Brand", "Warranty"]]
    city = df.drop_duplicates("CityID")[["CityID", "Cities"]]
    rep_cols = ["SalesrepID", "SalesPerson Name"] + [c for c in ("Designation", "Age") if c in df.columns]
    rep = df.drop_duplicates("SalesrepID")[rep_cols]
    scen = df.drop_duplicates("ScenarioID")[["ScenarioID", "Scenario Name"]]
    return {"product": prod.sort_values("Product Name").reset_index(drop=True),
            "city": city.sort_values("Cities").reset_index(drop=True),
            "rep": rep.sort_values("SalesPerson Name").reset_index(drop=True),
            "scenario": scen.reset_index(drop=True)}


def build_input_row(date, product_id, city_id, rep_id, scenario_id, units, tables: dict, t0) -> pd.DataFrame:
    date = pd.Timestamp(date)
    p = tables["product"].set_index("ProductID").loc[product_id]
    return pd.DataFrame([{
        "ProductID": product_id, "CityID": city_id, "SalesrepID": rep_id, "ScenarioID": scenario_id,
        "Category": p["Category"], "Brand": p["Brand"], "Warranty": p["Warranty"],
        "Units Sold": units, "Year": date.year, "Month": date.month, "Quarter": date.quarter,
        "DayOfWeek": date.dayofweek, "t": (date - pd.Timestamp(t0)).days,
    }])


# --------------------------------------------------------------------------------------
# 3. TIME-SERIES FORECASTING (monthly)
# --------------------------------------------------------------------------------------
SERIES_METRICS = ["Sales", "Profit", "Units Sold", "Cost of sales"]
MIN_OBS = 30


def monthly_series(df: pd.DataFrame, metric: str, scenario: str = "Actual",
                   dim: str | None = None, value: str | None = None) -> pd.Series:
    full_max = df["Date"].max()
    d = df
    if scenario != "All":
        d = d[d["Scenario Name"] == scenario]
    if dim and value is not None:
        d = d[d[dim] == value]
    s = d.groupby("YearMonth")[metric].sum()
    if s.empty:
        return s
    idx = pd.date_range(s.index.min(), s.index.max(), freq="MS")
    s = s.reindex(idx, fill_value=0.0)
    # drop an incomplete last month (partial data would look like a sales collapse)
    if full_max.day < full_max.days_in_month and s.index[-1] == full_max.to_period("M").to_timestamp():
        s = s.iloc[:-1]
    s.index.name = "Date"
    return s.astype(float)


def _future_index(last: pd.Timestamp, h: int) -> pd.DatetimeIndex:
    return pd.date_range(last + pd.offsets.MonthBegin(1), periods=h, freq="MS")


def fc_seasonal_naive(y: pd.Series, h: int) -> np.ndarray:
    v = y.values
    return np.array([v[len(v) - 12 + (k % 12)] for k in range(h)])


def fc_trend_season(y: pd.Series, h: int) -> np.ndarray:
    """Linear trend + monthly seasonal dummies (fitted with OLS)."""
    def X(idx, t0):
        t = np.arange(len(idx)) + t0
        D = pd.get_dummies(pd.Series(idx.month, index=idx).astype(pd.CategoricalDtype(range(1, 13))),
                           drop_first=True).values.astype(float)
        return np.column_stack([t, D])
    fut = _future_index(y.index[-1], h)
    lr = LinearRegression().fit(X(y.index, 0), y.values)
    return lr.predict(X(fut, len(y)))


def fc_holt_winters(y: pd.Series, h: int, m: int = 12) -> np.ndarray:
    """Additive Holt-Winters (level+trend+seasonality), parameters fitted by SSE minimisation."""
    v = y.values.astype(float)
    n = len(v)

    def run(params, ret_state=False):
        a, b, g = params
        level = v[:m].mean()
        trend = (v[m:2 * m].mean() - v[:m].mean()) / m
        season = list(v[:m] - level)
        sse = 0.0
        for i in range(n):
            s = season[i % m]
            fc = level + trend + s
            e = v[i] - fc
            sse += e * e
            new_level = a * (v[i] - s) + (1 - a) * (level + trend)
            trend = b * (new_level - level) + (1 - b) * trend
            season[i % m] = g * (v[i] - new_level) + (1 - g) * s
            level = new_level
        return (level, trend, season) if ret_state else sse

    best = None
    for x0 in ([0.3, 0.05, 0.2], [0.1, 0.01, 0.1], [0.6, 0.1, 0.4]):
        r = minimize(run, x0, bounds=[(0.01, 0.99)] * 3, method="L-BFGS-B")
        if best is None or r.fun < best.fun:
            best = r
    level, trend, season = run(best.x, ret_state=True)
    return np.array([level + (k + 1) * trend + season[(n + k) % m] for k in range(h)])


LAGS = [1, 2, 3, 6, 12]


def _lag_features(vals: np.ndarray, idx: pd.DatetimeIndex, t_start: int) -> pd.DataFrame:
    s = pd.Series(vals)
    f = {f"lag{l}": s.shift(l) for l in LAGS}
    f["roll3"] = s.shift(1).rolling(3).mean()
    f["roll12"] = s.shift(1).rolling(12).mean()
    f["sin"] = np.sin(2 * np.pi * idx.month / 12)
    f["cos"] = np.cos(2 * np.pi * idx.month / 12)
    f["t"] = np.arange(len(vals)) + t_start
    return pd.DataFrame(f)


def fc_gbm_lags(y: pd.Series, h: int) -> np.ndarray:
    """Gradient boosting on lag features. Target = y / rolling-12 mean (removes level so trees need not extrapolate)."""
    v = y.values.astype(float)
    F = _lag_features(v, y.index, 0)
    base = F["roll12"].values
    ok = ~np.isnan(F.values).any(axis=1)
    # scale lag columns by the base level too
    Xs = F.copy()
    for c in [f"lag{l}" for l in LAGS] + ["roll3", "roll12"]:
        Xs[c] = Xs[c] / base
    target = v / base
    model = GradientBoostingRegressor(n_estimators=150, max_depth=2, learning_rate=0.05,
                                      subsample=0.8, random_state=42)
    model.fit(Xs[ok].drop(columns=["t"]), target[ok])
    hist = list(v)
    fut = _future_index(y.index[-1], h)
    out = []
    for k in range(h):
        idx = y.index.append(fut[: k + 1])
        Fk = _lag_features(np.array(hist + [np.nan]), idx, 0).iloc[[-1]].copy()
        b = Fk["roll12"].values[0]
        for c in [f"lag{l}" for l in LAGS] + ["roll3", "roll12"]:
            Fk[c] = Fk[c] / b
        pred = float(model.predict(Fk.drop(columns=["t"]))[0] * b)
        out.append(pred)
        hist.append(pred)
    return np.array(out)


FORECAST_MODELS = {
    "Seasonal Naive": fc_seasonal_naive,
    "Trend + Seasonality (OLS)": fc_trend_season,
    "Holt-Winters (additive)": fc_holt_winters,
    "Gradient Boosting (lags)": fc_gbm_lags,
}
ENSEMBLE = "Ensemble (average)"


@dataclass
class ForecastResult:
    history: pd.Series
    backtest_actual: pd.Series
    backtest_preds: dict            # model -> Series over test window
    scores: pd.DataFrame
    best_model: str
    future: pd.DataFrame            # index = future dates; columns = model names + 'Forecast','Lower','Upper'
    ci: float


def run_forecast(y: pd.Series, horizon: int, test_months: int = 12, models: list[str] | None = None,
                 selected: str | None = None, ci: float = 0.95) -> ForecastResult:
    models = models or list(FORECAST_MODELS)
    if len(y) < max(MIN_OBS, test_months + 24):
        raise ValueError(f"Need at least {max(MIN_OBS, test_months + 24)} monthly observations "
                         f"(have {len(y)}). Reduce the back-test window or widen the filters.")
    train, test = y.iloc[:-test_months], y.iloc[-test_months:]
    bt = {m: pd.Series(FORECAST_MODELS[m](train, test_months), index=test.index) for m in models}
    if len(models) > 1:
        bt[ENSEMBLE] = pd.concat(bt.values(), axis=1).mean(axis=1)
    rows = []
    for m, p in bt.items():
        mae = mean_absolute_error(test, p)
        rows.append({"Model": m, "MAE": mae, "RMSE": float(np.sqrt(mean_squared_error(test, p))),
                     "MAPE %": float(np.mean(np.abs((test - p) / test.replace(0, np.nan))) * 100),
                     "Bias": float((p - test).mean())})
    scores = pd.DataFrame(rows).sort_values("RMSE").reset_index(drop=True)
    best = selected if selected in bt else scores.iloc[0]["Model"]

    fut_idx = _future_index(y.index[-1], horizon)
    fut = {m: FORECAST_MODELS[m](y, horizon) for m in models}
    fut_df = pd.DataFrame(fut, index=fut_idx)
    if len(models) > 1:
        fut_df[ENSEMBLE] = fut_df[models].mean(axis=1)
    fut_df["Forecast"] = fut_df[best]
    # prediction interval: back-test RMSE of chosen model, widening slowly with horizon
    rmse = float(scores.set_index("Model").loc[best, "RMSE"])
    from scipy.stats import norm
    z = norm.ppf(0.5 + ci / 2)
    width = z * rmse * np.sqrt(1 + 0.05 * np.arange(horizon))
    fut_df["Lower"] = np.clip(fut_df["Forecast"] - width, 0, None)
    fut_df["Upper"] = fut_df["Forecast"] + width
    return ForecastResult(y, test, bt, scores, best, fut_df, ci)


def forecast_breakdown(df: pd.DataFrame, metric: str, scenario: str, dim: str, horizon: int) -> pd.DataFrame:
    """Forecast every member of a dimension (e.g. every Category) with the fast OLS trend+season model."""
    out = {}
    for val in sorted(df[dim].dropna().unique()):
        s = monthly_series(df, metric, scenario, dim, val)
        if len(s) >= MIN_OBS:
            out[val] = pd.Series(np.clip(fc_trend_season(s, horizon), 0, None), index=_future_index(s.index[-1], horizon))
    return pd.DataFrame(out)
