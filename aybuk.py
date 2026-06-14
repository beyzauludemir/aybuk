# ============================================================
# IMPORTS
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np

import plotly.express as px
import plotly.graph_objects as go

from scipy import stats
import math

# Forecast Models
from statsmodels.tsa.statespace.sarimax import SARIMAX
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Steel Inventory Intelligence",
    page_icon="📊",
    layout="wide"
)

# ============================================================
# TITLE
# ============================================================

st.title("📊 Steel Inventory Intelligence Platform")

st.markdown("""
Integrated Demand Forecasting and Inventory Optimization System
""")

# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("Configuration")

    abc_file = st.file_uploader(
        "Upload ABC XYZ Dataset",
        type=["xlsx"]
    )

    forecast_file = st.file_uploader(
        "Upload Forecast Dataset",
        type=["xlsx"]
    )

    inventory_file = st.file_uploader(
        "Upload Inventory Dataset",
        type=["xlsx"]
    )

    forecast_horizon = st.slider(
        "Forecast Horizon (Days)",
        min_value=30,
        max_value=365,
        value=90
    )

    service_level = st.selectbox(
        "Service Level",
        [0.90,0.95,0.97,0.99],
        index=1
    )

# ============================================================
# TABS
# ============================================================

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    [
        "ABC-XYZ Analysis",
        "Demand Forecast",
        "Inventory Optimization",
        "Visualization",
        "Executive Summary"
    ]
)

# ============================================================
# ABC ANALYSIS
# ============================================================

def abc_analysis(df):

    temp = df.copy()

    temp["Annual_Value"] = temp["Consumption"] * temp["Unit_Price"]

    temp = temp.groupby(
        ["Material_Code","Material_Name"],
        as_index=False
    )["Annual_Value"].sum()

    temp = temp.sort_values(
        "Annual_Value",
        ascending=False
    )

    total = temp["Annual_Value"].sum()

    temp["Cum_%"] = (
        temp["Annual_Value"].cumsum()
        / total
        * 100
    )

    conditions = [
        temp["Cum_%"] <= 80,
        temp["Cum_%"] <= 95,
        temp["Cum_%"] > 95
    ]

    choices = ["A","B","C"]

    temp["ABC"] = np.select(
        conditions,
        choices,
        default="C"
    )

    return temp

# ============================================================
# XYZ ANALYSIS
# ============================================================

def xyz_analysis(df):

    xyz = (
        df.groupby(
            ["Material_Code","Material_Name"]
        )["Consumption"]
        .agg(["mean","std"])
        .reset_index()
    )

    xyz["CV"] = (
        xyz["std"]
        /
        xyz["mean"]
    ) * 100

    xyz["XYZ"] = np.where(
        xyz["CV"] < 10,
        "X",
        np.where(
            xyz["CV"] < 25,
            "Y",
            "Z"
        )
    )

    return xyz

# ============================================================
# ABC XYZ COMBINATION
# ============================================================

def build_abc_xyz(df):

    abc = abc_analysis(df)

    xyz = xyz_analysis(df)

    final = abc.merge(
        xyz[
            [
                "Material_Code",
                "XYZ",
                "CV"
            ]
        ],
        on="Material_Code"
    )

    final["Class"] = (
        final["ABC"]
        +
        final["XYZ"]
    )

    return final

# ============================================================
# LOAD ABC FILE
# ============================================================

if abc_file is not None:

    raw = pd.read_excel(
        abc_file,
        sheet_name="RawDataCombined"
    )

    abc_df = pd.DataFrame()

    abc_df["Material_Code"] = raw["Malzeme"]

    abc_df["Material_Name"] = raw["Malzeme kısa metni"]

    abc_df["Date"] = pd.to_datetime(
        raw["Kayıt tarihi"]
    )

    abc_df["Consumption"] = (
        raw["Miktar Abs"]
    )

    abc_df["Unit_Price"] = 1

    abc_result = build_abc_xyz(
        abc_df
    )

# ============================================================
# MATERIAL SELECTION
# ============================================================

if abc_file is not None:

    material_list = sorted(
        abc_result["Material_Name"]
        .unique()
        .tolist()
    )

    selected_material = st.sidebar.selectbox(
        "Select Material",
        material_list
    )

with tab1:

    st.subheader("ABC XYZ Classification")

    if abc_file is not None:

        st.dataframe(
            abc_result,
            use_container_width=True
        )

        heatmap = (
            abc_result
            .groupby(
                ["ABC","XYZ"]
            )
            .size()
            .reset_index(name="Count")
        )

        fig = px.density_heatmap(
            heatmap,
            x="ABC",
            y="XYZ",
            z="Count",
            text_auto=True
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

# ============================================================
# FORECAST CONFIG
# ============================================================

DATE_COL = "Tarih"
VALUE_COL = "Tüketim"

SEASONAL = 7

DEFAULT_ORDER = (1,1,2)

DEFAULT_SORDER = (
    1,
    1,
    1,
    SEASONAL
)

N_LAGS = 14

ROLL_WINDOWS = (
    7,
    14,
    30
)

ALPHA = 0.05
# ============================================================
# LOAD FORECAST FILE
# ============================================================

def load_forecast_data(file):

    df = pd.read_excel(file)

    df[DATE_COL] = pd.to_datetime(
        df[DATE_COL]
    )

    daily = (
        df
        .set_index(DATE_COL)[VALUE_COL]
        .sort_index()
        .asfreq("D")
        .fillna(0)
    )

    out = pd.DataFrame(
        {
            "date": daily.index,
            VALUE_COL: daily.values
        }
    )

    out["t"] = np.arange(
        1,
        len(out)+1
    )

    return out

def make_feature_row(
        history_values,
        idx_date,
        t_value
):

    h = np.asarray(
        history_values,
        dtype=float
    )

    feat = {}

    for lag in range(
        1,
        N_LAGS+1
    ):

        feat[f"lag_{lag}"] = (
            h[-lag]
            if len(h)>=lag
            else 0
        )

    for w in ROLL_WINDOWS:

        window = h[-w:]

        feat[f"rmean_{w}"] = (
            np.mean(window)
        )

        feat[f"rstd_{w}"] = (
            np.std(window)
        )

    dow = idx_date.dayofweek

    feat["dow"] = dow

    feat["month"] = idx_date.month

    feat["trend"] = t_value

    return feat

def build_training_matrix(
        train_frame
):

    values = train_frame[
        VALUE_COL
    ].values.astype(float)

    dates = train_frame[
        "date"
    ].values

    ts = train_frame[
        "t"
    ].values

    rows = []
    targets = []

    for i in range(
        N_LAGS,
        len(values)
    ):

        hist = values[:i]

        feat = make_feature_row(
            hist,
            pd.Timestamp(
                dates[i]
            ),
            int(ts[i])
        )

        rows.append(feat)

        targets.append(
            values[i]
        )

    X = pd.DataFrame(rows)

    y = np.asarray(
        targets,
        dtype=float
    )

    return X,y,list(X.columns)

def fit_xgboost(X,y):

    model = XGBRegressor(

        n_estimators=600,

        learning_rate=0.03,

        max_depth=4,

        subsample=0.8,

        colsample_bytree=0.8,

        random_state=42,

        verbosity=0

    )

    model.fit(X,y)

    return model
def fit_catboost(X,y):

    model = CatBoostRegressor(

        iterations=600,

        learning_rate=0.03,

        depth=4,

        verbose=0,

        random_seed=42

    )

    model.fit(X,y)

    return model

def fit_sarima(train):

    model = SARIMAX(

        train,

        order=DEFAULT_ORDER,

        seasonal_order=DEFAULT_SORDER,

        enforce_stationarity=False,

        enforce_invertibility=False

    )

    return model.fit(
        disp=False
    )

def error_stats(
        actual,
        pred
):

    actual = np.asarray(actual)

    pred = np.asarray(pred)

    err = actual-pred

    mae = np.mean(
        np.abs(err)
    )

    rmse = np.sqrt(
        np.mean(err**2)
    )

    mask = actual != 0

    mape = np.mean(
        np.abs(
            err[mask]
            /
            actual[mask]
        )
    ) * 100

    return mae,rmse,mape
# ============================================================
# TRAIN TEST SPLIT
# ============================================================

def split_train_test(
        frame,
        test_ratio=0.20
):

    split_idx = int(
        len(frame)
        *
        (1-test_ratio)
    )

    train = frame.iloc[
        :split_idx
    ].reset_index(drop=True)

    test = frame.iloc[
        split_idx:
    ].reset_index(drop=True)

    return train,test

# ============================================================
# SARIMA
# ============================================================

def sarima_forecast(
        train,
        test
):

    model = SARIMAX(

        train[VALUE_COL],

        order=DEFAULT_ORDER,

        seasonal_order=DEFAULT_SORDER,

        enforce_stationarity=False,

        enforce_invertibility=False

    )

    result = model.fit(
        disp=False
    )

    pred = result.forecast(
        len(test)
    )

    return pred

# ============================================================
# XGBOOST
# ============================================================

def xgb_forecast(
        train,
        test
):

    X_train,y_train,cols = (
        build_training_matrix(
            train
        )
    )

    model = fit_xgboost(
        X_train,
        y_train
    )

    history = list(
        train[
            VALUE_COL
        ].values
    )

    preds = []

    for _,row in test.iterrows():

        feat = make_feature_row(

            history,

            pd.Timestamp(
                row["date"]
            ),

            row["t"]

        )

        x = pd.DataFrame(
            [feat]
        )[cols]

        yhat = float(
            model.predict(x)[0]
        )

        preds.append(yhat)

        history.append(
            row[VALUE_COL]
        )

    return np.array(preds)

# ============================================================
# CATBOOST
# ============================================================

def cat_forecast(
        train,
        test
):

    X_train,y_train,cols = (
        build_training_matrix(
            train
        )
    )

    model = fit_catboost(
        X_train,
        y_train
    )

    history = list(
        train[
            VALUE_COL
        ].values
    )

    preds = []

    for _,row in test.iterrows():

        feat = make_feature_row(

            history,

            pd.Timestamp(
                row["date"]
            ),

            row["t"]

        )

        x = pd.DataFrame(
            [feat]
        )[cols]

        yhat = float(
            model.predict(x)[0]
        )

        preds.append(yhat)

        history.append(
            row[VALUE_COL]
        )

    return np.array(preds)

# ============================================================
# BEST MODEL
# ============================================================

def select_best_model(
        metrics
):

    rmse_values = {

        k:v["RMSE"]

        for k,v

        in metrics.items()

    }

    best_model = min(

        rmse_values,

        key=rmse_values.get

    )

    return best_model




    

with tab2:

    st.subheader(
        "Demand Forecast"
    )

    if forecast_file is not None:

        frame = load_forecast_data(
            forecast_file
        )

        train,test = (
            split_train_test(
                frame
            )
        )

        sarima_pred = (
            sarima_forecast(
                train,
                test
            )
        )

        xgb_pred = (
            xgb_forecast(
                train,
                test
            )
        )

        cat_pred = (
            cat_forecast(
                train,
                test
            )
        )

        metrics = {}

        for name,pred in {

            "SARIMA":sarima_pred,

            "XGBoost":xgb_pred,

            "CatBoost":cat_pred

        }.items():

            mae,rmse,mape = error_stats(

                test[VALUE_COL],

                pred

            )

            metrics[name] = {

                "MAE":mae,

                "RMSE":rmse,

                "MAPE":mape

            }

        metrics_df = pd.DataFrame(
            metrics
        ).T

        st.dataframe(
            metrics_df,
            use_container_width=True
        )

        best_model = select_best_model(
            metrics
        )

        st.success(
            f"Best Model: {best_model}"
        )

        st.session_state[
            "best_model"
        ] = best_model
# ============================================================
# BUILD FORECAST DEMAND SERIES
# ============================================================

def build_best_model_series(
        test,
        best_model,
        sarima_pred,
        xgb_pred,
        cat_pred
):

    if best_model == "SARIMA":

        demand = sarima_pred

    elif best_model == "XGBoost":

        demand = xgb_pred

    else:

        demand = cat_pred

    forecast_series = pd.DataFrame({

        "Date":
        test["date"].values,

        "Forecast":
        demand,

        "Actual":
        test[VALUE_COL].values

    })

    return forecast_series

forecast_series = build_best_model_series(

    test,

    best_model,

    sarima_pred,

    xgb_pred,

    cat_pred

)

st.session_state[
    "forecast_series"
] = forecast_series
comparison = pd.DataFrame({

    "Date":
    test["date"],

    "Actual":
    test[VALUE_COL],

    "SARIMA":
    sarima_pred,

    "XGBoost":
    xgb_pred,

    "CatBoost":
    cat_pred

})

fig = go.Figure()

fig.add_trace(

    go.Scatter(

        x=comparison["Date"],

        y=comparison["Actual"],

        name="Actual"

    )

)

fig.add_trace(

    go.Scatter(

        x=comparison["Date"],

        y=comparison["SARIMA"],

        name="SARIMA"

    )

)

fig.add_trace(

    go.Scatter(

        x=comparison["Date"],

        y=comparison["XGBoost"],

        name="XGBoost"

    )

)

fig.add_trace(

    go.Scatter(

        x=comparison["Date"],

        y=comparison["CatBoost"],

        name="CatBoost"

    )

)

st.plotly_chart(
    fig,
    use_container_width=True
)
# ============================================================
# INVENTORY CONFIG
# ============================================================

ORDERING_COST = 2792

HOLDING_COST_PCT = 0.25

DEFAULT_LEAD_TIME = 3

DEFAULT_UNIT_PRICE = 45

STOCKOUT_COST_PER_UNIT = 500

# ============================================================
# EOQ
# ============================================================

def calculate_eoq(
        annual_demand,
        ordering_cost,
        unit_price
):

    holding_cost = (
        HOLDING_COST_PCT
        *
        unit_price
    )

    if annual_demand <= 0:

        return 1

    eoq = math.sqrt(

        (
            2
            *
            annual_demand
            *
            ordering_cost
        )
        /
        holding_cost

    )

    return round(
        eoq,
        2
    )

# ============================================================
# SAFETY STOCK
# ============================================================

def calculate_safety_stock(

        std_demand,

        lead_time,

        service_level

):

    z = stats.norm.ppf(
        service_level
    )

    safety_stock = (

        z

        *

        std_demand

        *

        np.sqrt(
            lead_time
        )

    )

    return round(
        safety_stock,
        2
    )

# ============================================================
# ROP
# ============================================================

def calculate_rop(

        avg_daily_demand,

        lead_time,

        safety_stock

):

    rop = (

        avg_daily_demand

        *

        lead_time

        +

        safety_stock

    )

    return round(
        rop,
        2
    )
# ============================================================
# COST MODEL
# ============================================================

def calculate_inventory_cost(

        annual_demand,

        eoq,

        unit_price

):

    annual_ordering_cost = (

        annual_demand
        /
        eoq

    ) * ORDERING_COST

    annual_holding_cost = (

        eoq
        /
        2

    ) * (

        HOLDING_COST_PCT
        *
        unit_price

    )

    total_cost = (

        annual_ordering_cost

        +

        annual_holding_cost

    )

    return {

        "ordering_cost":
        annual_ordering_cost,

        "holding_cost":
        annual_holding_cost,

        "total_cost":
        total_cost

    }
forecast_series = st.session_state["forecast_series"]

with tab3:

    st.subheader(
        "Inventory Optimization"
    )

    if "forecast_series" in st.session_state:

        forecast_series = st.session_state[
            "forecast_series"
        ]

        avg_daily_demand = (
            forecast_series["Forecast"]
            .mean()
        )

        annual_demand = (
            avg_daily_demand
            * 365
        )

        std_demand = (
            forecast_series["Forecast"]
            .std()
        )

        lead_time = DEFAULT_LEAD_TIME

        unit_price = DEFAULT_UNIT_PRICE

        eoq = calculate_eoq(

            annual_demand,

            ORDERING_COST,

            unit_price

        )

        safety_stock = calculate_safety_stock(

            std_demand,

            lead_time,

            service_level

        )

        rop = calculate_rop(

            avg_daily_demand,

            lead_time,

            safety_stock

        )

        costs = calculate_inventory_cost(

            annual_demand,

            eoq,

            unit_price

        )

        c1,c2,c3,c4 = st.columns(4)

        c1.metric(
            "EOQ",
            f"{eoq:,.0f}"
        )

        c2.metric(
            "ROP",
            f"{rop:,.0f}"
        )

        c3.metric(
            "Safety Stock",
            f"{safety_stock:,.0f}"
        )

        c4.metric(
            "Annual Demand",
            f"{annual_demand:,.0f}"
        )

        st.divider()

        c5,c6,c7 = st.columns(3)

        c5.metric(
            "Ordering Cost",
            f"₺{costs['ordering_cost']:,.0f}"
        )

        c6.metric(
            "Holding Cost",
            f"₺{costs['holding_cost']:,.0f}"
        )

        c7.metric(
            "Total Cost",
            f"₺{costs['total_cost']:,.0f}"
        )

        inventory_summary = {

            "EOQ":eoq,

            "ROP":rop,

            "Safety_Stock":
            safety_stock,

            "Annual_Demand":
            annual_demand,

            "Total_Cost":
            costs["total_cost"]

        }

        st.session_state[
            "inventory_summary"
        ] = inventory_summary

with tab4:

    st.subheader(
        "Forecast vs Inventory"
    )

    if "forecast_series" in st.session_state:

       forecast_series = st.session_state["forecast_series"]

       fig = px.line(forecast_series,x="Date",y=["Actual","Forecast"])
best_model = st.session_state["best_model"]
Forecast Winner:
{best_model}

        st.plotly_chart(
            fig,
            use_container_width=True
        )

with tab5:

    st.subheader(
        "Executive Summary"
    )

    if (

        "inventory_summary"
        in st.session_state

    ):

        inv = st.session_state[
            "inventory_summary"
        ]

        st.markdown(
f"""

### Diesel (Motorin)

**ABC XYZ Class**
AX

**Forecast Horizon**
{forecast_horizon} Days

**Annual Demand**
{inv["Annual_Demand"]:,.0f}

**EOQ**
{inv["EOQ"]:,.0f}

**ROP**
{inv["ROP"]:,.0f}

**Safety Stock**
{inv["Safety_Stock"]:,.0f}

**Expected Annual Cost**
₺{inv["Total_Cost"]:,.0f}

**Recommended Policy**
Continuous Review

"""
        )



   
         
