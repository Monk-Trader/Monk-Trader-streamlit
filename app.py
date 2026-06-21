import streamlit.components.v1 as components
import requests
from datetime import datetime
import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from ta.trend import ADXIndicator

st.set_page_config(
    page_title="Monk Trader Quant Lab",
    layout="wide"
)

st.title(
    "🚀 Monk Trader Quant Lab"
)

st.caption(
    "Institutional Asset Allocation & Risk Regime Dashboard"
)

# =====================================
# VISITOR COUNTER
# =====================================

try:
    counter_url = "https://api.counterapi.dev/v1/monktrader/nifty_dashboard/up"

    response = requests.get(counter_url, timeout=5)

    visitor_count = response.json()["count"]

except:
    visitor_count = "N/A"


@st.cache_data
def load_strong_stocks():
    df = pd.read_csv("Strong_Stocks.csv")

    # Remove accidental spaces
    df.columns = df.columns.str.strip()

    return df
    
def get_data():
    df = yf.download("^NSEI", start="2013-01-01", progress=False)

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    return df


# =====================================
# DATA DOWNLOAD
# =====================================

with st.spinner("Downloading NIFTY data..."):
    df = get_data()

# =====================================
# FEATURE ENGINEERING
# =====================================

df['Log_Return'] = np.log(df['Close'] / df['Close'].shift(1))
df['Vol'] = df['Log_Return'].rolling(window=20).std()

df.dropna(inplace=True)

df['Feat_Return'] = df['Log_Return'].ewm(span=20).mean()
df['Feat_Vol'] = df['Vol'].ewm(span=20).mean()


adx = ADXIndicator(
    high=df['High'],
    low=df['Low'],
    close=df['Close'],
    window=14
)

df['ADX'] = adx.adx()

# =====================================
# REGIME ENGINE
# =====================================

def determine_regime(row):

    ret = row['Feat_Return']
    vol = row['Feat_Vol']

    if ret < -0.0002 and vol > 0.0085:
        return 0  # Bear

    elif ret > 0.0004 and vol < 0.0090:
        return 2  # Bull

    else:
        return 1  # Choppy


df['Raw_Regime'] = df.apply(determine_regime, axis=1)

df['Regime_Smoothed'] = (
    df['Raw_Regime']
    .ewm(span=7)
    .mean()
    .round()
    .astype(int)
)

df['Regime'] = np.clip(df['Regime_Smoothed'], 0, 2)

state_map = {
    0: "Bear",
    1: "Choppy",
    2: "Bull"
}

color_map = {
    0: '#d9534f',
    1: '#f0ad4e',
    2: '#5cb85c'
}

weight_map = {
    0: 0.20,
    1: 0.60,
    2: 1.00
}

df['Target_Equity_Weight'] = df['Regime'].map(weight_map)

# =====================================
# DIAGNOSTIC METRICS
# =====================================

bear_days = df[df['Regime'] == 0]

total_market_days = len(df)

bear_percentage = (
    len(bear_days) /
    total_market_days
) * 100

bear_return = np.exp(
    bear_days['Log_Return'].sum()
) - 1

current_regime = state_map[df['Regime'].iloc[-1]]

# =====================================
# LAST REGIME CHANGE
# =====================================

current_regime_code = df['Regime'].iloc[-1]

regime_start_date = df.index[-1]

for i in range(len(df)-2, -1, -1):

    if df['Regime'].iloc[i] != current_regime_code:
        regime_start_date = df.index[i+1]
        break

days_in_regime = (
    df.index[-1] - regime_start_date
).days

current_alloc = (
    df['Target_Equity_Weight'].iloc[-1] * 100
)
current_adx = df['ADX'].iloc[-1]

# =====================================
# SIDEBAR DASHBOARD
# =====================================

st.sidebar.title("📊 Dashboard Status")

# Current Regime
st.sidebar.metric(
    "Current Regime",
    current_regime
)

st.sidebar.write(
    f"Since: {regime_start_date.strftime('%Y-%m-%d')}"
)

st.sidebar.write(
    f"Days: {days_in_regime}"
)


# Allocation
st.sidebar.metric(
    "Equity Allocation",
    f"{current_alloc:.0f}%"
)

current_adx = df['ADX'].iloc[-1]

if current_adx < 20:
    trend_strength = "Weak"

elif current_adx < 25:
    trend_strength = "Moderate"

else:
    trend_strength = "Strong"

st.sidebar.metric(
    "Current ADX",
    f"{current_adx:.1f}"
)

st.sidebar.write(
    f"Trend Strength: {trend_strength}"
)


# Visitor Count
st.sidebar.metric(
    "Visitors",
    visitor_count
)

# Last Updated
st.sidebar.markdown("---")

st.sidebar.subheader("⏰ Last Updated")

st.sidebar.write(
    datetime.now().strftime(
        "%d-%b-%Y %H:%M:%S"
    )
)

st.sidebar.markdown("---")

st.sidebar.success(
    "Engine Operational"
)

st.subheader("📊 Bear State Capture Diagnostic Audit")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Market Days",
        f"{total_market_days:,}"
    )

with col2:
    st.metric(
        "Bear Days",
        f"{len(bear_days):,}"
    )

with col3:
    st.metric(
        "Bear %",
        f"{bear_percentage:.2f}%"
    )

with col4:
    st.metric(
        "Return During Bear",
        f"{bear_return*100:.2f}%"
    )

# =====================================
# CURRENT STATUS
# =====================================

st.subheader("🛡 Current Regime Status")

col1, col2 = st.columns(2)

with col1:
    st.metric(
        "Current Regime",
        current_regime
    )

with col2:
    st.metric(
        "Target Equity Allocation",
        f"{current_alloc:.0f}%"
    )

# =====================================
# LAST 15 DAYS TABLE
# =====================================

st.subheader("📅 Last 15 Days Regime Snapshot")

last_15 = df.tail(15).copy()

last_15['State_Name'] = (
    last_15['Regime']
    .map(state_map)
)

last_15['Allocation_%'] = (
    last_15['Target_Equity_Weight'] * 100
)

display_table = last_15[
    [
        'Close',
        'State_Name',
        'Allocation_%'
    ]
].copy()

display_table = display_table.round(2)

st.dataframe(
    display_table,
    use_container_width=True
)

# =====================================
# CHARTS
# =====================================
st.subheader("📈NIFTY Risk Regime")

fig = make_subplots(
    rows=2,
    cols=1,
    shared_xaxes=True,
    vertical_spacing=0.08,
    row_heights=[0.7, 0.3],
    subplot_titles=(
        "NIFTY Risk Regimes",
        "Institutional Asset Allocation"
    )
)

# =====================================
# REGIME SCATTERS
# =====================================

for state_code, state_name in state_map.items():

    mask = df['Regime'] == state_code

    fig.add_trace(

        go.Scatter(

            x=df.index[mask],

            y=df['Close'][mask],

            mode='markers',

            name=state_name,

            marker=dict(
                size=5,
                color=color_map[state_code]
            ),

            hovertemplate=
            "<b>Date:</b> %{x}<br>" +
            "<b>NIFTY:</b> %{y:,.0f}<br>" +
            "<b>State:</b> " + state_name +
            "<extra></extra>"

        ),

        row=1,
        col=1

    )

# =====================================
# ALLOCATION LINE
# =====================================

fig.add_trace(

    go.Scatter(

        x=df.index,

        y=df['Target_Equity_Weight'] * 100,

        mode='lines',

        name='Equity Allocation',

        hovertemplate=
        "<b>Date:</b> %{x}<br>" +
        "<b>Allocation:</b> %{y:.0f}%<extra></extra>"

    ),

    row=2,
    col=1

)

# =====================================
# LAYOUT
# =====================================

fig.update_layout(

    height=850,

    title="Monk Trader Asset Allocation Engine",

    hovermode='x unified',

    template='plotly_dark',

    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="left",
        x=0
    )

)

fig.update_yaxes(
    type="log",
    title_text="NIFTY",
    row=1,
    col=1
)

fig.update_yaxes(
    title_text="Allocation %",
    row=2,
    col=1
)

st.plotly_chart(
    fig,
    use_container_width=True
)

st.markdown("---")

st.header("🏆 Top Strong Stocks")
# =====================================
# LOAD STRONG STOCKS
# =====================================

strong_stocks = load_strong_stocks()

# =====================================
# TRADINGVIEW LINK SECTION
# =====================================
# Use first column as Company Name
company_col = strong_stocks.columns[0]

# Use Symbol column
symbol_col = "Symbol"

strong_stocks["Display"] = (
    strong_stocks[company_col].astype(str)
    + " ("
    + strong_stocks[symbol_col].astype(str)
    + ")"
)

selected = st.selectbox(
    "Select Stock",
    strong_stocks["Display"]
)

symbol = selected.split("(")[1].replace(")", "")

tv_url = (
    f"https://www.tradingview.com/symbols/NSE-{symbol}/"
)

st.link_button(
    "📊 Open Chart in TradingView",
    tv_url
)

# =====================================
# STRONG STOCK TABLE
# =====================================

st.dataframe(
    strong_stocks,
    use_container_width=True,
    hide_index=True
)

st.dataframe(
    strong_stocks,
    use_container_width=True,
    hide_index=True
)

# =====================================
# FOOTER
# =====================================

st.success(
    "Engine Operational Successfully"
)
