import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

st.set_page_config(
    page_title="Monk Trader Quant Lab",
    layout="wide"
)

st.title("🚀 NIFTY 50 Production Asset Allocation Engine")


@st.cache_data(ttl=3600)
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

current_alloc = (
    df['Target_Equity_Weight'].iloc[-1] * 100
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

st.subheader("📈 NIFTY Risk Regime Dashboard")

fig, (ax1, ax2) = plt.subplots(
    2,
    1,
    figsize=(14, 8),
    sharex=True,
    gridspec_kw={
        'height_ratios': [2, 1]
    }
)

ax1.set_yscale('log')

ax1.yaxis.set_major_formatter(
    plt.ScalarFormatter()
)

ax1.yaxis.set_minor_formatter(
    plt.NullFormatter()
)

for state_code, state_name in state_map.items():

    mask = df['Regime'] == state_code

    ax1.scatter(
        df.index[mask],
        df['Close'][mask],
        color=color_map[state_code],
        s=5,
        label=state_name,
        alpha=0.8
    )

ax1.set_title(
    "NIFTY 50 Risk Regimes"
)

ax1.set_ylabel(
    "Index Value (Log Scale)"
)

ax1.grid(
    True,
    linestyle="--",
    alpha=0.3
)

ax1.legend()

# --------------------

ax2.plot(
    df.index,
    df['Target_Equity_Weight'] * 100,
    linewidth=1.5
)

ax2.fill_between(
    df.index,
    df['Target_Equity_Weight'] * 100,
    0,
    alpha=0.2
)

ax2.set_title(
    "Institutional Asset Allocation"
)

ax2.set_ylabel(
    "Allocation %"
)

ax2.set_ylim(
    -5,
    105
)

ax2.grid(
    True,
    linestyle="--",
    alpha=0.4
)

ax2.xaxis.set_major_locator(
    mdates.YearLocator()
)

ax2.xaxis.set_major_formatter(
    mdates.DateFormatter('%Y')
)

plt.tight_layout()

st.pyplot(fig)

# =====================================
# FOOTER
# =====================================

st.success(
    "Engine Operational Successfully"
)
