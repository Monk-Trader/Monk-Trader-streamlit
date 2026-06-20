import streamlit as st
import random

st.title("Monk Trader Quant Lab")

atr_percentile = random.randint(1, 100)

st.metric("ATR Percentile", atr_percentile)

if atr_percentile > 70:
    st.error("Risk-Off Regime")
else:
    st.success("Risk-On Regime")
