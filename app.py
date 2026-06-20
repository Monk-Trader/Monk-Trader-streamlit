import streamlit as st
from datetime import datetime

st.title("Monk Trader Quant Lab")

st.write("Welcome to my first financial research website.")

st.subheader("Current Time")

st.write(datetime.now())

st.success("Website is running successfully.")
