import streamlit as st
import pandas as pd
import sqlite3
import plotly.express as px
import plotly.graph_objects as go
import time
import os
import logging

# Suppress Streamlit ScriptRunContext warnings
logging.getLogger("streamlit.runtime.scriptrunner").setLevel(logging.ERROR)
logging.getLogger("streamlit.runtime.state.session_state_proxy").setLevel(logging.ERROR)

# Check if running within Streamlit
try:
    from streamlit.runtime.scriptrunner import get_script_run_ctx
    if get_script_run_ctx() is None:
        print(" ERROR: Please run this script using 'streamlit run advanced_dashboard.py'")
        print("Use: .\\library\\Scripts\\streamlit.exe run advanced_dashboard.py")
        os._exit(1)
except ImportError:
    pass

# ==============================================================================
# CONFIGURATION
# ==============================================================================
st.set_page_config(page_title="Smart Checkout Analytics", page_icon="📊", layout="wide")
DB_FILE = os.path.abspath("checkout_data.db")
REFRESH_RATE = 2

# ==============================================================================
# DATA LOADING
# ==============================================================================
def get_data():
    if not os.path.exists(DB_FILE):
        return pd.DataFrame()
        
    try:
        # Connect with timeout to handle potential locks from the checkout app
        conn = sqlite3.connect(DB_FILE, timeout=10) 
        df = pd.read_sql("SELECT * FROM transactions", conn)
        conn.close()
        
        if not df.empty and 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        return df
    except Exception as e:
        # Avoid showing error in UI during transient DB locks
        return pd.DataFrame()

# ==============================================================================
# DASHBOARD
# ==============================================================================
st.title("🛒 Smart Checkout System - Performance Monitor")

df = get_data()

if df.empty:
    st.info("Waiting for data... Please scan items in the checkout app.")
    time.sleep(REFRESH_RATE)
    st.rerun()

# --- SIDEBAR FILTERS ---
st.sidebar.header("Filters")
min_conf = st.sidebar.slider("Confidence Threshold for 'Valid' Scans", 0.0, 1.0, 0.5)
filtered_df = df[df['confidence'] >= min_conf]
low_conf_df = df[df['confidence'] < min_conf]

# --- TOP METRICS ---
col1, col2, col3, col4 = st.columns(4)
total_sales = filtered_df['price'].sum()
item_count = len(filtered_df)
avg_conf = filtered_df['confidence'].mean() * 100 if not filtered_df.empty else 0
error_count = len(low_conf_df)

col1.metric("💰 Total Revenue", f"RM {total_sales:,.2f}")
col2.metric("📦 Items Sold", item_count)
col3.metric("🎯 Avg. Accuracy (Confidence)", f"{avg_conf:.1f}%")
col4.metric("⚠️ Low Confidence Scans", error_count, delta_color="inverse")

# --- ROW 1: CHARTS ---
c1, c2 = st.columns(2)

with c1:
    st.subheader("📈 Sales Trend Over Time")
    if not filtered_df.empty:
        # Group by minute for trend
        trend_df = filtered_df.set_index('timestamp').resample('1min')['price'].sum().reset_index()
        fig_line = px.line(trend_df, x='timestamp', y='price', markers=True, 
                           labels={'price': 'Revenue (RM)', 'timestamp': 'Time'})
        st.plotly_chart(fig_line, use_container_width=True)
    else:
        st.write("No data for trend.")

with c2:
    st.subheader("🥧 Sales by Product Category")
    if not filtered_df.empty:
        fig_pie = px.pie(filtered_df, names='item_name', values='price', hole=0.4)
        st.plotly_chart(fig_pie, use_container_width=True)
    else:
        st.write("No data for pie chart.")

# --- ROW 2: ACCURACY & ERRORS ---
c3, c4 = st.columns(2)

with c3:
    st.subheader("🎯 System Accuracy Gauge")
    fig_gauge = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = avg_conf,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Average Confidence Score"},
        gauge = {'axis': {'range': [None, 100]},
                 'bar': {'color': "#00CC96"},
                 'steps': [
                     {'range': [0, 50], 'color': "#FF4136"},
                     {'range': [50, 75], 'color': "#FFDC00"},
                     {'range': [75, 100], 'color': "#2ECC40"}]}))
    st.plotly_chart(fig_gauge, use_container_width=True)

with c4:
    st.subheader("⚠️ Potential False Positives (Low Confidence)")
    st.caption(f"Items detected below {min_conf*100:.0f}% confidence")
    if not low_conf_df.empty:
        st.dataframe(low_conf_df[['timestamp', 'item_name', 'confidence', 'price']], 
                     use_container_width=True)
    else:
        st.success("✅ No low-confidence detections found.")

# --- RAW DATA ---
with st.expander("📄 View All Transactions"):
    st.dataframe(df.sort_values(by='timestamp', ascending=False), use_container_width=True)

# Auto-refresh
time.sleep(REFRESH_RATE)
st.rerun()
