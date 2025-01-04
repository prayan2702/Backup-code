import streamlit as st
import pandas as pd
import datetime
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(layout="wide")  # Set full-width layout

# Replace with your actual Google Sheets CSV URL
google_sheets_url = "https://docs.google.com/spreadsheets/d/e/2PACX-1vTuyGRVZuafIk2s7moScIn5PAUcPYEyYIOOYJj54RXYUeugWmOP0iIToljSEMhHrg_Zp8Vab6YvBJDV/pub?output=csv"


@st.cache_data(ttl=0)  # Caching har baar bypass hoga
def load_data(url):
    data = pd.read_csv(url, header=0)
    data.columns = data.columns.str.strip().str.lower()  # Normalize column names

    date_col_candidates = [col for col in data.columns if 'date' in col.lower()]
    if date_col_candidates:
        data['date'] = pd.to_datetime(data[date_col_candidates[0]], errors='coerce')

    numeric_cols = ['nav', 'day change', 'day change %', 'nifty50 value', 'current value', 'nifty50 change %',
                    'dd', 'dd_n50', 'portfolio value', 'absolute gain', 'nifty50']
    for col in numeric_cols:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col].astype(str).str.replace(',', '').str.replace('%', ''), errors='coerce')

    if 'dd' not in data.columns and 'nav' in data.columns:
        data['dd'] = data['nav'] - data['nav'].cummax()

    data.fillna(0, inplace=True)
    return data


# Load data
data = load_data(google_sheets_url)

portfolio_value_raw = data.iloc[0, 0]  # Portfolio value from cell [0,0]
nifty50_value_raw = data.iloc[0, 2]  # Nifty50 value from cell [0,2]
day_change_raw = data.iloc[2, 0]  # Day Change from cell [0,3]

portfolio_value = pd.to_numeric(portfolio_value_raw, errors='coerce')
nifty50_value = pd.to_numeric(nifty50_value_raw, errors='coerce')
day_change = pd.to_numeric(day_change_raw, errors='coerce')

# Total Account Overview Section
st.write("## Total Account Overview", unsafe_allow_html=True)
col1, col2, col3, col4 = st.columns([3, 2, 2, 3])

with col1:
    st.metric("Total Account Value   ", f"₹{portfolio_value:,.0f}")
with col2:
    st.metric("Day Change", f"₹{day_change:,.0f}", f"{data['day change %'].iloc[-1]:,.2f}%")
with col3:
    st.metric("NIFTY50 Benchmark", f"{nifty50_value:,.0f}")
with col4:
    if len(data) > 30:
        month_change = data['current value'].iloc[-1] - data['current value'].iloc[-30]
        month_change_percent = (month_change / data['current value'].iloc[-30] * 100) if \
            data['current value'].iloc[-30] != 0 else 0
        st.metric("Month Change", f"₹{month_change:,.0f}", f"{month_change_percent:.2f}%")
    else:
        st.metric("Month Change", "Insufficient Data")

st.info("Status - Updated Every 20 min! [Last Update: {}]".format(
    datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')))

# Date Range Selector and Three-Column Layout
col1, col2, col3 = st.columns([2, 4, 2])

with col1:
    st.write("#### Filter by Date Range")
    start_date = st.date_input("Start Date", value=data['date'].min(), key='start_date')
    end_date = st.date_input("End Date", value=data['date'].max(), key='end_date')

# Apply the date filter
filtered_data = data[(data['date'] >= pd.Timestamp(start_date)) & (data['date'] <= pd.Timestamp(end_date))]

if filtered_data.empty:
    st.error("No data available for the selected date range.")
    st.stop()

# Live Charts Section in col2
with col2:
    st.write("### Model Live Chart")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=filtered_data['date'], y=filtered_data['nav'], mode='lines', name='NAV',
                             line=dict(color='blue')))
    fig.add_trace(
        go.Scatter(x=filtered_data['date'], y=filtered_data['nifty50 value'], mode='lines', name='Nifty50',
                   line=dict(color='red')))
    st.plotly_chart(fig, use_container_width=True)

    st.write("### Drawdown Chart")
    fig_dd = px.line(filtered_data, x='date', y='dd', title='Drawdown Over Time')
    fig_dd.update_traces(line_color='orange')
    st.plotly_chart(fig_dd, use_container_width=True)

# Model Performance Section in col3
with col3:
    st.write("### Model Performance")
    return_type = st.radio("Select Return Type", ['Inception', 'Yearly', 'Monthly', 'Weekly', 'Daily'], index=1)


    def calculate_performance(return_type):
        latest_value = filtered_data['nav'].iloc[-1]
        if return_type == 'Inception':
            inception_value = filtered_data['nav'].iloc[0]
            return (latest_value - inception_value) / inception_value * 100
        elif return_type == 'Yearly':
            past_date = filtered_data['date'].max() - pd.DateOffset(years=1)
            yearly_data = filtered_data[filtered_data['date'] >= past_date]
            if not yearly_data.empty:
                return (latest_value - yearly_data['nav'].iloc[0]) / yearly_data['nav'].iloc[0] * 100


    performance = calculate_performance(return_type)
    if performance is not None:
        st.write(f"{return_type} Performance: {performance:.2f}%")
