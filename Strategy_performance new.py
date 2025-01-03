import streamlit as st
import pandas as pd
import datetime
import plotly.express as px
import plotly.graph_objects as go

# Replace with your actual Google Sheets CSV URL
google_sheets_url = "https://docs.google.com/spreadsheets/d/e/2PACX-1vTuyGRVZuafIk2s7moScIn5PAUcPYEyYIOOYJj54RXYUeugWmOP0iIToljSEMhHrg_Zp8Vab6YvBJDV/pub?output=csv"

@st.cache_data(ttl=0)
def load_data(url):
    data = pd.read_csv(url, header=0)
    data.columns = data.columns.str.strip().str.lower()
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

portfolio_value = data.iloc[0, 0]
nifty50_value = data.iloc[0, 2]
day_change = data.iloc[2, 0]

st.sidebar.markdown("#### Filter by Date Range")
start_date = st.sidebar.date_input("Start Date", value=data['date'].min())
end_date = st.sidebar.date_input("End Date", value=data['date'].max())

filtered_data = data[(data['date'] >= pd.Timestamp(start_date)) & (data['date'] <= pd.Timestamp(end_date))]

if filtered_data.empty:
    st.error("No data available for the selected date range.")
else:
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Account Value", f"₹{portfolio_value:,.0f}")
    with col2:
        st.metric("Day Change", f"₹{day_change:,.0f}", f"{filtered_data['day change %'].iloc[-1]:,.2f}%")
    with col3:
        st.metric("NIFTY50 Benchmark", f"{nifty50_value:,.0f}")
    with col4:
        if len(filtered_data) > 30:
            month_change = filtered_data['current value'].iloc[-1] - filtered_data['current value'].iloc[-30]
            month_change_percent = (month_change / filtered_data['current value'].iloc[-30] * 100)
            st.metric("Month Change", f"₹{month_change:,.0f}", f"{month_change_percent:.2f}%")
        else:
            st.metric("Month Change", "Insufficient Data")

    st.write("### Model Live Chart")
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=filtered_data['date'], y=filtered_data['nav'],
                              mode='lines', name='Multi-Factor', line=dict(color='blue')))
    fig1.add_trace(go.Scatter(x=filtered_data['date'], y=filtered_data['nifty50 value'],
                              mode='lines', name='nifty50', line=dict(color='red')))
    fig1.update_layout(height=400, width=900)
    st.plotly_chart(fig1)

    st.write("### Drawdown Live Chart")
    fig2 = px.line(filtered_data, x='date', y='dd', title="Drawdown Live Chart")
    fig2.update_traces(line_color='orange')
    fig2.update_layout(height=400, width=900)
    st.plotly_chart(fig2)


st.sidebar.write("### Model Performance")
return_type = st.sidebar.radio("Select Return Type", ['Inception', 'Yearly', 'Monthly', 'Weekly', 'Daily'], index=1)

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
    st.sidebar.write(f"{return_type} Performance: {performance:.2f}%")
