import streamlit as st
import pandas as pd
import quantstats as qs
import numpy as np

# Replace with your published CSV link
csv_url = "https://docs.google.com/spreadsheets/d/e/2PACX-1vTuyGRVZuafIk2s7moScIn5PAUcPYEyYIOOYJj54RXYUeugWmOP0iIToljSEMhHrg_Zp8Vab6YvBJDV/pub?output=csv"

# Load the data into a Pandas DataFrame
@st.cache_data
def load_data(csv_url):
    try:
        data = pd.read_csv(csv_url)
        return data
    except Exception as e:
        st.error(f"An error occurred: {e}")
        return None

# Data cleaning and preprocessing
def preprocess_data(data):
    rows_to_delete = data[data['Date'].isin(['Portfolio Value', 'Absolute Gain', 'Nifty50', 'Day Change'])].index
    data.drop(rows_to_delete, inplace=True)
    data = data.dropna(subset=['NAV'])
    data['Date'] = pd.to_datetime(data['Date'], format='%d-%b-%y')
    data = data.sort_values(by='Date')
    data.set_index('Date', inplace=True)
    data['NAV'] = pd.to_numeric(data['NAV'])
    data['Nifty50 Change %'] = data['Nifty50 Change %'].str.rstrip('%').astype('float') / 100
    data['Nifty50 NAV'] = (1 + data['Nifty50 Change %']).cumprod()
    return data

# Calculate the daily returns
def calculate_returns(data):
    returns = data['NAV'].pct_change().dropna()
    nifty50 = data['Nifty50 Change %'].dropna()
    return returns, nifty50

# Main function for Streamlit app
def main():
    # Set page configuration to wide layout
    st.set_page_config(page_title="Portfolio Report", layout="wide")

    # Inject custom CSS for full-width iframe
    custom_css = """
    <style>
        .main iframe {
            width: 100% !important;
            height: calc(100vh - 2rem) !important;
            border: none !important;
        }
        .main > div {
            padding: 0 !important;
        }
    </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)

    # Load and preprocess data
    data = load_data(csv_url)
    if data is not None:
        processed_data = preprocess_data(data)
        returns, nifty50 = calculate_returns(processed_data)

        # Generate QuantStats report
        try:
            qs.reports.html(returns, nifty50, output="report.html")
            with open("report.html", "r") as f:
                report_html = f.read()

            # Embed the QuantStats report in full width
            st.components.v1.html(report_html, scrolling=True)
        except Exception as e:
            st.error(f"Error displaying QuantStats report: {e}")

if __name__ == "__main__":
    main()
