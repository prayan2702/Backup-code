#Pandas just released 2.0.0, which breaks QuantStats:
pip install pandas==1.5.3
#*********************************

import pandas as pd
import quantstats as qs
import pytz
import numpy as np

# Replace with your published CSV link
csv_url = "https://docs.google.com/spreadsheets/d/e/2PACX-1vTuyGRVZuafIk2s7moScIn5PAUcPYEyYIOOYJj54RXYUeugWmOP0iIToljSEMhHrg_Zp8Vab6YvBJDV/pub?output=csv"

# Load the data into a Pandas DataFrame
try:
    data = pd.read_csv(csv_url)
except Exception as e:
    print(f"An error occurred: {e}")
# Display the first few rows of the DataFrame
print(data.head())

# Delete rows with Portfolio Value, Absolute Gain, Nifty50, Day Change
rows_to_delete = data[data['Date'].isin(['Portfolio Value', 'Absolute Gain', 'Nifty50', 'Day Change'])].index
data.drop(rows_to_delete, inplace=True)

# Delete the rows which does not have NAV value
data = data.dropna(subset=['NAV'])

# Convert the 'Date' column to datetime objects and set it as the index
data['Date'] = pd.to_datetime(data['Date'], format='%d-%b-%y')
# Sort the data by Date in ascending order
data = data.sort_values(by='Date')
# Make the index tz aware
data['Date'] = data['Date'].apply(lambda x: x.replace(tzinfo=None))
data.set_index('Date', inplace=True)

# Convert the 'NAV' column to numeric values
data['NAV'] = pd.to_numeric(data['NAV'])

# Remove % in Nifty50 Change % Column
data['Nifty50 Change %'] = data['Nifty50 Change %'].str.rstrip('%').astype('float')/100

# Make a column of Nifty50 NAV
data['Nifty50 NAV'] = data['Nifty50 Value'].cumprod()

# Calculate the daily returns
returns = data['NAV'].pct_change().dropna()
nifty50 = data['Nifty50 Change %'].dropna()

# Verify if there are overlap in date range
start_date = max(returns.index[0], nifty50.index[0])
end_date = min(returns.index[-1], nifty50.index[-1])

# Convert start_date and end_date to Timestamp objects
start_date = pd.Timestamp(start_date)
end_date = pd.Timestamp(end_date)

# Filter the data to the overlapping date range
returns = returns[start_date:end_date]
nifty50 = nifty50[start_date:end_date]

# Replace NaN and infinite values with 0
returns = returns.replace([float('inf'), float('-inf')], 0).fillna(0)
nifty50 = nifty50.replace([float('inf'), float('-inf')], 0).fillna(0)

# Create the monthly return
returns_df = returns.to_frame()

#Add the Year, Month, and Returns column in dataframe
returns_df['Year'] = returns_df.index.year
returns_df['Month'] = returns_df.index.month
returns_df.rename(columns={"NAV": "Returns"}, inplace=True)

# Generate the full report
qs.reports.full(returns, benchmark=nifty50)

# You can save it as html
#qs.reports.html(returns, "your_full_report.html", benchmark=nifty50)
