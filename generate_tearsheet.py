import pandas as pd
import numpy as np
import quantstats as qs
import streamlit as st
import matplotlib.pyplot as plt


# Step 1: Fetch data from the published CSV link
def fetch_data_from_csv(csv_url):
    try:
        # Read the data from the published Google Sheet CSV link
        df = pd.read_csv(csv_url)

        # Convert the 'Date' column to datetime format (handling the 'DD-MMM-YY' format)
        df['Date'] = pd.to_datetime(df['Date'], format='%d-%b-%y')

        # Check if the Date column was successfully parsed
        if 'Date' not in df.columns:
            raise ValueError("The 'Date' column is missing in the data.")

        # Set 'Date' column as index
        df.set_index('Date', inplace=True)
        return df
    except Exception as e:
        print(f"Error fetching data: {e}")
        return None


# Step 2: Process Portfolio NAV and Benchmark (Nifty50 NAV)
def process_data(df):
    try:
        # Check for required columns
        if 'Portfolio NAV' not in df.columns or 'Nifty50 NAV' not in df.columns:
            raise ValueError("'Portfolio NAV' and/or 'Nifty50 NAV' columns are missing in the data.")

        # Calculate daily returns for portfolio (Portfolio NAV) and benchmark (Nifty50 NAV)
        df['Portfolio Returns'] = df['Portfolio NAV'].pct_change()
        df['Benchmark Returns'] = df['Nifty50 NAV'].pct_change()

        # Drop NaN values to ensure no issues while generating the tearsheet
        df.dropna(subset=['Portfolio Returns', 'Benchmark Returns'], inplace=True)

        return df
    except Exception as e:
        print(f"Error processing data: {e}")
        return None


# Step 3: Calculate Key Performance Metrics
def calculate_key_metrics(df):
    # Portfolio and Benchmark returns
    portfolio_returns = df['Portfolio Returns']
    benchmark_returns = df['Benchmark Returns']

    # 1. Cumulative Return
    cumulative_return = (1 + portfolio_returns).prod() - 1

    # 2. CAGR (Compound Annual Growth Rate)
    years = (df.index[-1] - df.index[0]).days / 365.25
    cagr = (1 + cumulative_return) ** (1 / years) - 1

    # 3. Sharpe Ratio
    sharpe_ratio = portfolio_returns.mean() / portfolio_returns.std() * np.sqrt(252)  # Annualize Sharpe Ratio

    # 4. Max Drawdown
    rolling_max = (1 + portfolio_returns).cumprod().cummax()
    drawdown = (1 + portfolio_returns).cumprod() / rolling_max - 1
    max_drawdown = drawdown.min()

    # 5. Sortino Ratio
    downside_returns = portfolio_returns[portfolio_returns < 0]
    sortino_ratio = portfolio_returns.mean() / downside_returns.std() * np.sqrt(252)

    # 6. Omega Ratio
    omega_ratio = (portfolio_returns[portfolio_returns > 0].sum() /
                   -portfolio_returns[portfolio_returns < 0].sum())

    # 7. Information Ratio
    tracking_error = (portfolio_returns - benchmark_returns).std() * np.sqrt(252)
    information_ratio = (portfolio_returns.mean() - benchmark_returns.mean()) / tracking_error

    # 8. Calmar Ratio
    calmar_ratio = cagr / -max_drawdown

    # 9. Recovery Factor
    recovery_factor = cumulative_return / abs(max_drawdown)

    # 10. Win Rate (Percentage of Positive Days)
    win_rate = (portfolio_returns > 0).mean() * 100

    # 11. Loss Rate (Percentage of Negative Days)
    loss_rate = (portfolio_returns < 0).mean() * 100

    # Return all the metrics in a dictionary
    metrics = {
        "Cumulative Return": f"{cumulative_return * 100:.2f}%",
        "CAGR": f"{cagr * 100:.2f}%",
        "Sharpe Ratio": f"{sharpe_ratio:.2f}",
        "Max Drawdown": f"{max_drawdown * 100:.2f}%",
        "Sortino Ratio": f"{sortino_ratio:.2f}",
        "Omega Ratio": f"{omega_ratio:.2f}",
        "Information Ratio": f"{information_ratio:.2f}",
        "Calmar Ratio": f"{calmar_ratio:.2f}",
        "Recovery Factor": f"{recovery_factor:.2f}",
        "Win Rate": f"{win_rate:.2f}%",
        "Loss Rate": f"{loss_rate:.2f}%",
    }

    return metrics


# Step 4: Generate a strategy tearsheet
def generate_tearsheet(portfolio_returns, benchmark_returns):
    try:
        # Generate the QuantStats tearsheet
        qs.reports.html(
            portfolio_returns,
            benchmark=benchmark_returns,
            output='strategy_tearsheet.html'
        )
        print("Tearsheet successfully generated: strategy_tearsheet.html")
    except Exception as e:
        print(f"Error generating tearsheet: {e}")


# Step 5: Generate Portfolio vs Benchmark Graph
def generate_performance_graph(portfolio_returns, benchmark_returns):
    plt.figure(figsize=(10, 6))

    # Cumulative sum for both portfolio and benchmark returns to plot cumulative performance
    portfolio_cumulative = portfolio_returns.cumsum()
    benchmark_cumulative = benchmark_returns.cumsum()

    # Plot the graphs
    plt.plot(portfolio_cumulative, label='Portfolio Cumulative Returns', color='blue')
    plt.plot(benchmark_cumulative, label='Nifty50 Cumulative Returns', color='orange')

    plt.title('Portfolio vs Benchmark (Nifty50) Cumulative Returns')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Returns')
    plt.legend(loc='best')
    plt.grid(True)

    # Show the plot using Streamlit
    st.pyplot(plt)


# Step 6: Generate Portfolio and Benchmark Daily Returns Graph
def generate_daily_returns_graph(portfolio_returns, benchmark_returns):
    plt.figure(figsize=(10, 6))

    # Plot the daily returns for both portfolio and benchmark
    plt.plot(portfolio_returns, label='Portfolio Daily Returns', color='blue', alpha=0.7)
    plt.plot(benchmark_returns, label='Nifty50 Daily Returns', color='orange', alpha=0.7)

    plt.title('Portfolio vs Benchmark Daily Returns')
    plt.xlabel('Date')
    plt.ylabel('Daily Returns')
    plt.legend(loc='best')
    plt.grid(True)

    # Show the plot using Streamlit
    st.pyplot(plt)


# Step 7: Generate Rolling Sharpe Ratio Graph
def generate_rolling_sharpe_ratio(portfolio_returns):
    plt.figure(figsize=(10, 6))

    # Calculate the rolling Sharpe ratio (with a 30-day window)
    rolling_sharpe = portfolio_returns.rolling(window=30).mean() / portfolio_returns.rolling(window=30).std()

    plt.plot(rolling_sharpe, label='Rolling Sharpe Ratio (30 days)', color='green')

    plt.title('Rolling Sharpe Ratio')
    plt.xlabel('Date')
    plt.ylabel('Sharpe Ratio')
    plt.legend(loc='best')
    plt.grid(True)

    # Show the plot using Streamlit
    st.pyplot(plt)


# Additional Graphs
def generate_eoy_vs_benchmark(df):
    df['Year'] = df.index.year
    eoy_returns = df.groupby('Year').apply(lambda x: (1 + x['Portfolio Returns']).prod() - 1)
    benchmark_eoy_returns = df.groupby('Year').apply(lambda x: (1 + x['Benchmark Returns']).prod() - 1)

    plt.figure(figsize=(10, 6))
    plt.plot(eoy_returns.index, eoy_returns, label="Portfolio EOY Returns", color='blue')
    plt.plot(benchmark_eoy_returns.index, benchmark_eoy_returns, label="Benchmark EOY Returns", color='orange')
    plt.title("End of Year (EOY) vs Benchmark")
    plt.xlabel("Year")
    plt.ylabel("Cumulative Return (%)")
    plt.legend()
    plt.grid(True)
    st.pyplot(plt)


def generate_monthly_returns_distribution(df):
    df['Month'] = df.index.month
    monthly_returns = df.groupby('Month')['Portfolio Returns'].mean()

    plt.figure(figsize=(10, 6))
    plt.hist(monthly_returns, bins=12, color='blue', alpha=0.7)
    plt.title("Distribution of Monthly Returns")
    plt.xlabel("Monthly Return (%)")
    plt.ylabel("Frequency")
    plt.grid(True)
    st.pyplot(plt)


def generate_worst_5_drawdowns(df):
    rolling_max = (1 + df['Portfolio Returns']).cumprod().cummax()
    drawdown = (1 + df['Portfolio Returns']).cumprod() / rolling_max - 1
    worst_drawdowns = drawdown.sort_values(ascending=False).head(5)

    plt.figure(figsize=(10, 6))
    plt.bar(worst_drawdowns.index, worst_drawdowns.values, color='red')
    plt.title("Worst 5 Drawdown Periods")
    plt.xlabel("Date")
    plt.ylabel("Drawdown (%)")
    plt.grid(True)
    st.pyplot(plt)


def generate_underwater_plot(df):
    rolling_max = (1 + df['Portfolio Returns']).cumprod().cummax()
    drawdown = (1 + df['Portfolio Returns']).cumprod() / rolling_max - 1

    plt.figure(figsize=(10, 6))
    plt.fill_between(drawdown.index, drawdown, color='red', alpha=0.5)
    plt.title("Underwater Plot")
    plt.xlabel("Date")
    plt.ylabel("Drawdown (%)")
    plt.grid(True)
    st.pyplot(plt)


def generate_monthly_returns(df):
    monthly_returns = df.resample('M').sum()

    st.subheader("Monthly Returns (%)")
    st.write(monthly_returns[['Portfolio Returns']] * 100)


def generate_return_quantiles(df):
    quantiles = df['Portfolio Returns'].quantile([0.25, 0.5, 0.75])

    plt.figure(figsize=(10, 6))
    plt.bar(quantiles.index, quantiles.values, color='blue')
    plt.title("Return Quantiles (25%, 50%, 75%)")
    plt.xlabel("Quantiles")
    plt.ylabel("Return (%)")
    plt.grid(True)
    st.pyplot(plt)


# Main Function (Streamlit App)
def main():
    st.title("Portfolio Strategy Tearsheet")

    # Provide the CSV URL here
    csv_url = "https://docs.google.com/spreadsheets/d/e/2PACX-1vTsrrgpxeUhvPSMut74cBGH8lLKJFQMKzQ789FhYs51peqb-MdiMd_ANl4aT4HzOL2hnnxTXSHx9sLJ/pub?gid=1312053825&single=true&output=csv"

    # Step 1: Fetch Data
    df = fetch_data_from_csv(csv_url)
    if df is None:
        st.error("Error fetching data.")
        return

    # Step 2: Process Data
    df = process_data(df)
    if df is None:
        st.error("Error processing data.")
        return

    # Step 3: Calculate Key Metrics
    metrics = calculate_key_metrics(df)
    for metric, value in metrics.items():
        st.subheader(metric)
        st.write(value)

    # Step 4: Generate Strategy Tearsheets
    generate_tearsheet(df['Portfolio Returns'], df['Benchmark Returns'])

    # Step 5: Generate Performance Graphs
    generate_performance_graph(df['Portfolio Returns'], df['Benchmark Returns'])

    # Step 6: Generate Additional Graphs
    generate_eoy_vs_benchmark(df)
    generate_monthly_returns_distribution(df)
    generate_worst_5_drawdowns(df)
    generate_underwater_plot(df)
    generate_monthly_returns(df)
    generate_return_quantiles(df)
    generate_rolling_sharpe_ratio(df['Portfolio Returns'])


if __name__ == "__main__":
    main()
