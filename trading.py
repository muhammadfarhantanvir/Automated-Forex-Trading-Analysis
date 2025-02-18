import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from statsmodels.tsa.statespace.sarimax import SARIMAX
from datetime import timedelta

# Function to calculate SMA
def calculate_sma(data, window):
    return data['Close'].rolling(window=window).mean()

# Function to calculate RSI
def calculate_rsi(data, period=14):
    delta = data['Close'].diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=period).mean()
    avg_loss = pd.Series(loss).rolling(window=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# Function to calculate MACD and Signal Line
def calculate_macd(data):
    short_ema = data['Close'].ewm(span=12, adjust=False).mean()
    long_ema = data['Close'].ewm(span=26, adjust=False).mean()
    macd = short_ema - long_ema
    signal = macd.ewm(span=9, adjust=False).mean()
    return macd, signal

# Function to generate trading signals
def generate_trading_signals(data):
    data['SMA_50'] = calculate_sma(data, 50)
    data['SMA_200'] = calculate_sma(data, 200)
    data['RSI'] = calculate_rsi(data)
    data['MACD'], data['Signal_Line'] = calculate_macd(data)
    # Fill NaN values to avoid errors
    data[['SMA_50', 'SMA_200', 'RSI', 'MACD', 'Signal_Line']] = data[['SMA_50', 'SMA_200', 'RSI', 'MACD', 'Signal_Line']].fillna(0)
    # Define trading conditions
    data['Trade_Signal'] = 'Hold'
    data.loc[(data['SMA_50'] > data['SMA_200']) & (data['RSI'] < 40) & (data['MACD'] > data['Signal_Line']), 'Trade_Signal'] = 'Buy'
    data.loc[(data['SMA_50'] < data['SMA_200']) & (data['RSI'] > 60) & (data['MACD'] < data['Signal_Line']), 'Trade_Signal'] = 'Sell'
    return data

# Function to predict future trend using ARIMAX model
def predict_future_prices_arimax(data, days=5):
    data = data.dropna()
    # Define exogenous variables
    exog_vars = data[['SMA_50', 'SMA_200', 'RSI', 'MACD', 'Signal_Line']]
    # Fit ARIMAX model
    model = SARIMAX(data['Close'], exog=exog_vars, order=(1, 1, 1), seasonal_order=(0, 0, 0, 0))
    model_fit = model.fit(disp=False)
    # Forecast future prices
    future_exog = exog_vars.iloc[-days:]
    future_predictions = model_fit.forecast(steps=days, exog=future_exog)
    future_dates = [data['Time'].iloc[-1] + timedelta(days=i) for i in range(1, days+1)]
    return pd.DataFrame({'Time': future_dates, 'Predicted_Close': future_predictions})

# Function to calculate trading metrics
def calculate_trading_metrics(data):
    data['Returns'] = data['Close'].pct_change()
    data['Strategy_Returns'] = data['Returns'] * (data['Trade_Signal'].shift(1).map({'Buy': 1, 'Sell': -1, 'Hold': 0}))
    
    # Calculate cumulative returns
    data['Cumulative_Returns'] = (1 + data['Returns']).cumprod()
    data['Cumulative_Strategy_Returns'] = (1 + data['Strategy_Returns']).cumprod()
    
    # Calculate Sharpe Ratio
    sharpe_ratio = np.sqrt(252) * (data['Strategy_Returns'].mean() / data['Strategy_Returns'].std())
    
    # Calculate Maximum Drawdown
    cumulative_strategy_returns = data['Cumulative_Strategy_Returns']
    peak = cumulative_strategy_returns.expanding(min_periods=1).max()
    drawdown = (cumulative_strategy_returns - peak) / peak
    max_drawdown = drawdown.min()
    
    # Calculate Total Return
    total_return = (data['Cumulative_Strategy_Returns'].iloc[-1] - 1) * 100
    
    return sharpe_ratio, max_drawdown, total_return

# Function to generate future buy/sell recommendations based on predicted prices
def generate_future_buy_sell_recommendations(data, future_data, stop_loss_pct=0.01, take_profit_pct=0.01):
    recommendations = []
    current_position = 'Hold'
    entry_price = None
    
    for i in range(len(future_data)):
        predicted_price = future_data['Predicted_Close'].iloc[i]
        current_price = data['Close'].iloc[-1] if i == 0 else future_data['Predicted_Close'].iloc[i-1]
        
        if current_position == 'Hold':
            # Check for buy signal based on future price
            if predicted_price > current_price * (1 + stop_loss_pct):
                recommendations.append({
                    'Time': future_data['Time'].iloc[i],
                    'Price': predicted_price,
                    'Action': 'Buy',
                    'Reason': 'Price is expected to rise'
                })
                current_position = 'Long'
                entry_price = predicted_price
        
        elif current_position == 'Long':
            # Check for sell signal based on future price
            if predicted_price < entry_price * (1 - stop_loss_pct):
                recommendations.append({
                    'Time': future_data['Time'].iloc[i],
                    'Price': predicted_price,
                    'Action': 'Sell',
                    'Reason': 'Stop-Loss Triggered'
                })
                current_position = 'Hold'
                entry_price = None
            elif predicted_price > entry_price * (1 + take_profit_pct):
                recommendations.append({
                    'Time': future_data['Time'].iloc[i],
                    'Price': predicted_price,
                    'Action': 'Sell',
                    'Reason': 'Take-Profit Triggered'
                })
                current_position = 'Hold'
                entry_price = None
    
    return pd.DataFrame(recommendations)

# Function to generate specific buy/sell recommendations for tomorrow
def generate_specific_recommendations_for_tomorrow(data, future_data, stop_loss_pct=0.01, take_profit_pct=0.01):
    recommendations = []
    current_price = data['Close'].iloc[-1]
    tomorrow_date = data['Time'].iloc[-1] + timedelta(days=1)
    
    # Check for buy signal based on tomorrow's predicted price
    if future_data.iloc[0]['Predicted_Close'] > current_price * (1 + stop_loss_pct):
        recommendations.append({
            'Time': tomorrow_date,
            'Price': future_data.iloc[0]['Predicted_Close'],
            'Action': 'Buy',
            'Reason': 'Price is expected to rise'
        })
    
    # Check for sell signal based on tomorrow's predicted price
    if future_data.iloc[0]['Predicted_Close'] < current_price * (1 - stop_loss_pct):
        recommendations.append({
            'Time': tomorrow_date,
            'Price': future_data.iloc[0]['Predicted_Close'],
            'Action': 'Sell',
            'Reason': 'Price is expected to fall'
        })
    
    return pd.DataFrame(recommendations)

# Streamlit UI
st.title("📈 Automated Forex Trading Analysis")
st.write("Upload your market data files (CSV) and get automated trading insights.")

# File uploader
uploaded_files = st.file_uploader("Upload CSV Files", type=['csv'], accept_multiple_files=True)

# Store dataframes in a dictionary
dfs = {}
if uploaded_files:
    for file in uploaded_files:
        df = pd.read_csv(file)
        # Ensure proper column names
        column_names = ["Time", "Open", "High", "Low", "Close", "Volume"]
        df.columns = column_names
        df["Time"] = pd.to_datetime(df["Time"])
        # Convert data types
        df[["Open", "High", "Low", "Close"]] = df[["Open", "High", "Low", "Close"]].astype(float)
        df["Volume"] = df["Volume"].astype(int)
        # Filter data to include only January and February 2025
        df = df[(df["Time"].dt.year == 2025) & (df["Time"].dt.month.isin([1, 2]))]
        # Apply trading analysis
        df = generate_trading_signals(df)
        # Store dataframe in dictionary with filename as key
        dfs[file.name] = df

# Dropdown menu to select which file to analyze
if dfs:
    selected_file = st.selectbox("Select a file to analyze", list(dfs.keys()))
    df = dfs[selected_file]

    # Summary of trade signals
    buy_signals = df[df["Trade_Signal"] == "Buy"].shape[0]
    sell_signals = df[df["Trade_Signal"] == "Sell"].shape[0]

    st.subheader("Trading Summary")
    st.write(f"✅ **Buy Signals:** {buy_signals}")
    st.write(f"❌ **Sell Signals:** {sell_signals}")

    # Determine market trend
    if df['SMA_50'].iloc[-1] > df['SMA_200'].iloc[-1]:
        trend = "Uptrend 📈 (Bullish)"
    else:
        trend = "Downtrend 📉 (Bearish)"
    st.subheader("📊 Market Trend")
    st.write(f"**Current Market Trend:** {trend}")

    # Calculate trading metrics
    sharpe_ratio, max_drawdown, total_return = calculate_trading_metrics(df)
    st.subheader("Trading Metrics")
    st.write(f"**Sharpe Ratio:** {sharpe_ratio:.2f}")
    st.write(f"**Maximum Drawdown:** {max_drawdown:.2%}")
    st.write(f"**Total Return:** {total_return:.2f}%")

    # Extract buy and sell prices
    buy_points = df[df["Trade_Signal"] == "Buy"]
    sell_points = df[df["Trade_Signal"] == "Sell"]

    # Display buy and sell prices
    st.subheader("Buy and Sell Prices")
    if not buy_points.empty:
        st.write("Buy Prices:")
        st.dataframe(buy_points[["Time", "Close"]])
    else:
        st.write("No Buy Signals Detected")

    if not sell_points.empty:
        st.write("Sell Prices:")
        st.dataframe(sell_points[["Time", "Close"]])
    else:
        st.write("No Sell Signals Detected")

    # Predict future trend using ARIMAX model
    future_data = predict_future_prices_arimax(df, days=10)

    # Generate future buy/sell recommendations based on predicted prices
    recommendations = generate_future_buy_sell_recommendations(df, future_data, stop_loss_pct=0.01, take_profit_pct=0.01)

    # Display recommendations
    st.subheader("Future Trade Recommendations")
    if not recommendations.empty:
        st.dataframe(recommendations)
    else:
        st.write("No Future Trade Recommendations Detected")


    # Plot signals on chart
    st.subheader("📈 Trading Signals Visualization")
    fig = go.Figure()
    # Add closing price
    fig.add_trace(go.Scatter(x=df["Time"], y=df["Close"], mode="lines", name="Close Price"))
    # Add moving averages
    fig.add_trace(go.Scatter(x=df["Time"], y=df["SMA_50"], mode="lines", name="SMA 50", line=dict(color="blue")))
    fig.add_trace(go.Scatter(x=df["Time"], y=df["SMA_200"], mode="lines", name="SMA 200", line=dict(color="red")))
    # Add Buy signals
    if not buy_points.empty:
        fig.add_trace(go.Scatter(x=buy_points["Time"], y=buy_points["Close"], mode="markers", name="Buy Signal", marker=dict(color="green", size=10)))
    # Add Sell signals
    if not sell_points.empty:
        fig.add_trace(go.Scatter(x=sell_points["Time"], y=sell_points["Close"], mode="markers", name="Sell Signal", marker=dict(color="red", size=10)))
    # Add Future Buy/Sell Recommendations
    if not recommendations.empty:
        buy_recs = recommendations[recommendations['Action'] == 'Buy']
        sell_recs = recommendations[recommendations['Action'] == 'Sell']
        if not buy_recs.empty:
            fig.add_trace(go.Scatter(x=buy_recs["Time"], y=buy_recs["Price"], mode="markers", name="Future Buy Recommendation", marker=dict(color="yellow", size=10)))
        if not sell_recs.empty:
            fig.add_trace(go.Scatter(x=sell_recs["Time"], y=sell_recs["Price"], mode="markers", name="Future Sell Recommendation", marker=dict(color="purple", size=10)))
    # Update layout
    fig.update_layout(title="Trading Signals", xaxis_title="Time", yaxis_title="Price", template="plotly_dark")
    st.plotly_chart(fig)

    # Plot future trend
    st.subheader("🔮 Predicted Future Trend")
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=df["Time"], y=df["Close"], mode="lines", name="Historical Close Price"))
    fig2.add_trace(go.Scatter(x=future_data["Time"], y=future_data["Predicted_Close"], mode="lines+markers", name="Predicted Close Price", line=dict(dash="dash", color="orange")))
    fig2.update_layout(title="Future Price Prediction", xaxis_title="Time", yaxis_title="Price", template="plotly_dark")
    st.plotly_chart(fig2)