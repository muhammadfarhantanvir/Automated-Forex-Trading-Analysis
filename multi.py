import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense
from sklearn.preprocessing import MinMaxScaler
from datetime import timedelta
from tensorflow.keras.callbacks import EarlyStopping

# Existing technical analysis functions
def calculate_sma(data, window):
    return data['Close'].rolling(window=window).mean()

def calculate_rsi(data, period=14):
    delta = data['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_macd(data):
    short_ema = data['Close'].ewm(span=12, adjust=False).mean()
    long_ema = data['Close'].ewm(span=26, adjust=False).mean()
    macd = short_ema - long_ema
    signal = macd.ewm(span=9, adjust=False).mean()
    return macd, signal

def calculate_bollinger_bands(data, window=20):
    sma = data['Close'].rolling(window=window).mean()
    std = data['Close'].rolling(window=window).std()
    upper_band = sma + 2 * std
    lower_band = sma - 2 * std
    return upper_band, lower_band

def calculate_stochastic_oscillator(data, k_period=14, d_period=3):
    high = data['High'].rolling(window=k_period).max()
    low = data['Low'].rolling(window=k_period).min()
    k = 100 * (data['Close'] - low) / (high - low)
    d = k.rolling(window=d_period).mean()
    return k, d

def generate_trading_signals(data, entry_rsi=30, exit_rsi=70, entry_macd_above_signal=True, exit_macd_below_signal=True):
    data['SMA_50'] = calculate_sma(data, 50)
    data['SMA_200'] = calculate_sma(data, 200)
    data['RSI'] = calculate_rsi(data)
    data['MACD'], data['Signal_Line'] = calculate_macd(data)
    data['Upper_BB'], data['Lower_BB'] = calculate_bollinger_bands(data)
    data['K'], data['D'] = calculate_stochastic_oscillator(data)
    data[['SMA_50', 'SMA_200', 'RSI', 'MACD', 'Signal_Line', 'Upper_BB', 'Lower_BB', 'K', 'D']] = data[['SMA_50', 'SMA_200', 'RSI', 'MACD', 'Signal_Line', 'Upper_BB', 'Lower_BB', 'K', 'D']].fillna(0)
    data['Trade_Signal'] = 'Hold'
    data.loc[(data['SMA_50'] > data['SMA_200']) & (data['RSI'] < entry_rsi) & (data['MACD'] > data['Signal_Line']), 'Trade_Signal'] = 'Buy'
    data.loc[(data['SMA_50'] < data['SMA_200']) & (data['RSI'] > exit_rsi) & (data['MACD'] < data['Signal_Line']), 'Trade_Signal'] = 'Sell'
    return data

def calculate_trading_metrics(data):
    data['Returns'] = data['Close'].pct_change()
    data['Strategy_Returns'] = data['Returns'] * (data['Trade_Signal'].shift(1).map({'Buy': 1, 'Sell': -1, 'Hold': 0}))
    data['Cumulative_Returns'] = (1 + data['Returns']).cumprod()
    data['Cumulative_Strategy_Returns'] = (1 + data['Strategy_Returns']).cumprod()
    sharpe_ratio = np.sqrt(252) * (data['Strategy_Returns'].mean() / data['Strategy_Returns'].std())
    cumulative_strategy_returns = data['Cumulative_Strategy_Returns']
    peak = cumulative_strategy_returns.expanding(min_periods=1).max()
    drawdown = (cumulative_strategy_returns - peak) / peak
    max_drawdown = drawdown.min()
    total_return = (data['Cumulative_Strategy_Returns'].iloc[-1] - 1) * 100
    return sharpe_ratio, max_drawdown, total_return

def generate_future_buy_sell_recommendations(data, future_data, entry_threshold=0.01, exit_threshold=0.01):
    recommendations = []
    current_position = 'Hold'
    entry_price = None
    for i in range(len(future_data)):
        predicted_price = future_data['Predicted_Close'].iloc[i]
        current_price = data['Close'].iloc[-1] if i == 0 else future_data['Predicted_Close'].iloc[i-1]
        if current_position == 'Hold':
            if predicted_price > current_price * (1 + entry_threshold):
                recommendations.append({
                    'Time': future_data['Time'].iloc[i],
                    'Price': predicted_price,
                    'Action': 'Buy',
                    'Reason': 'Price is expected to rise'
                })
                current_position = 'Long'
                entry_price = predicted_price
        elif current_position == 'Long':
            if predicted_price < entry_price * (1 - exit_threshold):
                recommendations.append({
                    'Time': future_data['Time'].iloc[i],
                    'Price': predicted_price,
                    'Action': 'Sell',
                    'Reason': 'Stop-Loss Triggered'
                })
                current_position = 'Hold'
                entry_price = None
            elif predicted_price > entry_price * (1 + exit_threshold):
                recommendations.append({
                    'Time': future_data['Time'].iloc[i],
                    'Price': predicted_price,
                    'Action': 'Sell',
                    'Reason': 'Take-Profit Triggered'
                })
                current_position = 'Hold'
                entry_price = None
    return pd.DataFrame(recommendations)

# New prediction functions
def predict_future_prices_arimax(data, days=5):
    data = data.dropna()
    exog_vars = data[['SMA_50', 'SMA_200', 'RSI', 'MACD', 'Signal_Line']]
    model = ARIMA(data['Close'], exog=exog_vars, order=(1, 1, 1))
    model_fit = model.fit()
    future_exog = exog_vars.iloc[-days:]
    future_predictions = model_fit.forecast(steps=days, exog=future_exog)
    future_dates = [data['Time'].iloc[-1] + timedelta(days=i) for i in range(1, days+1)]
    return pd.DataFrame({'Time': future_dates, 'Predicted_Close': future_predictions})

def predict_sarima(data, days=5):
    model = SARIMAX(data['Close'], order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
    model_fit = model.fit(disp=False)
    forecast = model_fit.forecast(steps=days)
    future_dates = [data['Time'].iloc[-1] + timedelta(days=i) for i in range(1, days+1)]
    return pd.DataFrame({'Time': future_dates, 'Predicted_Close': forecast})

def prepare_lstm_data(data, look_back=5):
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))
    X, y = [], []
    for i in range(len(scaled_data) - look_back):
        X.append(scaled_data[i:(i + look_back), 0])
        y.append(scaled_data[i + look_back, 0])
    return np.array(X), np.array(y), scaler

def predict_lstm(data, days=5, look_back=5):
    X, y, scaler = prepare_lstm_data(data, look_back)
    X = X.reshape((X.shape[0], X.shape[1], 1))
    model = Sequential()
    model.add(LSTM(50, input_shape=(look_back, 1)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    early_stopping = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)
    model.fit(X, y, epochs=100, batch_size=32, callbacks=[early_stopping], verbose=0)
    last_sequence = scaler.transform(data['Close'].values[-look_back:].reshape(-1, 1))
    predictions = []
    for _ in range(days):
        pred = model.predict(last_sequence.reshape(1, look_back, 1), verbose=0)
        predictions.append(pred[0, 0])
        last_sequence = np.roll(last_sequence, -1)
        last_sequence[-1] = pred
    predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
    future_dates = [data['Time'].iloc[-1] + timedelta(days=i) for i in range(1, days+1)]
    return pd.DataFrame({'Time': future_dates, 'Predicted_Close': predictions.flatten()})

def predict_gru(data, days=5, look_back=5):
    X, y, scaler = prepare_lstm_data(data, look_back)
    X = X.reshape((X.shape[0], X.shape[1], 1))
    model = Sequential()
    model.add(GRU(50, input_shape=(look_back, 1)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    early_stopping = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)
    model.fit(X, y, epochs=100, batch_size=32, callbacks=[early_stopping], verbose=0)
    last_sequence = scaler.transform(data['Close'].values[-look_back:].reshape(-1, 1))
    predictions = []
    for _ in range(days):
        pred = model.predict(last_sequence.reshape(1, look_back, 1), verbose=0)
        predictions.append(pred[0, 0])
        last_sequence = np.roll(last_sequence, -1)
        last_sequence[-1] = pred
    predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
    future_dates = [data['Time'].iloc[-1] + timedelta(days=i) for i in range(1, days+1)]
    return pd.DataFrame({'Time': future_dates, 'Predicted_Close': predictions.flatten()})

def predict_ensemble(data, days=5):
    arimax_pred = predict_future_prices_arimax(data, days)
    sarima_pred = predict_sarima(data, days)
    lstm_pred = predict_lstm(data, days)
    gru_pred = predict_gru(data, days)
    ensemble_pred = (arimax_pred['Predicted_Close'] + sarima_pred['Predicted_Close'] + 
                     lstm_pred['Predicted_Close'] + gru_pred['Predicted_Close']) / 4
    return pd.DataFrame({'Time': arimax_pred['Time'], 'Predicted_Close': ensemble_pred})

def calculate_metrics(actual, predicted):
    r2 = r2_score(actual, predicted)
    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    return r2, mae, rmse

# Streamlit UI
st.title("📈 Automated Forex Trading Analysis")
st.write("Upload your market data files (CSV) and get automated trading insights.")

# Sidebar for model selection
st.sidebar.title("Model Selection")
model_options = ['ARIMAX', 'SARIMA', 'LSTM', 'GRU', 'Ensemble']
selected_model = st.sidebar.selectbox("Select Prediction Model", model_options)

# File uploader
uploaded_files = st.file_uploader("Upload CSV Files", type=['csv'], accept_multiple_files=True)
dfs = {}
if uploaded_files:
    for file in uploaded_files:
        df = pd.read_csv(file)
        column_names = ["Time", "Open", "High", "Low", "Close", "Volume"]
        df.columns = column_names
        df["Time"] = pd.to_datetime(df["Time"])
        df[["Open", "High", "Low", "Close"]] = df[["Open", "High", "Low", "Close"]].astype(float)
        df["Volume"] = df["Volume"].astype(int)
        df = df[(df["Time"].dt.year == 2025) & (df["Time"].dt.month.isin([1, 2]))]
        df = generate_trading_signals(df, entry_rsi=30, exit_rsi=70, entry_macd_above_signal=True, exit_macd_below_signal=True)
        dfs[file.name] = df

if dfs:
    selected_file = st.selectbox("Select a file to analyze", list(dfs.keys()))
    df = dfs[selected_file]

    # Trading Summary
    buy_signals = df[df["Trade_Signal"] == "Buy"].shape[0]
    sell_signals = df[df["Trade_Signal"] == "Sell"].shape[0]
    st.subheader("Trading Summary")
    st.write(f"✅ **Buy Signals:** {buy_signals}")
    st.write(f"❌ **Sell Signals:** {sell_signals}")
    st.write(f"- **Business Context:** The strategy generated {buy_signals} buy signals and {sell_signals} sell signals over the period from January to February 2025. The higher number of sell signals compared to buy signals suggests that the strategy is more inclined to exit positions, possibly due to the bearish market trend.")

    # Market Trend
    if df['SMA_50'].iloc[-1] > df['SMA_200'].iloc[-1]:
        trend = "Uptrend 📈"
        trend_context = "The strategy identified an uptrend"
    else:
        trend = "Downtrend 📉"
        trend_context = "The strategy identified a downtrend"
    st.subheader("📊 Market Trend")
    st.write(f"**Current Market Trend:** {trend}")
    st.write(f"- **Business Context:** {trend_context}")

    # Trading Metrics
    sharpe_ratio, max_drawdown, total_return = calculate_trading_metrics(df)
    st.subheader("Trading Metrics")
    st.write("**Sharpe Ratio:**")
    st.write(f"- **Value:** {sharpe_ratio:.2f}")
    st.write(f"- **Business Context:** A Sharpe Ratio of {sharpe_ratio:.2f} indicates that the strategy is not providing a significant return relative to the risk taken. A negative Sharpe Ratio suggests that the strategy is underperforming compared to a risk-free asset.")
    st.write("**Maximum Drawdown:**")
    st.write(f"- **Value:** {max_drawdown:.2%}")
    st.write(f"- **Business Context:** The maximum drawdown of {max_drawdown:.2%} indicates that the strategy experienced a small decline from its peak value. This is relatively low, suggesting that the strategy is not subject to significant losses during volatile periods.")
    st.write("**Total Return:**")
    st.write(f"- **Value:** {total_return:.2f}%")
    st.write(f"- **Business Context:** The total return of {total_return:.2f}% indicates that the strategy resulted in a slight loss over the period from January to February 2025. This suggests that the strategy needs improvement to generate positive returns.")

    # Buy and Sell Prices
    buy_points = df[df["Trade_Signal"] == "Buy"]
    sell_points = df[df["Trade_Signal"] == "Sell"]
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

    # Model prediction
    model_functions = {
        'ARIMAX': predict_future_prices_arimax,
        'SARIMA': predict_sarima,
        'LSTM': predict_lstm,
        'GRU': predict_gru,
        'Ensemble': predict_ensemble
    }
    future_data = model_functions[selected_model](df, days=5)

    # Calculate metrics (using last 5 days of actual data)
    actual_last_5 = df['Close'].iloc[-5:].values
    if len(actual_last_5) == len(future_data['Predicted_Close']):
        r2, mae, rmse = calculate_metrics(actual_last_5, future_data['Predicted_Close'])
        st.subheader(f"{selected_model} Model Performance Metrics")
        st.write(f"R-squared: {r2:.4f}")
        st.write(f"MAE: {mae:.4f}")
        st.write(f"RMSE: {rmse:.4f}")

    # Future Trade Recommendations
    recommendations = generate_future_buy_sell_recommendations(df, future_data, entry_threshold=0.01, exit_threshold=0.01)
    st.subheader("Future Trade Recommendations")
    if not recommendations.empty:
        st.dataframe(recommendations)
    else:
        st.write("No Future Trade Recommendations Detected")

    # Trading Signals Visualization
    st.subheader("📈 Trading Signals Visualization")
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=df["Time"], open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"], name="Candlestick"))
    fig.add_trace(go.Scatter(x=df["Time"], y=df["SMA_50"], mode="lines", name="SMA 50", line=dict(color="blue")))
    fig.add_trace(go.Scatter(x=df["Time"], y=df["SMA_200"], mode="lines", name="SMA 200", line=dict(color="red")))
    if not buy_points.empty:
        fig.add_trace(go.Scatter(x=buy_points["Time"], y=buy_points["Close"], mode="markers", name="Buy Signal", marker=dict(color="green", size=10)))
    if not sell_points.empty:
        fig.add_trace(go.Scatter(x=sell_points["Time"], y=sell_points["Close"], mode="markers", name="Sell Signal", marker=dict(color="red", size=10)))
    if not recommendations.empty:
        buy_recs = recommendations[recommendations['Action'] == 'Buy']
        sell_recs = recommendations[recommendations['Action'] == 'Sell']
        if not buy_recs.empty:
            fig.add_trace(go.Scatter(x=buy_recs["Time"], y=buy_recs["Price"], mode="markers", name="Future Buy Recommendation", marker=dict(color="yellow", size=10)))
        if not sell_recs.empty:
            fig.add_trace(go.Scatter(x=sell_recs["Time"], y=sell_recs["Price"], mode="markers", name="Future Sell Recommendation", marker=dict(color="purple", size=10)))
    fig.update_layout(title="Trading Signals", xaxis_title="Time", yaxis_title="Price", template="plotly_dark")
    st.plotly_chart(fig)

    # Predicted Future Trend
    st.subheader(f"🔮 Predicted Future Trend ({selected_model})")
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=df["Time"], y=df["Close"], mode="lines", name="Historical Close Price"))
    fig2.add_trace(go.Scatter(x=future_data["Time"], y=future_data["Predicted_Close"], mode="lines+markers", 
                             name=f"Predicted Close Price ({selected_model})", line=dict(dash="dash", color="orange")))
    fig2.update_layout(title=f"Future Price Prediction ({selected_model})", xaxis_title="Time", yaxis_title="Price", template="plotly_dark")
    st.plotly_chart(fig2)