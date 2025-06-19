from flask import Flask, render_template, request
import plotly.graph_objs as go
from plotly.offline import plot
import yfinance as yf
import numpy as np
import os
import joblib
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout

app = Flask(__name__)

MODEL_PATH = 'lstm_model.h5'
SCALER_PATH = 'scaler.pkl'
LOSS_PATH = 'loss.pkl'

def train_and_save_model(symbol):
    df = yf.download(symbol, period='5y', interval='1d')
    if df.empty or 'Close' not in df:
        return None, None, None

    data = df[['Close']].dropna()
    if len(data) < 100:
        return None, None, None

    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)

    X, y = [], []
    for i in range(60, len(scaled_data)):
        X.append(scaled_data[i - 60:i, 0])
        y.append(scaled_data[i, 0])

    X, y = np.array(X), np.array(y)
    X = X.reshape((X.shape[0], 60, 1))

    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(60, 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(50))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    history = model.fit(X, y, epochs=5, batch_size=32, verbose=0)

    model.save(MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    joblib.dump(history.history['loss'], LOSS_PATH)

    return model, scaler, data, history.history['loss']

def load_model_and_scaler(symbol='AAPL'):
    if not (os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH) and os.path.exists(LOSS_PATH)):
        return train_and_save_model(symbol)

    model = load_model(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    loss = joblib.load(LOSS_PATH)
    df = yf.download(symbol, period='5y', interval='1d')
    data = df[['Close']].dropna()
    return model, scaler, data, loss

def get_live_price(symbol):
    stock = yf.Ticker(symbol)
    data = stock.history(period='1d', interval='1m')
    if not data.empty:
        latest = data.iloc[-1]
        change = latest['Close'] - latest['Open']
        return round(latest['Close'], 2), round(change, 2)
    return None, None

def get_live_chart_html(symbol):
    stock = yf.Ticker(symbol)
    df = stock.history(period='1d', interval='1m')

    if df.empty:
        return "<p class='text-danger'>Live chart unavailable: No intraday data.</p>"

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines', name='Live Price'))

    fig.update_layout(
        title=f"{symbol} - Live Intraday Chart",
        xaxis_title="Time",
        yaxis_title="Price (USD)",
        height=400,
        margin=dict(l=30, r=30, t=50, b=30)
    )

    return plot(fig, output_type='div', include_plotlyjs=False)

def predict_stock_price(symbol):
    model, scaler, data, loss = load_model_and_scaler(symbol)

    if model is None:
        return None, None, None, None, None, None

    scaled_data = scaler.transform(data)
    if len(scaled_data) < 60:
        return None, None, None, None, None, None

    last_60_days = scaled_data[-60:]
    X_test = last_60_days.reshape(1, 60, 1)
    predicted_price = model.predict(X_test)
    predicted_value = round(scaler.inverse_transform(predicted_price)[0][0], 2)

    actual_prices = data[-60:].values.flatten()
    trace1 = go.Scatter(y=actual_prices, mode='lines', name='Last 60 Days')
    trace2 = go.Scatter(x=[60], y=[predicted_value], mode='markers+text',
                        name='Predicted Next', text=[f"${predicted_value}"],
                        textposition='top center', marker=dict(color='red', size=10))
    layout = go.Layout(title=f"{symbol} - Next Day Price Prediction",
                       xaxis=dict(title="Days"), yaxis=dict(title="Price (USD)"), height=400)
    fig = go.Figure(data=[trace1, trace2], layout=layout)
    plot_html = plot(fig, output_type='div', include_plotlyjs=True)

    X, y = [], []
    for i in range(60, len(scaled_data)):
        X.append(scaled_data[i - 60:i, 0])
        y.append(scaled_data[i, 0])
    X = np.array(X).reshape(len(X), 60, 1)
    y_pred = model.predict(X)
    rmse = round(np.sqrt(mean_squared_error(y, y_pred)), 2)
    mae = round(mean_absolute_error(y, y_pred), 2)

    fig_loss = go.Figure()
    fig_loss.add_trace(go.Scatter(y=loss, mode='lines', name="Training Loss"))
    fig_loss.update_layout(title="Model Training Loss", xaxis_title="Epoch", yaxis_title="Loss")
    loss_chart = plot(fig_loss, output_type='div', include_plotlyjs=False)

    return predicted_value, plot_html, rmse, mae, loss_chart, data

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    symbol = request.form['symbol'].upper().strip()

    # Validate symbol
    ticker = yf.Ticker(symbol)
    info = ticker.info
    if 'shortName' not in info or info.get('regularMarketPrice') is None:
        return render_template('index.html', error=f"❌ '{symbol}' is not a valid or supported stock symbol.")

    prediction, plot_html, rmse, mae, loss_chart, data = predict_stock_price(symbol)

    if prediction is None:
        return render_template('index.html', error=f"⚠️ Could not generate prediction for '{symbol}'. Try another stock.")

    live_price, price_change = get_live_price(symbol)
    live_chart = get_live_chart_html(symbol)

    return render_template('index.html',
                           prediction=prediction,
                           symbol=symbol,
                           plot_html=plot_html,
                           rmse=rmse,
                           mae=mae,
                           loss_chart=loss_chart,
                           live_price=live_price,
                           price_change=price_change,
                           live_chart=live_chart)

if __name__ == '__main__':
    app.run(debug=True)
