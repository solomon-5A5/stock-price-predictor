from flask import Flask, render_template, request
import plotly.graph_objs as go
from plotly.offline import plot
import yfinance as yf
import numpy as np
import pandas as pd
import os
import joblib
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout

app = Flask(__name__)

MODEL_PATH = 'lstm_model.h5'
SCALER_PATH = 'scaler.pkl'

def train_and_save_model(symbol):
    df = yf.download(symbol, period='5y', interval='1d')
    if df.empty or 'Close' not in df:
        return "No data found", None, None

    data = df[['Close']].dropna()
    if len(data) < 100:
        return "Not enough data", None, None

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
    model.fit(X, y, epochs=5, batch_size=32, verbose=0)

    model.save(MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)

    return model, scaler, data

def load_model_and_scaler(symbol='AAPL'):
    if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
        model = load_model(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        df = yf.download(symbol, period='5y', interval='1d')
        data = df[['Close']].dropna()
        return model, scaler, data
    else:
        return train_and_save_model(symbol)

def predict_stock_price(symbol):
    model, scaler, data = load_model_and_scaler(symbol)

    if model is None:
        return "Model loading failed", None

    scaled_data = scaler.transform(data)
    if len(scaled_data) < 60:
        return "Not enough data after scaling", None

    last_60_days = scaled_data[-60:]
    X_test = last_60_days.reshape(1, 60, 1)
    predicted_price = model.predict(X_test)
    predicted_value = round(scaler.inverse_transform(predicted_price)[0][0], 2)

    # Generate interactive Plotly chart
    actual_prices = data[-60:].values.flatten()
    trace1 = go.Scatter(y=actual_prices, mode='lines', name='Last 60 Days')
    trace2 = go.Scatter(
        x=[60],
        y=[predicted_value],
        mode='markers+text',
        name='Predicted Next',
        text=[f"${predicted_value}"],
        textposition='top center',
        marker=dict(color='red', size=10)
    )

    layout = go.Layout(
        title=f"{symbol} - Next Day Price Prediction",
        xaxis=dict(title="Days"),
        yaxis=dict(title="Price (USD)"),
        height=400
    )

    fig = go.Figure(data=[trace1, trace2], layout=layout)
    plot_html = plot(fig, output_type='div', include_plotlyjs=True)

    return predicted_value, plot_html

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    symbol = request.form['symbol'].upper()
    prediction, plot_html = predict_stock_price(symbol)
    return render_template('index.html', prediction=prediction, symbol=symbol, plot_html=plot_html)

if __name__ == '__main__':
    app.run(debug=True)
