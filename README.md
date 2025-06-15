# stock-price-predictor
A web-based stock price predictor using LSTM and Flask
📈 Stock Price Predictor using LSTM and Flask
A web-based application that predicts the next day's closing price of selected stocks using an LSTM (Long Short-Term Memory) neural network. Built with Flask, Plotly, and TensorFlow, and powered by real-time data from the Yahoo Finance API.

🚀 Features
📊 Predicts next-day stock closing prices using deep learning

🌐 Web-based interface built with Flask + Bootstrap

📉 Interactive Plotly chart for visualizing trends

🏦 Supports popular stock tickers (AAPL, GOOGL, TSLA, etc.)

🧠 LSTM model trained on last 5 years of daily stock data

⚡ Caches trained model & scaler to avoid re-training every time

🖥️ Screenshots
Prediction Form	Interactive Chart

Replace these with your own screenshots saved in a docs/ folder.

📂 Project Structure
cpp
Copy
Edit
stock-price-predictor/
├── app.py
├── requirements.txt
├── .gitignore
├── templates/
│   └── index.html
├── lstm_model.h5
├── scaler.pkl
⚙️ How to Run Locally
bash
Copy
Edit
# 1. Clone this repo
git clone https://github.com/solomon-5A5/stock-price-predictor
cd stock-price-predictor

# 2. (Optional) Create a virtual environment
python -m venv venv
source venv/bin/activate  # or venv\\Scripts\\activate on Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the Flask app
python app.py
Go to http://127.0.0.1:5000 in your browser.

🧠 Technologies Used
Python 3

Flask

yfinance (Yahoo Finance API)

TensorFlow / Keras

Plotly

Bootstrap 5

scikit-learn + joblib

🔮 Future Scope
7-day or 30-day forecasting

Sentiment analysis integration from financial news

Support for technical indicators (volume, RSI, etc.)

Live deployment via Render/Heroku

User login and watchlist system

📚 References
TensorFlow

yFinance

Plotly Python

Flask

🧑‍💻 Author
SOLOMON PATTAPU

GitHub: @solomon-5A5

Email: pattapusolomon89@gmail.com







