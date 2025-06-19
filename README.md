📈 Stock Price Predictor using LSTM and Flask

A web-based machine learning application that predicts the **next day's closing price** for any valid stock using an **LSTM (Long Short-Term Memory)** model. Built with **Flask**, **TensorFlow**, **Plotly**, and real-time data from **Yahoo Finance** via the `yfinance` API.

---

## 🚀 Key Features

- 🔍 **Dynamic stock symbol search** (not limited to a dropdown)
- 🤖 Predicts **next-day stock price** using deep learning (LSTM)
- 📉 Interactive **Plotly** charts (prediction + live intraday graph)
- 📊 Displays **model performance**: RMSE, MAE, loss curve
- ⚡ **Caches trained model** and scaler for faster reuse
- 🧠 Trained on **5 years of daily stock data**
- 💵 Displays **live stock price** and price change
- 🌐 Built with Flask + Bootstrap for a clean UI

---

## 📸 Screenshots

> 📌 _Add your actual images inside a `docs/` folder and update links below_

- ![Prediction Form](docs/predict_form.png)
- ![Output with Live Chart](docs/chart_output.png)

---

## 🧠 Technologies Used

- **Python 3**
- **Flask**
- **TensorFlow / Keras**
- **yfinance** (Yahoo Finance API)
- **Plotly**
- **Bootstrap 5**
- **scikit-learn**
- **joblib**

---

## 🗂 Project Structure

stock-price-predictor/
│
├── app.py # Flask app with model and route logic
├── requirements.txt # Project dependencies
├── .gitignore
├── lstm_model.h5 # Saved LSTM model
├── scaler.pkl # Saved MinMaxScaler
├── loss.pkl # Saved training loss history
│
├── templates/
│ └── index.html # Frontend interface (Bootstrap + Jinja2)
│
└── docs/ # (optional) Screenshots or documentation

yaml
Copy
Edit

---

## ⚙️ How to Run the Project Locally

### 1. Clone the Repository

```bash
git clone https://github.com/solomon-5A5/stock-price-predictor.git
cd stock-price-predictor
2. (Optional) Create and Activate a Virtual Environment
bash
Copy
Edit
python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
3. Install Dependencies
bash
Copy
Edit
pip install -r requirements.txt
4. Start the Flask Server
bash
Copy
Edit
python app.py
Then open your browser and go to:
👉 http://127.0.0.1:5000

🔮 Future Scope
📅 Extend to multi-day forecasts (7-day / 30-day)

🧾 Add sentiment analysis from news headlines

📈 Integrate technical indicators (RSI, MACD, etc.)

☁️ Deploy the app on Render, Heroku, or AWS

👥 Enable user accounts and stock watchlists

📚 References
TensorFlow

yFinance

Plotly Python

Flask Documentation