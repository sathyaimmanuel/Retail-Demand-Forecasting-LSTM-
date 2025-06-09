
# 📈 Retail Demand Forecasting System

A Streamlit-based web application to forecast daily demand for seasonal retail products using an LSTM deep learning model.

---

## 🚀 Overview

This project predicts future product demand for different retail stores using historical sales data. It is particularly designed for seasonal products where demand fluctuates over time.

Key Features:
- Select product and store for analysis
- Visualize historical demand trends
- Predict future demand using an LSTM model
- Interactive plots and forecast tables
- Download forecast results as a CSV

---

## 📂 Project Structure

```
.
├── retail_demand_app.py        # Streamlit application script
├── retail_demand_data.csv      # Historical retail demand dataset
├── demand_forecast_model.h5    # Pre-trained LSTM model
├── requirements.txt            # List of required packages
└── README.md                   # Project documentation
```

---

## 🛠️ How to Run

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/retail-demand-forecasting.git
cd retail-demand-forecasting
```

### 2. Set up your Python environment

Create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit app

```bash
streamlit run retail_demand_app.py
```

---

## 📊 Input Data Format

The `retail_demand_data.csv` file must include the following columns:

| Column Name   | Description                      |
|---------------|----------------------------------|
| `date`        | Date of the demand record        |
| `product_id`  | Unique identifier for product     |
| `store_id`    | Unique identifier for store       |
| `demand`      | Quantity sold or required demand  |

---

## 🧠 Model Details

- **Architecture**: LSTM (Long Short-Term Memory)
- **Lookback window**: 30 days
- **Forecast Horizon**: Adjustable (7 to 90 days)
- **Preprocessing**: Min-Max Scaling for model input

---

## 📦 Requirements

See `requirements.txt` for a full list. Key libraries include:

- `pandas`
- `numpy`
- `matplotlib`
- `scikit-learn`
- `tensorflow`
- `streamlit`

---

## 📌 Example Use Cases

- Inventory planning for seasonal products
- Demand forecasting in retail supply chains
- Business intelligence dashboards for retail managers

---

## 📝 Future Improvements

- Add multiple model support (ARIMA, Prophet)
- Incorporate external features (e.g., holidays, weather)
- Provide accuracy metrics with backtesting
- Integrate with a database or cloud storage

---

## 🙌 Credits

Developed by: **Sathya Immanuel Raj B**  
Email: sathyaimmanuelraj@gmail.com  
LinkedIn: [Sathya Immanuel Raj B](https://www.linkedin.com/in/sathya-immanuel-raj-b-43530b303)

---

## 📜 License

This project is licensed under the MIT License.
