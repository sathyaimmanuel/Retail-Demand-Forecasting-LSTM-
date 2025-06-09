import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import streamlit as st
import matplotlib.dates as mdates

# Load data and model with caching
@st.cache_data
def load_data():
    return pd.read_csv('retail_demand_data.csv')

@st.cache_resource
def load_forecast_model():
    return load_model('demand_forecast_model.h5')

def main():
    st.title("Retail Demand Forecasting System")
    st.write("Predict future demand for seasonal retail products")
    
    # Load data and ensure proper datetime format
    df = load_data()
    df['date'] = pd.to_datetime(df['date'])  # Ensure datetime format
    
    try:
        model = load_forecast_model()
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        st.stop()
    
    # Sidebar controls
    st.sidebar.header("Forecast Parameters")
    product_id = st.sidebar.selectbox("Select Product", df['product_id'].unique())
    store_id = st.sidebar.selectbox("Select Store", df['store_id'].unique())
    forecast_days = st.sidebar.slider("Forecast Days", 7, 90, 30)
    
    # Filter data and sort by date
    product_store_data = df[(df['product_id'] == product_id) & 
                          (df['store_id'] == store_id)].copy()
    product_store_data = product_store_data.sort_values('date')
    
    # Show historical data
    st.subheader(f"Historical Demand for {product_id} at {store_id}")
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(product_store_data['date'], product_store_data['demand'])
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax.set_xlabel("Date")
    ax.set_ylabel("Daily Demand")
    fig.autofmt_xdate()
    st.pyplot(fig)
    
    # Prepare data for forecasting
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(product_store_data['demand'].values.reshape(-1, 1))
    
    # Make predictions
    look_back = 30
    last_sequence = scaled_data[-look_back:]
    predictions = []
    
    for _ in range(forecast_days):
        X = last_sequence.reshape(1, look_back, 1)
        pred = model.predict(X, verbose=0)  # Added verbose=0 to suppress output
        predictions.append(pred[0, 0])
        last_sequence = np.append(last_sequence[1:], pred)
    
    predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
    
    # Create forecast dates properly
    last_date = product_store_data['date'].iloc[-1]
    forecast_dates = pd.date_range(
        start=last_date + pd.Timedelta(days=1),
        periods=forecast_days
    )
    
    # Show forecast with proper date handling
    st.subheader(f"{forecast_days}-Day Demand Forecast")
    fig2, ax2 = plt.subplots(figsize=(10, 4))
    
    # Plot history
    ax2.plot(
        product_store_data['date'][-60:], 
        product_store_data['demand'][-60:], 
        label='History'
    )
    
    # Plot forecast
    ax2.plot(
        forecast_dates,
        predictions,
        label='Forecast',
        color='orange'
    )
    
    # Format x-axis
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax2.set_xlabel("Date")
    ax2.set_ylabel("Daily Demand")
    fig2.autofmt_xdate()
    ax2.legend()
    st.pyplot(fig2)
    
    # Show forecast table
    forecast_df = pd.DataFrame({
        'Date': forecast_dates.strftime('%Y-%m-%d'),  # Format dates as strings
        'Forecasted Demand': predictions.flatten().round(2)  # Round to 2 decimals
    })
    st.dataframe(forecast_df)

if __name__ == "__main__":
    main()