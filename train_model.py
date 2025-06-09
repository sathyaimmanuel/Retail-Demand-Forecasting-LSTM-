import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

def prepare_data(df, product_id, store_id, look_back=30):
    # Filter for specific product and store
    product_store_data = df[(df['product_id'] == product_id) & 
                           (df['store_id'] == store_id)].sort_values('date')
    
    # Create time series data
    series = product_store_data['demand'].values.reshape(-1, 1)
    
    # Normalize data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(series)
    
    # Create sequences
    X, y = [], []
    for i in range(look_back, len(scaled_data)):
        X.append(scaled_data[i-look_back:i, 0])
        y.append(scaled_data[i, 0])
    
    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    
    return X, y, scaler

def build_lstm_model(look_back):
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(look_back, 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Example usage
df = pd.read_csv('retail_demand_data.csv')
X, y, scaler = prepare_data(df, 'P001', 'S001')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

model = build_lstm_model(look_back=30)
model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))
model.save('demand_forecast_model.h5') 