import torch
import torch.nn as nn 
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from sklearn.preprocessing import MinMaxScaler

app = Flask(__name__)

class TransformerModel(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_layers, seq_length=24, prediction_length=24):
        super(TransformerModel, self).__init__()
        self.input_layer = nn.Linear(input_dim, d_model)
        self.positional_encoding = nn.Parameter(torch.randn(seq_length, d_model))
        self.transformer = nn.Transformer(d_model=d_model, nhead=nhead, num_encoder_layers=num_layers)
        self.output_layer = nn.Linear(seq_length * d_model, prediction_length)

    def forward(self, x):
        x = self.input_layer(x) + self.positional_encoding  # Add positional encoding
        x = x.permute(1, 0, 2)  # Transformer expects (seq_length, batch_size, d_model)
        x = self.transformer(x, x)  # Encoder-only Transformer
        x = x.permute(1, 0, 2).reshape(x.size(1), -1)  # Flatten for the output layer
        x = self.output_layer(x)  # Predict next 24 AQI values
        return x

# Model parameters
input_dim = 16
d_model = 64
nhead = 4
num_layers = 2

# Load the saved model and scalers
model = TransformerModel(input_dim, d_model, nhead, num_layers)  # Define the model first
model.load_state_dict(torch.load("aqi_transformer_model.pkl"))  # Load the trained weights

# Initialize the MinMaxScaler
scaler_features = MinMaxScaler()
scaler_target = MinMaxScaler()

# Ensure the model is in evaluation mode
model.eval()

# Helper function to preprocess the incoming data
def preprocess_data(data):
    df = pd.DataFrame(data)

    # Convert 'Time' column to datetime format
    df['Time'] = pd.to_datetime(df['Time'])

    # Sort by time
    df = df.sort_values(by='Time')

    # Extract time-related features
    df['hour'] = df['Time'].dt.hour

    # Replace AQI values of 0 with NaN for interpolation
    df['aqi'] = df['aqi'].replace(0, np.nan)

    # Interpolate missing AQI values (linear interpolation)
    df['aqi'] = df['aqi'].interpolate(method='linear', limit_direction='both')

    # Add 1-hour lag features
    df['Clouds %_lag1'] = df['Clouds %'].shift(1)
    df['Humidity_lag1'] = df['Humidity'].shift(1)
    df['Rain_lag1'] = df['Rain'].shift(1)
    df['Temperature_lag1'] = df['Temperature'].shift(1)
    df['Wind Speed_lag1'] = df['Wind Speed'].shift(1)
    df['aqi_lag1'] = df['aqi'].shift(1)

    # Add momentum features
    df['aqi_momentum'] = df['aqi'] - df['aqi_lag1']

    # Step 1: Create a flag for rain events
    df['Rain_event'] = (df['Rain'] > 0).astype(int)

    # Step 2: Calculate AQI 2 hours before and after each reading
    df['AQI_2hr_before'] = df['aqi'].shift(2)
    df['AQI_2hr_after'] = df['aqi'].shift(-2)

    # Step 3: Calculate AQI reduction due to rain
    df['Rain_aqi_reduction'] = df.apply(
        lambda row: row['AQI_2hr_before'] - row['AQI_2hr_after']
        if row['Rain_event'] == 1 else np.nan,
        axis=1
    )

    # Calculate cumulative hours since the last rain
    df['Hours_since_rain'] = df['Rain_event'].cumsum()
    df['Hours_since_rain'] = df.groupby('Hours_since_rain').cumcount()

    # Fill NaN values
    df['Rain_aqi_reduction'] = df['Rain_aqi_reduction'].fillna(0)
    df = df.fillna(0)

    # Select only the required features
    features = [
        'Clouds %', 'Humidity', 'Rain', 'Temperature', 'Wind Speed',
        'Clouds %_lag1', 'Humidity_lag1', 'Rain_lag1',
        'Temperature_lag1', 'Wind Speed_lag1', 'aqi_lag1', 'hour',
        'aqi_momentum', 'Rain_event', 'Rain_aqi_reduction', 'Hours_since_rain'
    ]

    # Return the preprocessed DataFrame
    return df[features]

@app.route("/")
def home():
    return "Welcome to the AQI Prediction API! Use the /predict endpoint with a POST request to get predictions."

# Endpoint to predict AQI for the next 24 hours
@app.route("/predict", methods=["POST"])
def predict():
    # Get 24-hour values from the incoming JSON request
    data = request.get_json()

    # Preprocess the data
    preprocessed_data = preprocess_data(data)

    # Scale the features using the MinMaxScaler directly
    scaled_features = scaler_features.fit_transform(preprocessed_data)

    # Convert to torch tensor for prediction
    scaled_features_tensor = torch.tensor(scaled_features, dtype=torch.float32).unsqueeze(0)

    # Make prediction
    with torch.no_grad():
        prediction = model(scaled_features_tensor)

    # Reshape prediction to (24,) and inverse transform
    prediction_reshaped = prediction.reshape(-1)  # Flatten the prediction output
    predicted_aqi = scaler_target.inverse_transform(prediction_reshaped.numpy().reshape(1, -1))

    # Send the prediction as a response
    return jsonify(predicted_aqi.tolist())

if __name__ == "__main__":
    app.run(debug=True)
