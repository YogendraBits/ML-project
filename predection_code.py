import pandas as pd
import numpy as np
import joblib

# Load the saved model, label encoder, and imputer
model = joblib.load('hotel_booking_model.pkl')
label_encoder = joblib.load('label_encoder.pkl')
imputer = joblib.load('imputer.pkl')

# Define function to preprocess input data
def preprocess_input(data):
    # Preprocess the data similar to training data
    features = []

    # Additional features
    data['booking_flexibility_score'] = data['lead_time'] * data['length_of_stay'] * (1 / (data['previous_cancellations'] + 1))
    data['booking_stability_indicator'] = data['lead_time'] * data['length_of_stay'] * (1 / (data['room_type'].map({'single': 1, 'double': 2, 'suite': 3}) + 1))
    data['weather_impact_index'] = data['temperature'] * (data['precipitation'].map({'none': 0, 'light': 0.5, 'moderate': 1, 'heavy': 2}) + 1)
    data['competitive_pricing_gap'] = data['competitor_price'] - data['total_cost']

    features.extend(['booking_flexibility_score',
                    'booking_stability_indicator', 'weather_impact_index', 'competitive_pricing_gap'])

    # Encode categorical features
    data_encoded = pd.get_dummies(data[features])

    # Impute missing values
    data_imputed = imputer.transform(data_encoded)

    return data_imputed

def get_user_input():
    user_input = {}

    print("Please enter the following information for the hotel booking:")
    user_input['lead_time'] = int(input("Lead time (days): "))
    user_input['length_of_stay'] = int(input("Length of stay (nights): "))
    user_input['previous_cancellations'] = int(input("Number of previous cancellations: "))
    user_input['temperature'] = float(input("Temperature during stay (Celsius): "))
    user_input['precipitation'] = input("Precipitation intensity ('none', 'light', 'moderate', 'heavy'): ")
    user_input['room_type'] = input("Room type (e.g., 'single', 'double', 'suite'): ")
    user_input['total_cost'] = float(input("Total Cost: "))
    user_input['competitor_price'] = float(input("Competitor price: "))

    return pd.DataFrame([user_input])

# Get user input for a single booking
user_data = get_user_input()

# Preprocess user input
preprocessed_input = preprocess_input(user_data)

# Make predictions
predictions = model.predict(preprocessed_input)

# Decode predictions
decoded_predictions = label_encoder.inverse_transform(predictions)

# Print predictions
print("Predicted reservation status:", decoded_predictions[0])
