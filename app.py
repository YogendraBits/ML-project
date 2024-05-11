import dash
import dash_bootstrap_components as dbc
from dash import dcc, html, Input, Output, State
import pandas as pd
import numpy as np
import joblib

# Load the saved model, label encoder, and imputer
model = joblib.load('hotel_booking_model.pkl')
label_encoder = joblib.load('label_encoder.pkl')
imputer = joblib.load('imputer.pkl')

# Initialize the Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

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

# Define layout of the app
app.layout = html.Div([
    html.H1("Hotel Cancellation Prediction", style={'textAlign': 'center', 'margin-bottom': '30px', 'font-family': 'Arial, sans-serif'}),
    html.Div(id='prediction-content', children=[
        html.Div([
            html.Label('Lead Time (days)', style={'font-weight': 'bold', 'margin-right': '10px'}),
            dcc.Input(id='lead-time', type='number', placeholder="Lead time (days)", style={'margin-bottom': '15px', 'width': '100%', 'padding': '10px', 'border-radius': '5px'}),
        ], style={'margin-bottom': '20px'}),
        html.Div([
            html.Label('Length of Stay (nights)', style={'font-weight': 'bold', 'margin-right': '10px'}),
            dcc.Input(id='length-of-stay', type='number', placeholder="Length of stay (nights)", style={'margin-bottom': '15px', 'width': '100%', 'padding': '10px', 'border-radius': '5px'}),
        ], style={'margin-bottom': '20px'}),
        html.Div([
            html.Label('Number of Previous Cancellations', style={'font-weight': 'bold', 'margin-right': '10px'}),
            dcc.Input(id='previous-cancellations', type='number', placeholder="Number of previous cancellations", style={'margin-bottom': '15px', 'width': '100%', 'padding': '10px', 'border-radius': '5px'}),
        ], style={'margin-bottom': '20px'}),
        html.Div([
            html.Label('Temperature during Stay (Celsius)', style={'font-weight': 'bold', 'margin-right': '10px'}),
            dcc.Input(id='temperature', type='number', placeholder="Temperature during stay (Celsius)", style={'margin-bottom': '15px', 'width': '100%', 'padding': '10px', 'border-radius': '5px'}),
        ], style={'margin-bottom': '20px'}),
        html.Div([
            html.Label('Precipitation Intensity', style={'font-weight': 'bold', 'margin-right': '10px'}),
            dcc.Dropdown(
                id='precipitation',
                options=[
                    {'label': 'None', 'value': 'none'},
                    {'label': 'Light', 'value': 'light'},
                    {'label': 'Moderate', 'value': 'moderate'},
                    {'label': 'Heavy', 'value': 'heavy'}
                ],
                placeholder="Precipitation intensity",
                style={'margin-bottom': '15px', 'width': '100%', 'padding': '10px', 'border-radius': '5px'}
            ),
        ], style={'margin-bottom': '20px'}),
        html.Div([
            html.Label('Room Type', style={'font-weight': 'bold', 'margin-right': '10px'}),
            dcc.Dropdown(
                id='room-type',
                options=[
                    {'label': 'Single', 'value': 'single'},
                    {'label': 'Double', 'value': 'double'},
                    {'label': 'Suite', 'value': 'suite'}
                ],
                placeholder="Room type",
                style={'margin-bottom': '15px', 'width': '100%', 'padding': '10px', 'border-radius': '5px'}
            ),
        ], style={'margin-bottom': '20px'}),
        html.Div([
            html.Label('Total Cost', style={'font-weight': 'bold', 'margin-right': '10px'}),
            dcc.Input(id='total-cost', type='number', placeholder="Total Cost", style={'margin-bottom': '15px', 'width': '100%', 'padding': '10px', 'border-radius': '5px'}),
        ], style={'margin-bottom': '20px'}),
        html.Div([
            html.Label('Competitor Price', style={'font-weight': 'bold', 'margin-right': '10px'}),
            dcc.Input(id='competitor-price', type='number', placeholder="Competitor price", style={'margin-bottom': '15px', 'width': '100%', 'padding': '10px', 'border-radius': '5px'}),
        ], style={'margin-bottom': '20px'}),

        html.Div([
            html.Button('Predict', id='submit-val', n_clicks=0, style={'margin-right': '10px', 'margin-bottom': '20px', 'width': '48%', 'padding': '15px', 'border-radius': '5px', 'background-color': '#4CAF50', 'color': 'white', 'font-size': '1.2em', 'cursor': 'pointer'}),
            html.Button('Reset', id='reset-val', n_clicks=0, style={'margin-bottom': '20px', 'width': '48%', 'padding': '15px', 'border-radius': '5px', 'background-color': '#f44336', 'color': 'white', 'font-size': '1.2em', 'cursor': 'pointer'}),
        ], style={'display': 'flex', 'justify-content': 'space-between'}),
        html.Div(id='output-modal'),
        html.Div(id='output-state', style={'display': 'none'})  # Hidden div to store prediction result
    ], style={'max-width': '500px', 'margin': 'auto', 'padding': '20px', 'background-color': '#f9f9f9', 'border-radius': '10px', 'box-shadow': '0px 0px 10px 2px rgba(0,0,0,0.1)'})
])

# Define callback to predict based on user input
@app.callback(
    Output('output-modal', 'children'),
    [Input('submit-val', 'n_clicks')],
    [State('lead-time', 'value'),
     State('length-of-stay', 'value'),
     State('previous-cancellations', 'value'),
     State('temperature', 'value'),
     State('precipitation', 'value'),
     State('room-type', 'value'),
     State('total-cost', 'value'),
     State('competitor-price', 'value')]
)
def update_output(submit_click, lead_time, length_of_stay, previous_cancellations, temperature, precipitation, room_type, total_cost, competitor_price):
    if submit_click > 0:
        if any(v is None for v in [lead_time, length_of_stay, previous_cancellations, temperature, precipitation, room_type, total_cost, competitor_price]):
            return dbc.Modal([
                dbc.ModalHeader("Error"),
                dbc.ModalBody("Please fill in all fields before predicting."),
                dbc.ModalFooter(
                    dbc.Button("Close", id="close-modal", className="ml-auto")
                ),
            ], id="modal", is_open=True)

        user_input = {
            'lead_time': lead_time,
            'length_of_stay': length_of_stay,
            'previous_cancellations': previous_cancellations,
            'temperature': temperature,
            'precipitation': precipitation,
            'room_type': room_type,
            'total_cost': total_cost,
            'competitor_price': competitor_price
        }

        user_data = pd.DataFrame([user_input])

        preprocessed_input = preprocess_input(user_data)
        predictions = model.predict(preprocessed_input)
        decoded_predictions = label_encoder.inverse_transform(predictions)

        return dbc.Modal([
            dbc.ModalHeader("Prediction Result"),
            dbc.ModalBody(f"Predicted reservation status: {decoded_predictions[0]}"),
            dbc.ModalFooter(
                dbc.Button("Close", id="close-modal", className="ml-auto")
            ),
        ], id="modal", is_open=True)
    else:
        return None

@app.callback(
    Output("modal", "is_open"),
    [Input("close-modal", "n_clicks")],
    [State("modal", "is_open")],
)
def toggle_modal(n1, is_open):
    if n1:
        return not is_open
    return is_open

# Callback to reset the form fields
@app.callback(
    [Output('lead-time', 'value'),
     Output('length-of-stay', 'value'),
     Output('previous-cancellations', 'value'),
     Output('temperature', 'value'),
     Output('precipitation', 'value'),
     Output('room-type', 'value'),
     Output('total-cost', 'value'),
     Output('competitor-price', 'value')],
    [Input('reset-val', 'n_clicks')]
)
def reset_form_fields(reset_click):
    if reset_click > 0:
        return None, None, None, None, None, None, None, None
    else:
        raise dash.exceptions.PreventUpdate

# Run the app
if __name__ == '__main__':
    app.run_server(debug=False)
