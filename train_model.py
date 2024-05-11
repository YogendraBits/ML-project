from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
import joblib

# Define a function to get hyperparameters based on the model type
def get_param_grid(model):
    """
    Defines hyperparameter grids for different models.

    Args:
        model: The machine learning model object.

    Returns:
        A dictionary containing hyperparameter grids specific to the model.
    """
    param_grid = {}
    if isinstance(model, RandomForestClassifier):
        # Common hyperparameters for Random Forest
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
    elif isinstance(model, XGBClassifier):
        # Common hyperparameters for XGBoost
        param_grid = {
            'n_estimators': [100, 200, 300],
            'learning_rate': [0.05, 0.1, 0.3],
            'max_depth': [3, 5, 8],
            'min_child_weight': [1, 3, 5]
        }
    else:
        # Raise an error or handle models not explicitly defined
        raise ValueError(f"Hyperparameter grid not defined for model type {type(model)}")
    return param_grid

# Load the extended dataset (add try-except for potential file errors)
try:
    df = pd.read_csv('hotel_booking_data_extended.csv')
except FileNotFoundError:
    print("Error: File 'hotel_booking_data_extended.csv' not found!")
    exit()

# Define features and target variable
features = []

# Additional features
df_copy = df.copy()  # Create a copy of the DataFrame for feature engineering
df_copy['booking_flexibility_score'] = df_copy['lead_time'] * df_copy['length_of_stay'] * (1 / (df_copy['previous_cancellations'] + 1))
df_copy['booking_stability_indicator'] = df_copy['lead_time'] * df_copy['length_of_stay'] * (1 / (df_copy['room_type'].map({'single': 1, 'double': 2, 'suite': 3}) + 1))
df_copy['weather_impact_index'] = df_copy['temperature'] * (df_copy['precipitation'].map({'none': 0, 'light': 0.5, 'moderate': 1, 'heavy': 2}) + 1)
df_copy['competitive_pricing_gap'] = df_copy['competitor_price'] - df_copy['total_cost']

features.extend(['booking_flexibility_score','booking_stability_indicator', 'weather_impact_index', 'competitive_pricing_gap'])

# Encode target variable
label_encoder = LabelEncoder()
df_copy['reservation_status_encoded'] = label_encoder.fit_transform(df_copy['reservation_status'])

# Save LabelEncoder object
joblib.dump(label_encoder, 'label_encoder.pkl')

# Target variable
target = 'reservation_status_encoded'  # Use encoded target variable

# Impute missing values
imputer = SimpleImputer(strategy='median')
df_encoded = pd.get_dummies(df_copy[features])
X_imputed = imputer.fit_transform(df_encoded)

# Save SimpleImputer object
joblib.dump(imputer, 'imputer.pkl')

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_imputed, df_copy[target], test_size=0.2, random_state=42)

# Define models to compare
models = [
    ('Random Forest', RandomForestClassifier(random_state=42)),
    ('XGBoost', XGBClassifier(random_state=42))
]

# Feature Importance Analysis (after encoding)
for name, model in models:
    model.fit(X_train, y_train)
    print(f"\nFeature Importance for {name}:")
    print(model.feature_importances_)  # Analyze importances for each model

# Hyperparameter tuning using GridSearchCV
best_model = None
best_report = None
for name, model in models:
    try:  # Catch potential GridSearchCV errors
        grid_search = GridSearchCV(estimator=model, param_grid=get_param_grid(model), cv=5, n_jobs=-1, verbose=2)
        grid_search.fit(X_train, y_train)
        if best_model is None or grid_search.best_score_ > best_report:
            best_model = grid_search.best_estimator_
            best_report = grid_search.best_score_
            print(f"\nBest Parameters for {name}:", grid_search.best_params_)
    except Exception as e:
        print(f"\nError during GridSearchCV for {name}: {e}")

# Evaluate and save the best model
if best_model is not None:
    y_pred = best_model.predict(X_test)
    print("\nClassification Report for Best Model:")
    print(classification_report(y_test, y_pred))
    joblib.dump(best_model, 'hotel_booking_model.pkl')
else:
    print("Error: No model found due to potential GridSearchCV errors.")
