import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
try:
    df = pd.read_csv('hotel_booking_data_extended.csv')
except FileNotFoundError:
    print("Error: File 'hotel_booking_data_extended.csv' not found!")
    exit()

# Feature engineering
df['booking_flexibility_score'] = df['lead_time'] * df['length_of_stay'] * (1 / (df['previous_cancellations'] + 1))
df['booking_stability_indicator'] = df['lead_time'] * df['length_of_stay'] * (1 / (df['room_type'].map({'single': 1, 'double': 2, 'suite': 3}) + 1))
df['weather_impact_index'] = df['temperature'] * (df['precipitation'].map({'none': 0, 'light': 0.5, 'moderate': 1, 'heavy': 2}) + 1)
df['competitive_pricing_gap'] = df['competitor_price'] - df['total_cost']

# Define features
features = ['booking_flexibility_score', 'booking_stability_indicator', 'weather_impact_index', 'competitive_pricing_gap']

# Calculate the correlation matrix
corr_matrix = df[features].corr()

# Plot the correlation matrix
plt.figure(figsize=(12, 10))  # Increase figure size
heatmap = sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Correlation Matrix of Features', fontsize=16)

# Rotate the x-axis and y-axis labels
heatmap.set_xticklabels(heatmap.get_xticklabels(), rotation=0, horizontalalignment='right', fontsize=8)
heatmap.set_yticklabels(heatmap.get_yticklabels(), rotation=0, fontsize=8)

# Adjust layout
plt.tight_layout()

# Show the plot
plt.show()
