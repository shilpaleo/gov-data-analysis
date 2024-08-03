
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_percentage_error

# Load the dataset
file_path = '../data/question2/AverageDailyPolyclinicAttendancesforSelectedDiseases.csv'
df = pd.read_csv(file_path)

# Convert epi_week to datetime format
df['epi_week'] = pd.to_datetime(df['epi_week'] + '-1', format='%Y-W%W-%w')

# Create a new column for "average_cases" for modeling
df['average_cases'] = df['no._of_cases'] / 7

# Setting date column as index for time series
df.set_index('epi_week', inplace=True)
df.sort_index(inplace=True)

# Specify the disease of interest
disease_of_interest = 'Acute Diarrhoea'

# Filter for the disease of interest
df = df[df['disease'] == disease_of_interest]

# Load the SARIMA model for the specific disease
model = joblib.load(f'sarima_model_{disease_of_interest}.pkl')

# Streamlit app
st.title(f"Polyclinic Attendance Forecasting for {disease_of_interest}")

# Get the data for the disease of interest
disease_data = df['average_cases']

# Split data into training and testing sets based on week 26 of 2022
train_end_date = pd.to_datetime('2022-07-03')  # 2022-W26
train_data = disease_data[disease_data.index <= train_end_date]
test_data = disease_data[disease_data.index > train_end_date]

# Make predictions using the loaded model
forecast = model.predict(n_periods=len(test_data))

# Calculate MAPE
mape = mean_absolute_percentage_error(test_data, forecast)

# Plot the forecast vs actual
fig, ax = plt.subplots(figsize=(15, 8))
ax.plot(train_data.index, train_data, color='blue', label='Training')
ax.plot(test_data.index, test_data, color='orange', label='Actual', alpha=0.7)
ax.plot(test_data.index, forecast, color='green', label='Forecast', alpha=0.9, ls='--')
ax.set_title(f'{disease_of_interest} Forecast\nMAPE: {mape:.2f}', fontsize=16)
ax.set_xlabel('Week')
ax.set_ylabel('Avg Number of Cases')
ax.legend()
st.pyplot(fig)
