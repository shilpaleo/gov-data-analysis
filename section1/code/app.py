
import os
import urllib.request

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from sklearn.metrics import mean_absolute_percentage_error

# Define GCS details
BUCKET_NAME = 'sarima_model_poly_timeseries'
MODEL_FILENAME = 'sarima_model_Acute_Diarrhoea.pkl'
MODEL_URL = f'https://storage.googleapis.com/{BUCKET_NAME}/{MODEL_FILENAME}'
LOCAL_MODEL_PATH = MODEL_FILENAME

# Function to download the model from public URL
def download_model():
    if not os.path.exists(LOCAL_MODEL_PATH):
        urllib.request.urlretrieve(MODEL_URL, LOCAL_MODEL_PATH)
        st.success('Model downloaded successfully!')
    else:
        st.info('Model already downloaded.')

# Streamlit app
st.title("Polyclinic Attendance Forecasting for Acute Diarrhoea")

# Download the model
download_model()
model = joblib.load(LOCAL_MODEL_PATH)

# Load the dataset (assuming it's local or publicly accessible)
@st.cache_data
def load_data():
    file_path = '../data/question2/AverageDailyPolyclinicAttendancesforSelectedDiseases.csv'
    df = pd.read_csv(file_path)
    df['epi_week'] = pd.to_datetime(df['epi_week'] + '-1', format='%Y-W%W-%w')
    df['average_cases'] = df['no._of_cases'] / 7
    df.set_index('epi_week', inplace=True)
    df.sort_index(inplace=True)
    return df

df = load_data()

# Filter the dataset for Acute Diarrhoea
disease_data = df[df['disease'] == 'Acute Diarrhoea']['average_cases']

# Split data into training and testing sets based on week 26 of 2022
train_end_date = pd.to_datetime('2022-07-03')  # 2022-W26
train_data = disease_data[disease_data.index <= train_end_date]
test_data = disease_data[disease_data.index > train_end_date]

# Forecast using the model
forecast = model.predict(n_periods=len(test_data))

# Calculate MAPE
mape = mean_absolute_percentage_error(test_data, forecast) *100

# Plot the forecast vs actual
fig, ax = plt.subplots(figsize=(15, 8))
ax.plot(train_data.index, train_data, color='blue', label='Training')
ax.plot(test_data.index, test_data, color='orange', label='Actual', alpha=0.7)
ax.plot(test_data.index, forecast, color='green', label='Forecast', alpha=0.9, ls='--')
ax.set_title(f'Acute Diarrhoea Forecast\nMAPE: {mape:.2f}', fontsize=16)
ax.set_xlabel('Week')
ax.set_ylabel('Number of Cases')
ax.legend()
st.pyplot(fig)
