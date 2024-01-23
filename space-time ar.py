import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
import io
import pandas as pd
from datetime import datetime
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from math import radians, sin, cos, sqrt, atan2
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
import os
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.api import VAR

train_data = pd.read_csv('train_data.csv')
test_data = pd.read_csv('test_data.csv')

csv_file_path = 'space-time-ar.csv'


# Convert date to datetime type
train_data['date'] = pd.to_datetime(train_data['date'], errors='coerce')
test_data['date'] = pd.to_datetime(test_data['date'], errors='coerce')

# Check if 'date' is in the columns
if 'date' in train_data.columns:
    # Set 'date' as the index
    train_data.set_index('date', inplace=True)
else:
    print("No 'date' column found in the DataFrame.")

# Train VAR model
model = VAR(train_data)
model_fit = model.fit()

# Forecast
predictions = model_fit.forecast(train_data.values, steps=len(test_data))

# Evaluate model
mae = mean_absolute_error(test_data['temperture'], predictions[:, 0])
rmse = np.sqrt(mean_squared_error(test_data['temperture'], predictions[:, 0]))
r2 = r2_score(test_data['temperture'], predictions[:, 0])
test_data['predict temp']=predictions[:, 0]
print(test_data)

print(f'Mean Absolute Error (MAE): {mae}')
print(f'Root Mean Squared Error (RMSE): {rmse}')
print(f'R-squared (R²): {r2}')
test_data.to_csv(csv_file_path)
df = pd.read_csv(csv_file_path)
df['MAE'] = mae
df['RMSE'] = rmse
df['R²'] = r2
df.to_csv(csv_file_path)