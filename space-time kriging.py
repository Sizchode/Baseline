import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
import io
import pandas as pd
from datetime import datetime
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from math import radians, sin, cos, sqrt, atan2
from pykrige.ok3d import OrdinaryKriging3D

train_data = pd.read_csv("train_data.csv")
test_data = pd.read_csv("test_data.csv")
print(train_data)
print(test_data)
# Extract relevant columns
train_data['date'] = pd.to_datetime(train_data['date']).astype(int)
test_data['date'] = pd.to_datetime(test_data['date']).astype(int)

# Extract relevant columns

train_coordinates = train_data[['lat', 'lon', 'date']].values
train_values = train_data['temperture'].values

# Perform Ordinary Kriging
ok3d = OrdinaryKriging3D(train_coordinates[:, 0], train_coordinates[:, 1], train_coordinates[:, 2],
                         train_values, variogram_model='linear', verbose=True)

# Iterate through test data and make predictions
predictions = []


'''
for test_row in test_data.itertuples(index=False):
    test_coordinates = np.array([test_row.lat, test_row.lon, test_row.date])
    prediction, _ = ok3d.execute('points', test_coordinates[0], test_coordinates[1], test_coordinates[2])
    prediction_value = prediction.data[0] if not prediction.mask[0] else None
    predictions.append(prediction_value)


for test_row in test_data.itertuples(index=False):
    target_date = test_row.date
    target_sensor = pd.DataFrame([[test_row.date, test_row.temperture, test_row.lat, test_row.lon]],
                                 columns=['date', 'temperture', 'lat', 'lon'])

    test_coordinates = np.array([test_row.lat, test_row.lon, test_row.date])
    prediction, _ = ok3d.execute('points', test_coordinates[0], test_coordinates[1], test_coordinates[2])
    prediction_value = prediction.data[0] if not prediction.mask[0] else None
    predictions.append((target_date, test_row.lat, test_row.lon,test_row.temperture, prediction_value))
'''

for test_row in test_data.itertuples(index=False):
    test_coordinates = np.array([test_row.lat, test_row.lon, test_row.date])
    prediction, _ = ok3d.execute('points', test_coordinates[0], test_coordinates[1], test_coordinates[2])

    # Extract the number from the prediction
    prediction_value = prediction.data[0] if not prediction.mask[0] else None
    predictions.append(prediction_value)

# Add predictions to the test_data DataFrame
test_data['predicted_temperature'] = predictions

# Convert timestamp to human-readable date
test_data['date'] = pd.to_datetime(test_data['date'])

# Save the test_data DataFrame with predictions to a new CSV file
test_data.to_csv("space-time kriging.csv", index=False)

df = pd.read_csv("space-time kriging.csv")
prediction_temp = df['predicted_temperature']
actual_temp = df['temperture']

mae = mean_absolute_error(actual_temp, prediction_temp)

# Calculate RMSE
rmse = np.sqrt(mean_squared_error(actual_temp, prediction_temp))

# Calculate R²
r2 = r2_score(actual_temp, prediction_temp)

df['MAE'] = mae
df['RMSE'] = rmse
df['R²'] = r2

df.to_csv("space-time kriging.csv", index=False)

# Print results
print("MAE:", mae)
print("RMSE:", rmse)
print("R²:", r2)

'''
folder_path = '/Users/liuzhenke/PycharmProjects/baseline st prediction/drive-download-20231228T142352Z-001/point'
prediction_result = []
ground_truth_idw = [
    43.4225, 29.88666667, 41.175, 44.18416667, 43.54458333, 43.17666667, 44.62375, 40.45625, 40.38791667,
    45.90875, 43.3475, 46.03666667, 44.54166667, 42.79625, 42.78416667, 46.06875, 45.99958333, 45.53416667,
    43.04208333, 43.47416667, 46.265, 45.00666667, 45.54916667, 45.07375, 45.91458333, 46.52791667, 46.41458333,
    45.52833333, 46.72, 45.87916667, 45.64416667, 45.43375, 46.34875, 45.83291667, 45.8325, 46.21708333,
    46.96125, 46.98583333, 47.13041667, 46.04458333, 46.02166667, 46.13625, 46.1125, 45.475, 46.18083333,
    45.38583333, 47.36958333, 46.995, 46.36125, 46.65708333, 46.83125, 47.28, 48.10666667, 46.84125, 46.91083333,
    48.3125, 48.36625, 49.00666667, 48.73916667, 49.49458333, 49.06291667, 50.52875, 49.44125, 48.80666667,
    48.46666667, 49.69041667, 48.27666667, 49.59, 48.27375, 47.17666667, 49.80375, 50.15833333, 49.12791667,
    48.82208333, 48.19208333, 46.59041667, 48.73541667, 49.27291667, 48.60166667, 48.87625, 49.1975, 50.05916667,
    49.2525, 49.31166667, 49.54291667, 49.49125, 49.69916667, 49.7025, 49.8225, 48.38541667, 49.7675, 49.44333333
]

def read_ground_truth(file_name, folder_path):
    columns = ['temperture']
    # Read the file into a DataFrame
    file_path = f'{folder_path}/{file_name}'
    data = pd.read_csv(file_path, usecols=columns)
    return data

def read_sensor_data(file_name, folder_path):
    # Assuming the file has columns: sensorid, temperature, lon, lat

    columns = ['sensorid', 'temperture', 'lon', 'lat']
    # Read the file into a DataFrame
    file_path = f'{folder_path}/{file_name}'
    data = pd.read_csv(file_path, usecols=columns)
    date_str = file_name.split('.')[0]  # Extract date from file name
    data['date'] = pd.to_datetime(date_str)  # Convert to datetime format

    return data

def convert_to_dataframe(sensor_data):
    return pd.DataFrame(sensor_data)

def read_sensor_data_except_ids(excluded_id, folder_path):
    file_list = [f'2019-12-{str(day).zfill(2)}.csv' for day in range(21, 26)]
    file_list2 = [f'2019-12-{str(day).zfill(2)}.csv' for day in range(27, 32)]
    date_file = '2019-12-26.csv'
    Data = []
    for file_name in file_list:
        sensor_data = read_sensor_data(file_name, folder_path)
        sensor_df = convert_to_dataframe(sensor_data)
        Data.append(sensor_df)

    sensor_data = read_sensor_data(date_file, folder_path)
    sensor_data = sensor_data[sensor_data['sensorid'] != excluded_id]
    sensor_df = convert_to_dataframe(sensor_data)
    Data.append(sensor_df)

    for file_name in file_list2:
        sensor_data = read_sensor_data(file_name, folder_path)
        sensor_df = convert_to_dataframe(sensor_data)
        Data.append(sensor_df)

    return Data

def prepare_data(excluded_id, folder_path):
    # Use the read_sensor_data_except_ids method to gather relevant data
    data = read_sensor_data_except_ids(excluded_id, folder_path)
    return data

def fetch_target_sensor(folder_path,id):
    target_sensor = read_sensor_data('2019-12-26.csv', folder_path)
    target_sensor = target_sensor[target_sensor['sensorid'] == id]
    return target_sensor


# Test cases use OK
from pykrige.ok3d import OrdinaryKriging3D

# Assuming you have functions prepare_data and fetch_target_sensor ready
for i in range(0,92):
    data = prepare_data(i, folder_path)  # Prepare the spatial-temporal data excluding sensor 0 on 2019-12-26
    target_sensor_data = fetch_target_sensor(folder_path, i)  # Fetch data for sensor 0 on 2019-12-26
    data = pd.concat(data, ignore_index=True)
    #print(data)
    #print(target_sensor_data)
    # Extract relevant columns
    x = data['lon'].values  # Longitude as X-coordinate
    y = data['lat'].values  # Latitude as Y-coordinate
    t = pd.to_datetime(data['date']).astype(int) / 10**9  # Convert date to UNIX timestamp for time dimension
    values = data['temperture'].values  # Temperature values for spatial-temporal Kriging

    # Fetching data for the target sensor
    x_target = target_sensor_data['lon'].values  # Longitude of target sensor
    y_target = target_sensor_data['lat'].values  # Latitude of target sensor
    t_target = pd.to_datetime(target_sensor_data['date']).astype(int) / 10**9  # Timestamp of target sensor

    # Initialize the Kriging model
    ok3d = OrdinaryKriging3D(x, y, t, values, variogram_model='linear', verbose=False)

    # Perform the prediction for the target sensor on 2019-12-26
    temperature_prediction, variance = ok3d.execute('points', x_target, y_target, t_target)

    print(f'Predicted temperature for sensor-{i}-on 2019-12-26:', temperature_prediction)
    prediction_result.append(temperature_prediction[0])


print(prediction_result)
df = pd.DataFrame(list(zip(prediction_result, ground_truth_idw)), columns=['Prediction', 'Ground_Truth'])
mae = mean_absolute_error(df['Ground_Truth'], df['Prediction'])

# Calculate RMSE
rmse = np.sqrt(mean_squared_error(df['Ground_Truth'], df['Prediction']))

# Calculate R-squared
r2 = r2_score(df['Ground_Truth'], df['Prediction'])

# Add these results to the DataFrame
df['MAE'] = mae
df['RMSE'] = rmse
df['R-squared'] = r2

# Save the DataFrame as a result file
df.to_csv('ST-Kriging.csv', index=False)
'''