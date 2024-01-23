import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
import io
import pandas as pd
from datetime import datetime
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from math import radians, sin, cos, sqrt, atan2
import os


data_path = "/Users/liuzhenke/PycharmProjects/baseline st prediction/point_processed"

#print(f"Checking path: {data_path}")

csv_file_path = 'space-time-nn.csv'

from scipy.spatial.distance import cdist

# Sample train_data and test_data
# Make sure to replace this with your actual train_data and test_data
train_data = pd.read_csv('train_data.csv')
test_data = pd.read_csv('test_data.csv')


# Function to find space-time nearest neighbor
def find_space_time_nearest_neighbor(test_row, train_data):
    # Extract latitude, longitude, and date from the test row
    test_lat = test_row['lat']
    test_lon = test_row['lon']
    test_date = pd.to_datetime(test_row['date']).timestamp()

    # Calculate distances in space and time
    test_point = np.array([[test_lat, test_lon, test_date]])
    train_points = train_data[['lat', 'lon', 'date']].apply(
        lambda x: [x['lat'], x['lon'], pd.to_datetime(x['date']).timestamp()], axis=1).tolist()
    distances = cdist(test_point, train_points, metric='euclidean')[0]

    # Find index of the nearest neighbor
    nearest_neighbor_index = np.argmin(distances)

    # Get temperature of the nearest neighbor
    predicted_temperature = train_data.iloc[nearest_neighbor_index]['temperture']

    return predicted_temperature


# Apply the function to each row in the test data
test_data['predicted_temperature'] = test_data.apply(lambda row: find_space_time_nearest_neighbor(row, train_data),
                                                     axis=1)
print(test_data)
test_data[['date', 'lat', 'lon', 'temperture', 'predicted_temperature']].to_csv("space-time-nn.csv")
df = pd.read_csv("space-time-nn.csv")
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
df.to_csv("space-time-nn.csv", index=False)

print("MAE:", mae)
print("RMSE:", rmse)
print("R²:", r2)

# Print the resulting DataFrame with predictions
#print(test_data[['date', 'lat', 'lon', 'predicted_temperature']])

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



import numpy as np
from scipy.spatial.distance import cdist

def space_time_nearest_neighbor_interpolation(data, target_sensor_id):
    # Extract data for the target sensor
    target_sensor = fetch_target_sensor(folder_path, target_sensor_id)

    # Convert date columns to datetime type
    data['date'] = pd.to_datetime(data['date'])
    target_sensor['date'] = pd.to_datetime(target_sensor['date'])

    # Convert 'lat' and 'lon' columns to numeric
    data['lat'] = pd.to_numeric(data['lat'], errors='coerce')
    data['lon'] = pd.to_numeric(data['lon'], errors='coerce')
    target_sensor['lat'] = pd.to_numeric(target_sensor['lat'], errors='coerce')
    target_sensor['lon'] = pd.to_numeric(target_sensor['lon'], errors='coerce')

    # Convert 'date' column to numeric (assuming it represents a timestamp)
    data['date'] = pd.to_numeric(data['date'], errors='coerce')
    target_sensor['date'] = pd.to_numeric(target_sensor['date'], errors='coerce')

    # Combine latitude, longitude, and date into 3D spatial coordinates
    coords = data[['lat', 'lon', 'date']].values
    values = data['temperture'].values

    # Convert the query point's date to numeric
    query_point = np.array([[target_sensor['lat'].values[0], target_sensor['lon'].values[0], target_sensor['date'].values[0]]])

    # Calculate distances
    distances = cdist(coords, query_point, metric='euclidean')

    # Find the index of the nearest neighbor
    nearest_neighbor_index = np.argmin(distances)

    # Return the temperature of the nearest neighbor
    predicted_temperature = values[nearest_neighbor_index]

    return predicted_temperature

prediction_result = []
# 例子：预测传感器0的温度
for i in range(92):
    sensor_id_to_predict = i
    data = prepare_data(sensor_id_to_predict, folder_path)

    # 将2019-12-26的数据从目标传感器中排除
    target_sensor = fetch_target_sensor(folder_path, sensor_id_to_predict)
    data = pd.concat(data, ignore_index=True)
    # 进行插值预测
    predicted_temperature = space_time_nearest_neighbor_interpolation(data, sensor_id_to_predict)

    # 输出预测结果
    print(f'Predicted temperature for sensor-{i}-on 2019-12-26:', predicted_temperature)
    prediction_result.append(predicted_temperature)

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
df.to_csv('space-time nn.csv', index=False)
'''