import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
import io



#Data Process Functions
def extract_sensor_data(sensor_id, file_names, folder_path):
    sensor_data = []
    for file_name in file_names:
        file_path = f'{folder_path}/{file_name}'
        data = pd.read_csv(file_path)
        sensor_data.append({
            'date': file_name[:-4],  # Extract date from the file name
            'lat': data.iloc[sensor_id]['lat'],
            'lon': data.iloc[sensor_id]['lon'],
            'temperature': data.iloc[sensor_id]['temperture']
        })
    return sensor_data

def print_sensor_data(sensor_data, sensor_id):
    for data in sensor_data:
        print(f"Date: {data['date']}")
        print(f"Sensor {sensor_id} - Lat: {data['lat']}, Lon: {data['lon']}, Temperature: {data['temperature']}")
        print()

def convert_to_dataframe(sensor_data):
    return pd.DataFrame(sensor_data)

def idw_prediction(data, target_lat, target_lon):
    # Convert coordinates to radians
    data[['lat', 'lon']] = np.radians(data[['lat', 'lon']])
    target_lat, target_lon = np.radians(target_lat), np.radians(target_lon)

    distances = cdist(data[['lat', 'lon']].values, [(target_lat, target_lon)])
    distances[distances == 0] = 0.0001  # To avoid division by zero, replace zeros with a small value
    weights = 1 / distances
    weights /= np.sum(1 / distances, axis=0)  # Normalize weights

    # Print information for analysis
    print("Distances:", distances)
    print("Weights:", weights)

    predicted_temperature = (weights * data['temperature'].values[:, np.newaxis]).sum(axis=0)

    return predicted_temperature


# Spatial-Temporal IDW Implementation and Experiment
file_list = [f'2019-01-{str(day).zfill(2)}.csv' for day in range(1, 31)]

# Create an empty list to store DataFrames for each sensor
sensor_data_df = []

for file_name in file_list:
    # Extract data for all sensors from the same date
    sensor_data = [extract_sensor_data(sensor_id, [file_name], '/Users/liuzhenke/PycharmProjects/baseline st prediction/drive-download-20231228T142352Z-001/point') for sensor_id in range(92)]
    sensor_df = pd.concat([convert_to_dataframe(data) for data in sensor_data])
    sensor_data_df.append(sensor_df)

# Combine all DataFrames into a single DataFrame
sensor_data_combined = pd.concat(sensor_data_df)

# Rest of the code for idw_prediction remains the same
target_lat = 33.755  # Example latitude of the target location
target_lon = -117.945728933223  # Example longitude of the target location

# Assuming sensor_data_combined contains columns 'lat', 'lon', 'temperature'
predicted_temp = idw_prediction(sensor_data_combined, target_lat, target_lon)
print(f"Predicted temperature at target location: {predicted_temp}")


