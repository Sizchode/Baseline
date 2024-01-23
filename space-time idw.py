import os
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
import io
import pandas as pd
from datetime import datetime
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from math import radians, sin, cos, sqrt, atan2
import csv

data_path = "/Users/liuzhenke/PycharmProjects/baseline st prediction/point_processed"

#print(f"Checking path: {data_path}")

csv_file_path = 'space-time-idw.csv'
train_data = []
# Iterate through each date in June 2019
for day in range(1, 31):
    # Format the date string
    date_str = f"2019-06-{day:02d}"

    # Construct the file path for train.csv
    file_path = os.path.join(data_path, f"{date_str}/train.csv")

    # Check if the file exists
    if os.path.exists(file_path):
        # Read the CSV file into a DataFrame
        df = pd.read_csv(file_path)

        # Extract the required columns (date, temperature, latitude, and longitude)
        df_extracted = df[['temperture', 'lat', 'lon']].copy()

        # Add the date column
        df_extracted['date'] = date_str

        # Reorder the columns
        df_extracted = df_extracted[['date', 'temperture', 'lat', 'lon']]

        # Append the DataFrame to the list
        train_data.append(df_extracted)
    else:
        print(f"File not found for date {date_str}")

train_data = pd.concat(train_data, ignore_index=True)

    # Print the resulting DataFrame
print("Train data")
print(train_data)

test_data = []

# Iterate through each date in June 2019
for day in range(1, 31):
    # Format the date string
    date_str = f"2019-06-{day:02d}"

    # Construct the file path for train.csv
    file_path = os.path.join(data_path, f"{date_str}/test.csv")

    # Check if the file exists
    if os.path.exists(file_path):
        # Read the CSV file into a DataFrame
        df = pd.read_csv(file_path)

        # Extract the required columns (date, temperature, latitude, and longitude)
        df_extracted = df[['temperture', 'lat', 'lon']].copy()

        # Add the date column
        df_extracted['date'] = date_str

        # Reorder the columns
        df_extracted = df_extracted[['date', 'temperture', 'lat', 'lon']]

        # Append the DataFrame to the list
        test_data.append(df_extracted)
    else:
        print(f"File not found for date {date_str}")

test_data = pd.concat(test_data, ignore_index=True)

    # Print the resulting DataFrame
print("\n Test data")
print(test_data)

train_data.to_csv("train_data.csv", index=False)
test_data.to_csv("test_data.csv",index=False)


def calculate_space_time_distances(target_date, target_sensor, data, alpha):
    distances = []
    for dataframe in data:
        for sensor_data in dataframe.itertuples(index=False):
            spatial_dist = np.sqrt((target_sensor['lat'].iloc[0] - sensor_data.lat) ** 2 + (
                        target_sensor['lon'].iloc[0] - sensor_data.lon) ** 2)
            temporal_dist = np.abs(pd.to_datetime(target_date) - pd.to_datetime(sensor_data.date)).days
            combined_distance = np.sqrt(
                (alpha * spatial_dist) ** 2 + ((1 - alpha) * temporal_dist) ** 2)  # Combine spatial and temporal distances with alpha
            distances.append((sensor_data, combined_distance))
    return distances

def idw_prediction(target_sensor, nearest_sensors, alpha, power=2):
    weights = []
    values = []

    for sensor, distance in nearest_sensors:
        # Calculating the inverse distance weight
        weight = 1 / (distance ** power)
        weights.append(weight)

        # Extracting the value from the sensor
        value = sensor.temperture  # Assuming temperature is the value of interest
        values.append(value)

    # Avoid division by zero
    if sum(weights) == 0:
        return np.nan

    # Calculate the IDW prediction with weighted alpha
    weighted_alpha = np.array(weights) * alpha
    weighted_distance_sum = np.sum(weighted_alpha)
    idw_prediction = np.sum(np.array(values) * weighted_alpha) / weighted_distance_sum

    return idw_prediction

def select_nearest_sensors(k, distances):
    sorted_distances = sorted(distances, key=lambda x: x[1])  # Sort by combined distance
    nearest_sensors = sorted_distances[:k]  # Selecting the top k nearest sensors
    return nearest_sensors

alpha = 0.5  # You can adjust the alpha parameter

predictions = []

for test_row in test_data.itertuples(index=False):
    target_date = test_row.date
    target_sensor = pd.DataFrame([[test_row.date, test_row.temperture, test_row.lat, test_row.lon]],
                                 columns=['date', 'temperture', 'lat', 'lon'])

    # Calculate distances with alpha
    distances = calculate_space_time_distances(target_date, target_sensor, [train_data], alpha)

    # Select k nearest sensors
    k = 300
    nearest_sensors = select_nearest_sensors(k, distances)

    # Make IDW prediction with alpha
    idw_result = idw_prediction(target_sensor, nearest_sensors, alpha)
    predictions.append((target_date, test_row.lat, test_row.lon,test_row.temperture, idw_result))

# Display the predictions
for date, lat, lon,temperture, prediction in predictions:
    print(f"IDW Prediction for Date: {date}, Lat: {lat}, Lon: {lon}, actual temp: {temperture} - prediction {prediction}")

print(predictions)


# Writing data to CSV file
with open(csv_file_path, 'w', newline='') as csvfile:
    # Create a CSV writer object
    csv_writer = csv.writer(csvfile)

    # Write the header
    csv_writer.writerow(['Date', 'Lat', 'Lon', 'Actual Temp', 'Prediction'])

    # Write the data
    csv_writer.writerows(predictions)

print(f'Data has been successfully stored in {csv_file_path}')

df = pd.read_csv("space-time-idw.csv")

# Extract Actual Temp and Prediction columns
actual_temps = df['Actual Temp']
predictions = df['Prediction']

actual_temps = df['Actual Temp']
predictions = df['Prediction']

mae = mean_absolute_error(actual_temps, predictions)

# Calculate RMSE
rmse = np.sqrt(mean_squared_error(actual_temps, predictions))

# Calculate R²
r2 = r2_score(actual_temps, predictions)

# Add new columns to the DataFrame
df['MAE'] = mae
df['RMSE'] = rmse
df['R²'] = r2

# Save the updated DataFrame to CSV
df.to_csv("space-time-idw.csv", index=False)

# Print results
print("MAE:", mae)
print("RMSE:", rmse)
print("R²:", r2)

