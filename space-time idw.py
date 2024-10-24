import os
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import argparse
import csv

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
        weight = 1 / (distance ** power)
        weights.append(weight)
        value = sensor.temperture
        values.append(value)

    if sum(weights) == 0:
        return np.nan

    weighted_alpha = np.array(weights) * alpha
    weighted_distance_sum = np.sum(weighted_alpha)
    idw_prediction = np.sum(np.array(values) * weighted_alpha) / weighted_distance_sum

    return idw_prediction

def select_nearest_sensors(k, distances):
    sorted_distances = sorted(distances, key=lambda x: x[1])
    nearest_sensors = sorted_distances[:k]
    return nearest_sensors

def main():
    # Argument parsing
    parser = argparse.ArgumentParser(description="Space-Time IDW Model")
    parser.add_argument("--train_data_path", type=str, required=True, help="Path to the training data CSV file")
    parser.add_argument("--test_data_path", type=str, required=True, help="Path to the testing data CSV file")
    parser.add_argument("--output_csv", type=str, required=True, help="Path to save the output CSV file")
    parser.add_argument("--alpha", type=float, default=0.5, help="Alpha parameter for spatial-temporal distance weighting")
    parser.add_argument("--k", type=int, default=300, help="Number of nearest sensors to consider")
    parser.add_argument("--power", type=float, default=2, help="Power parameter for inverse distance weighting")
    args = parser.parse_args()

    # Load data
    train_data = pd.read_csv(args.train_data_path)
    test_data = pd.read_csv(args.test_data_path)

    predictions = []
    alpha = args.alpha
    k = args.k
    power = args.power

    for test_row in test_data.itertuples(index=False):
        target_date = test_row.date
        target_sensor = pd.DataFrame([[test_row.date, test_row.temperture, test_row.lat, test_row.lon]],
                                     columns=['date', 'temperture', 'lat', 'lon'])

        # Calculate distances with alpha
        distances = calculate_space_time_distances(target_date, target_sensor, [train_data], alpha)

        # Select k nearest sensors
        nearest_sensors = select_nearest_sensors(k, distances)

        # Make IDW prediction with alpha and power
        idw_result = idw_prediction(target_sensor, nearest_sensors, alpha, power)
        predictions.append((target_date, test_row.lat, test_row.lon, test_row.temperture, idw_result))

    # Display the predictions
    for date, lat, lon, temperture, prediction in predictions:
        print(f"IDW Prediction for Date: {date}, Lat: {lat}, Lon: {lon}, actual temp: {temperture} - prediction {prediction}")

    # Writing data to CSV file
    with open(args.output_csv, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['Date', 'Lat', 'Lon', 'Actual Temp', 'Prediction'])
        csv_writer.writerows(predictions)

    print(f'Data has been successfully stored in {args.output_csv}')

    # Load the output CSV for evaluation
    df = pd.read_csv(args.output_csv)
    actual_temps = df['Actual Temp']
    predictions = df['Prediction']

    mae = mean_absolute_error(actual_temps, predictions)
    rmse = np.sqrt(mean_squared_error(actual_temps, predictions))
    r2 = r2_score(actual_temps, predictions)

    # Add new columns to the DataFrame
    df['MAE'] = mae
    df['RMSE'] = rmse
    df['R²'] = r2

    # Save the updated DataFrame to CSV
    df.to_csv(args.output_csv, index=False)

    # Print results
    print(f"MAE: {mae}")
    print(f"RMSE: {rmse}")
    print(f"R²: {r2}")

if __name__ == "__main__":
    main()
