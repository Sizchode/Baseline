import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import argparse

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

# Main function to run the space-time nearest neighbor model
def main():
    # Argument parsing
    print("Starting the space-time nearest neighbor script...")  
    parser = argparse.ArgumentParser(description="Space-Time Nearest Neighbor")
    parser.add_argument("--train_data_path", type=str, default="train_data.csv", help="Path to the training data CSV file")
    parser.add_argument("--test_data_path", type=str, default="test_data.csv", help="Path to the testing data CSV file")
    parser.add_argument("--output_csv", type=str, default="space-time-nn.csv", help="Path to save the output CSV file")
    args = parser.parse_args()

    # Load data
    train_data = pd.read_csv(args.train_data_path)
    test_data = pd.read_csv(args.test_data_path)

    # Apply the function to each row in the test data
    test_data['predicted_temperature'] = test_data.apply(lambda row: find_space_time_nearest_neighbor(row, train_data),
                                                         axis=1)

    # Save predictions
    test_data[['date', 'lat', 'lon', 'temperture', 'predicted_temperature']].to_csv(args.output_csv, index=False)

    # Evaluate the model
    actual_temp = test_data['temperture']
    prediction_temp = test_data['predicted_temperature']

    mae = mean_absolute_error(actual_temp, prediction_temp)
    rmse = np.sqrt(mean_squared_error(actual_temp, prediction_temp))
    r2 = r2_score(actual_temp, prediction_temp)

    # Save evaluation metrics
    df = pd.read_csv(args.output_csv)
    df['MAE'] = mae
    df['RMSE'] = rmse
    df['R²'] = r2
    df.to_csv(args.output_csv, index=False)

    # Print evaluation results
    print(f"Mean Absolute Error (MAE): {mae}")
    print(f"Root Mean Squared Error (RMSE): {rmse}")
    print(f"R-squared (R²): {r2}")

if __name__ == "__main__":
    main()
