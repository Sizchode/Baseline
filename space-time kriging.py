import numpy as np
import pandas as pd
from pykrige.ok3d import OrdinaryKriging3D
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import argparse

def main():
    # Argument parsing
    print("Starting the space-time nearest neighbor script...")
    parser = argparse.ArgumentParser(description="Space-Time Kriging Model")
    parser.add_argument("--train_data_path", type=str, required=True, help="Path to the training data CSV file")
    parser.add_argument("--test_data_path", type=str, required=True, help="Path to the testing data CSV file")
    parser.add_argument("--output_csv", type=str, required=True, help="Path to save the output CSV file")
    parser.add_argument("--variogram_model", type=str, default="linear", help="Variogram model to use (e.g., 'linear')")
    parser.add_argument("--verbose", type=bool, default=True, help="Whether to print detailed kriging information")
    args = parser.parse_args()

    # Load data
    train_data = pd.read_csv(args.train_data_path)
    test_data = pd.read_csv(args.test_data_path)

    # Convert date to datetime and then to integer format (UNIX timestamp)
    train_data['date'] = pd.to_datetime(train_data['date']).astype(int)
    test_data['date'] = pd.to_datetime(test_data['date']).astype(int)

    # Extract relevant columns
    train_coordinates = train_data[['lat', 'lon', 'date']].values
    train_values = train_data['temperture'].values

    # Perform Ordinary Kriging with specified hyperparameters
    ok3d = OrdinaryKriging3D(train_coordinates[:, 0], train_coordinates[:, 1], train_coordinates[:, 2],
                             train_values, variogram_model=args.variogram_model, verbose=args.verbose)

    # Iterate through test data and make predictions
    predictions = []
    for test_row in test_data.itertuples(index=False):
        test_coordinates = np.array([test_row.lat, test_row.lon, test_row.date])
        prediction, _ = ok3d.execute('points', test_coordinates[0], test_coordinates[1], test_coordinates[2])

        # Extract the number from the prediction
        prediction_value = prediction.data[0] if not prediction.mask[0] else None
        predictions.append(prediction_value)

    # Add predictions to the test_data DataFrame
    test_data['predicted_temperature'] = predictions

    # Convert timestamp back to human-readable date
    test_data['date'] = pd.to_datetime(test_data['date'])

    # Save the test_data DataFrame with predictions to a new CSV file
    test_data.to_csv(args.output_csv, index=False)

    # Evaluate the model
    prediction_temp = test_data['predicted_temperature']
    actual_temp = test_data['temperture']

    mae = mean_absolute_error(actual_temp, prediction_temp)
    rmse = np.sqrt(mean_squared_error(actual_temp, prediction_temp))
    r2 = r2_score(actual_temp, prediction_temp)

    # Save evaluation metrics to the CSV
    df = pd.read_csv(args.output_csv)
    df['MAE'] = mae
    df['RMSE'] = rmse
    df['R²'] = r2
    df.to_csv(args.output_csv, index=False)

    # Print results
    print(f"Mean Absolute Error (MAE): {mae}")
    print(f"Root Mean Squared Error (RMSE): {rmse}")
    print(f"R² Score: {r2}")

if __name__ == "__main__":
    main()
