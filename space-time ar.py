import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.tsa.api import VAR
import argparse


def main():
    # Argument parsing
    print("Starting the space-time nearest neighbor script...")  # Debugging output
    parser = argparse.ArgumentParser(description="Space-Time Autoregressive Model (VAR)")
    parser.add_argument("--train_data_path", type=str, required=True, help="Path to the training data CSV file")
    parser.add_argument("--test_data_path", type=str, required=True, help="Path to the testing data CSV file")
    parser.add_argument("--output_csv", type=str, required=True, help="Path to save the output CSV file")
    args = parser.parse_args()

    # Load data
    train_data = pd.read_csv(args.train_data_path)
    test_data = pd.read_csv(args.test_data_path)

    # Convert date to datetime type
    train_data['date'] = pd.to_datetime(train_data['date'], errors='coerce')
    test_data['date'] = pd.to_datetime(test_data['date'], errors='coerce')

    # Set 'date' as the index if available
    if 'date' in train_data.columns:
        train_data.set_index('date', inplace=True)
    else:
        print("No 'date' column found in the training data.")

    # Train VAR model
    model = VAR(train_data)
    model_fit = model.fit()

    # Forecast the same number of steps as in the test data
    forecast_steps = len(test_data)
    predictions = model_fit.forecast(train_data.values, steps=forecast_steps)

    # Evaluate model
    mae = mean_absolute_error(test_data['temperture'], predictions[:, 0])
    rmse = np.sqrt(mean_squared_error(test_data['temperture'], predictions[:, 0]))
    r2 = r2_score(test_data['temperture'], predictions[:, 0])
    test_data['predict temp'] = predictions[:, 0]
    # print(test_data)

    print(f'Mean Absolute Error (MAE): {mae}')
    print(f'Root Mean Squared Error (RMSE): {rmse}')
    print(f'R-squared (R²): {r2}')

    # Save predictions and metrics to CSV
    test_data.to_csv(args.output_csv, index=False)
    df = pd.read_csv(args.output_csv)
    df['MAE'] = mae
    df['RMSE'] = rmse
    df['R²'] = r2
    df.to_csv(args.output_csv, index=False)


if __name__ == "__main__":
    main()
