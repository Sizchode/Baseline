import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import argparse

def main():

    # Argument parsing
    print("Starting the space-time nearest neighbor script...")  # Debugging output
    parser = argparse.ArgumentParser(description="Space-Time Linear Regression")
    parser.add_argument("--train_data_path", type=str, required=True, help="Path to the training data CSV file")
    parser.add_argument("--test_data_path", type=str, required=True, help="Path to the testing data CSV file")
    parser.add_argument("--output_csv", type=str, required=True, help="Path to save the output CSV file")
    args = parser.parse_args()

    # Load training data
    train_data = pd.read_csv(args.train_data_path)

    # Extract features and target variable
    X_train = train_data[['date', 'lat', 'lon']]
    y_train = train_data['temperture']

    # Initialize OneHotEncoder for date encoding
    date_encoder = OneHotEncoder()

    # Column transformer to apply one-hot encoding to the date column
    preprocessor = ColumnTransformer(
        transformers=[
            ('date', date_encoder, ['date'])
        ],
        remainder='passthrough'
    )

    # Initialize the linear regression model
    model = LinearRegression()

    # Build the pipeline that includes preprocessor and model
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', model)
    ])

    # Train the model
    pipeline.fit(X_train, y_train)

    # Load testing data
    test_data = pd.read_csv(args.test_data_path)

    # Extract features from test data
    X_test = test_data[['date', 'lat', 'lon']]

    # Make predictions on the test data
    y_pred = pipeline.predict(X_test)

    # Add the predicted temperature to the test data
    test_data['predicted_temp'] = y_pred

    # Save predictions to CSV
    test_data[['date', 'lat', 'lon', 'temperture', 'predicted_temp']].to_csv(args.output_csv, index=False)

    # Evaluate the model
    mae = mean_absolute_error(test_data['temperture'], test_data['predicted_temp'])
    rmse = np.sqrt(mean_squared_error(test_data['temperture'], test_data['predicted_temp']))
    r2 = r2_score(test_data['temperture'], test_data['predicted_temp'])

    # Save evaluation metrics to the CSV
    df = pd.read_csv(args.output_csv)
    df['MAE'] = mae
    df['RMSE'] = rmse
    df['R²'] = r2
    df.to_csv(args.output_csv, index=False)

    # Print evaluation metrics
    print(f"Mean Absolute Error (MAE): {mae}")
    print(f"Root Mean Squared Error (RMSE): {rmse}")
    print(f"R² Score: {r2}")

if __name__ == "__main__":
    main()
