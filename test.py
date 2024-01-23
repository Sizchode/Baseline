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

train_data.to_csv("train_data.csv", index=False)
