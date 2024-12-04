import numpy as np
import pandas as pd
import yaml
import os
from sklearn.preprocessing import MinMaxScaler

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

with open(project_root + '/config.yaml') as stream:
    try:
        config = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)

df = pd.read_csv(project_root + config['paths']['apple_path'])

# Convert date column to DateTime Object
df['Date'] = pd.to_datetime(df["Date"], utc=True)
df = df.sort_values('Date').reset_index(drop=True)

# Calculate Relevant Features
df['Return'] = df['Close'].pct_change()
df['SMA10'] = df['Close'].rolling(window=10).mean()
df['SMA50'] = df['Close'].rolling(window=50).mean()
df['Volatility'] = df['Return'].rolling(window=10).std()

# Drop the dividends and stock split as they are unused at the moment
df = df.drop(columns=['Dividends', 'Stock Splits'])
columns_to_scale = ['Open', 'High', 'Low', 'Close', 'Volume', 'SMA10', 'SMA50', 'Volatility']

# Scale the data
scaler = MinMaxScaler()
df[columns_to_scale] = scaler.fit_transform(df[columns_to_scale])

# drop null values caused by the rolling calculations
df = df.dropna().reset_index(drop=True)

# Now split the data into training and testing

# Training: 2010-2018
train_df = df[df['Date'] < '2018-01-01']

# Validation: 2019-2020
val_df = df[(df['Date'] >= '2018-01-01') & (df['Date'] < '2021-01-01')]

# Testing: 2021-2023
test_df = df[df['Date'] >= '2021-01-01']

# Save the data into the data folder under srd
train_df.to_csv(project_root + '/src/data/train.csv', index=False)
val_df.to_csv(project_root + '/src/data/val.csv', index=False)
test_df.to_csv(project_root + '/src/data/test.csv', index=False)
