"""
Data preprocessing functions for decay modeling.
"""
import pandas as pd

def remove_anomalies(data):
    """
    Clean streaming data by removing outliers using IQR method and interpolation.
    
    Args:
        data: DataFrame with 'Date' and 'Streams' columns
        
    Returns:
        DataFrame with anomalies removed and replaced with interpolated values
    """
    # Calculate the 4-week moving average
    data['4_Week_MA'] = data['Streams'].rolling(window=4, min_periods=1).mean()

    # Resample the data to monthly sums
    data.set_index('Date', inplace=True)
    monthly_data = data['4_Week_MA'].resample('M').sum().reset_index()

    # Detect anomalies using the IQR method
    Q1 = monthly_data['4_Week_MA'].quantile(0.25)
    Q3 = monthly_data['4_Week_MA'].quantile(0.75)
    IQR = Q3 - Q1

    # Define bounds for anomaly detection
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Detect anomalies
    monthly_data['is_anomaly'] = (monthly_data['4_Week_MA'] < lower_bound) | (monthly_data['4_Week_MA'] > upper_bound)

    # Interpolate anomalies
    for i in range(1, len(monthly_data) - 1):
        if monthly_data.loc[i, 'is_anomaly']:
            monthly_data.loc[i, '4_Week_MA'] = (monthly_data.loc[i - 1, '4_Week_MA'] + monthly_data.loc[i + 1, '4_Week_MA']) / 2

    return monthly_data 