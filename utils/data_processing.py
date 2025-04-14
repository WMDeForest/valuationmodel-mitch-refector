"""
Utility functions for data processing and date handling.

This module provides reusable functions for common data processing tasks
such as date conversion, formatting, and validation.
"""

import pandas as pd

def convert_to_datetime(df, column_name, dayfirst=True):
    """
    Convert a column to datetime format with error handling.
    
    Args:
        df: DataFrame containing the date column
        column_name: Name of the column to convert
        dayfirst: Whether to interpret dates as day first format
        
    Returns:
        tuple: (DataFrame with converted dates, list of issues)
    """
    issues = []
    try:
        df[column_name] = pd.to_datetime(df[column_name], dayfirst=dayfirst, errors='coerce')
        
        # Check for unparseable dates
        if df[column_name].isna().any():
            issues.append("Some dates couldn't be parsed and have been set to 'NaT'.")
    except Exception as e:
        issues.append(f"Failed to convert {column_name} column: {e}")
    
    return df, issues

def sample_data(df, sample_rate=7):
    """
    Sample data by keeping every nth row.
    
    Args:
        df: DataFrame to sample
        sample_rate: Take every nth row
        
    Returns:
        DataFrame with sampled data
    """
    return df.iloc[::sample_rate, :]

def select_columns(df, columns):
    """
    Select only specified columns from a DataFrame.
    
    Args:
        df: DataFrame containing the columns
        columns: List of column names to select
        
    Returns:
        DataFrame with only the selected columns
    """
    return df[columns]

def rename_columns(df, column_map):
    """
    Rename columns according to a mapping.
    
    Args:
        df: DataFrame containing the columns to rename
        column_map: Dictionary mapping original column names to new names
        
    Returns:
        DataFrame with renamed columns
    """
    return df.rename(columns=column_map)

def validate_columns(df, required_columns):
    """
    Check if DataFrame contains all required columns.
    
    Args:
        df: DataFrame to validate
        required_columns: List of column names that must be present
        
    Returns:
        bool: True if all required columns are present, False otherwise
    """
    return all(column in df.columns for column in required_columns)

def extract_earliest_date(df, date_column, input_format='%b %d, %Y', output_format='%d/%m/%Y'):
    """
    Extract the earliest date from a DataFrame's date column and format it.
    
    Args:
        df: DataFrame containing the date column
        date_column: Name of the column containing dates
        input_format: Format of the dates in the input data
        output_format: Desired format for the output date
        
    Returns:
        str: The earliest date formatted according to output_format
    """
    earliest_date = pd.to_datetime(df[date_column].iloc[0], format=input_format).strftime(output_format)
    return earliest_date

def calculate_period_streams(df, cumulative_column, days_back):
    """
    Calculate streams for a specific period by comparing the most recent 
    cumulative value with an earlier value.
    
    Args:
        df: DataFrame containing streaming data with cumulative values
        cumulative_column: Name of the column containing cumulative stream values
        days_back: Number of days to look back for the comparison
        
    Returns:
        int/float: Stream count for the specified period
    """
    total_streams = df[cumulative_column].iloc[-1]
    
    if len(df) > days_back:
        period_streams = total_streams - df[cumulative_column].iloc[-(days_back + 1)]
    else:
        period_streams = total_streams
        
    return period_streams

def process_audience_geography(geography_file=None):
    """
    Process audience geography data from uploaded file.
    Returns a DataFrame with audience distribution and the USA percentage.
    
    Parameters:
    -----------
    geography_file : file object, optional
        The uploaded CSV file containing audience geography data
        
    Returns:
    --------
    tuple:
        A tuple containing two elements:
        
        listener_geography_df : pandas.DataFrame
            A DataFrame containing geographical distribution of listeners with columns:
            - 'Country': The country name
            - 'Spotify Monthly Listeners': Raw count of listeners from this country
            - 'Spotify monthly listeners (%)': Percentage of total listeners (as a decimal)
            This data is used to apply country-specific royalty rates during valuation.
            
        listener_percentage_usa : float
            The proportion of listeners from the United States as a decimal (0.0-1.0).
            This is extracted from the listener_geography_df for convenience since US streams
            are often calculated separately in royalty formulas.
            Defaults to 1.0 (100% USA) if no geography data is provided or if USA
            is not found in the data.
    """
    # Default value if no geography data
    listener_percentage_usa = 1.0
    listener_geography_df = pd.DataFrame()
    
    if geography_file:
        # Process uploaded file
        listener_geography_df = pd.read_csv(geography_file)
        
        # Extract and process geographical data
        listener_geography_df = listener_geography_df[['Country', 'Spotify Monthly Listeners']]
        listener_geography_df = listener_geography_df.groupby('Country', as_index=False)['Spotify Monthly Listeners'].sum()
        
        # Calculate percentage distribution
        total_listeners = listener_geography_df['Spotify Monthly Listeners'].sum()
        listener_geography_df['Spotify monthly listeners (%)'] = (listener_geography_df['Spotify Monthly Listeners'] / total_listeners) * 100
        
        # Normalize percentage values
        listener_geography_df["Spotify monthly listeners (%)"] = pd.to_numeric(listener_geography_df["Spotify monthly listeners (%)"], errors='coerce')
        listener_geography_df["Spotify monthly listeners (%)"] = listener_geography_df["Spotify monthly listeners (%)"] / 100
        
        # Extract US percentage for royalty calculations
        if "United States" in listener_geography_df["Country"].values:
            listener_percentage_usa = listener_geography_df.loc[listener_geography_df["Country"] == "United States", "Spotify monthly listeners (%)"].values[0]
    
    return listener_geography_df, listener_percentage_usa

def process_ownership_data(ownership_file, track_names):
    """
    Process song ownership data from uploaded file.
    
    Parameters:
    -----------
    ownership_file : file object, optional
        The uploaded CSV file containing ownership data
    track_names : list
        List of track names for which to create default ownership entries if no file is provided

    Returns:
    --------
    pandas.DataFrame:
        A DataFrame containing standardized ownership information with columns:
        - 'track_name': Name of the track
        - 'Ownership(%)': Percentage ownership as a decimal (0.0-1.0)
        - 'MLC Claimed(%)': Percentage of mechanical license claims as a decimal (0.0-1.0)
        
        Both percentage values are normalized to decimal format (e.g., 50% → 0.5).
        When no ownership file is provided, defaults to 100% ownership (1.0) and 
        0% MLC claims (0.0) for all tracks in track_names. 
    """
    if ownership_file is not None:
        # Load ownership data with encoding handling
        try:
            ownership_df = pd.read_csv(ownership_file, encoding='latin1')
        except UnicodeDecodeError:
            ownership_df = pd.read_csv(ownership_file, encoding='utf-8')
    
        # Clean and normalize ownership data
        ownership_df['Ownership(%)'] = ownership_df['Ownership(%)'].replace('', 1)
        ownership_df['MLC Claimed(%)'] = ownership_df['MLC Claimed(%)'].replace('', 0)
        ownership_df['Ownership(%)'] = pd.to_numeric(ownership_df['Ownership(%)'], errors='coerce').fillna(1)
        ownership_df['MLC Claimed(%)'] = pd.to_numeric(ownership_df['MLC Claimed(%)'], errors='coerce').fillna(0)
        
        # Convert percentages to decimal format
        ownership_df['Ownership(%)'] = ownership_df['Ownership(%)'].apply(lambda x: x / 100 if x > 1 else x)
        ownership_df['MLC Claimed(%)'] = ownership_df['MLC Claimed(%)'].apply(lambda x: x / 100 if x > 1 else x)
    else:
        # Create empty ownership dataframe if no file is uploaded
        ownership_df = pd.DataFrame({
            'track_name': track_names,
            'Ownership(%)': [1.0] * len(track_names),
            'MLC Claimed(%)': [0.0] * len(track_names)
        })
    
    return ownership_df 