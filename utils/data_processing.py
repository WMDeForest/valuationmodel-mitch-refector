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