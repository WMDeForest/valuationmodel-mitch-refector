def split_listener_history_for_backtesting(df):
    """
    Split the artist listener history data into training and validation sets,
    rounding to whole months.
    
    Args:
        df: DataFrame with 'Date' and 'Monthly Listeners' columns
        
    Returns:
        tuple: (training_df, validation_df) or (None, None) if data is insufficient
    """
    # Convert Date to datetime if it's not already
    df['Date'] = pd.to_datetime(df['Date'], format='%b %d, %Y')
    
    # Sort by date
    df = df.sort_values('Date')
    
    # Get the first and last dates
    start_date = df['Date'].min()
    end_date = df['Date'].max()
    
    # Round start date to beginning of month and end date to end of month
    start_date = start_date.replace(day=1)
    end_date = (end_date.replace(day=1) + pd.DateOffset(months=1) - pd.DateOffset(days=1))
    
    # Calculate total months of data
    total_months = ((end_date.year - start_date.year) * 12 + end_date.month - start_date.month)
    
    # Check if we have minimum required data (27 months)
    if total_months < 27:
        return None, None
        
    # Calculate the cutoff date for validation (exactly 24 months from the end)
    validation_start = end_date - pd.DateOffset(months=24)
    validation_start = validation_start.replace(day=1)  # Start of month
    
    # Split the data
    training_df = df[df['Date'] < validation_start].copy()
    validation_df = df[df['Date'] >= validation_start].copy()
    
    return training_df, validation_df

def split_track_streaming_for_backtesting(df):
    """
    Split the track streaming data into training and validation sets,
    rounding to whole months.
    
    Args:
        df: DataFrame with 'Date' and 'Value' columns
        
    Returns:
        tuple: (training_df, validation_df) or (None, None) if data is insufficient
    """
    # Convert Date to datetime if it's not already
    df['Date'] = pd.to_datetime(df['Date'], format='%b %d, %Y')
    
    # Sort by date
    df = df.sort_values('Date')
    
    # Get the first and last dates
    start_date = df['Date'].min()
    end_date = df['Date'].max()
    
    # Round start date to beginning of month and end date to end of month
    start_date = start_date.replace(day=1)
    end_date = (end_date.replace(day=1) + pd.DateOffset(months=1) - pd.DateOffset(days=1))
    
    # Calculate total months of data
    total_months = ((end_date.year - start_date.year) * 12 + end_date.month - start_date.month)
    
    # Check if we have minimum required data (27 months)
    if total_months < 27:
        return None, None
        
    # Calculate the cutoff date for validation (exactly 24 months from the end)
    validation_start = end_date - pd.DateOffset(months=24)
    validation_start = validation_start.replace(day=1)  # Start of month
    
    # Split the data
    training_df = df[df['Date'] < validation_start].copy()
    validation_df = df[df['Date'] >= validation_start].copy()
    
    return training_df, validation_df