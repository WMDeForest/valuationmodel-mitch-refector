"""
Forecast projections utility functions.

This module provides functions to process stream forecasts into revenue projections
with geographic distribution and time value adjustments.
"""

import pandas as pd

def create_monthly_track_revenue_projections(
    track_name,
    track_streams_forecast_df,
    listener_geography_df,
    worldwide_royalty_rates_df,
    discount_rate
):
    """
    Create a DataFrame with monthly track revenue projections based on stream forecasts
    and geographic distribution of listeners.
    
    Parameters
    ----------
    track_name : str
        Name of the track being analyzed
    track_streams_forecast_df : pandas.DataFrame
        DataFrame containing the forecasted streams by month
    listener_geography_df : pandas.DataFrame
        DataFrame containing the geographic distribution of listeners
    worldwide_royalty_rates_df : pandas.DataFrame
        DataFrame containing country-specific royalty rates
    discount_rate : float
        Annual discount rate for time value calculations
        
    Returns
    -------
    pandas.DataFrame
        DataFrame containing monthly revenue projections with country-specific
        breakdown and time-value adjusted calculations
    """
    # Create the base DataFrame with track info and stream forecasts
    projections_df = pd.DataFrame({
        'track_name': [track_name] * len(track_streams_forecast_df),
        'month': track_streams_forecast_df['month'],
        'predicted_streams_for_month': track_streams_forecast_df['predicted_streams_for_month'],
        'month_index': range(1, len(track_streams_forecast_df) + 1)  # Add month index right away
    })
    
    # Apply geographic distribution and calculate royalty values
    projections_df = apply_geographic_distribution(
        projections_df,
        listener_geography_df,
        worldwide_royalty_rates_df
    )
    
    # Apply time value of money discount
    projections_df['DISC'] = (projections_df['Total']) / ((1 + discount_rate / 12) ** (projections_df['month_index'] + 2.5))
    
    return projections_df

def apply_geographic_distribution(
    projections_df,
    listener_geography_df,
    worldwide_royalty_rates_df
):
    """
    Apply geographic distribution to stream forecasts and calculate country-specific royalties.
    
    Parameters
    ----------
    projections_df : pandas.DataFrame
        DataFrame containing the base monthly projections
    listener_geography_df : pandas.DataFrame
        DataFrame containing the geographic distribution of listeners
    worldwide_royalty_rates_df : pandas.DataFrame
        DataFrame containing country-specific royalty rates
        
    Returns
    -------
    pandas.DataFrame
        DataFrame with added country-specific royalty calculations
    """
    # Process all geography data in one pass
    for index, row in listener_geography_df.iterrows():
        country = row['Country']
        percentage = row['Spotify monthly listeners (%)']
        
        # Add country percentage
        projections_df[country + ' %'] = percentage
        
        # Get country-specific royalty rate
        if country in worldwide_royalty_rates_df.columns:
            mean_final_5 = worldwide_royalty_rates_df[country].dropna().tail(5).mean()
            projections_df[country + ' Royalty Rate'] = mean_final_5
        else:
            # Use a default rate if country not found
            projections_df[country + ' Royalty Rate'] = 0
        
        # Calculate stream value and royalty value in one go
        projections_df[country + ' Value'] = projections_df['predicted_streams_for_month'] * projections_df[country + ' %']
        projections_df[country + ' Royalty Value'] = projections_df[country + ' Value'] * projections_df[country + ' Royalty Rate']
    
    # Clean up intermediate calculation columns
    percentage_columns = [country + ' %' for country in listener_geography_df['Country']]
    columns_to_drop = [country + ' Value' for country in listener_geography_df['Country']] + [country + ' Royalty Rate' for country in listener_geography_df['Country']]
    
    # Sum all country royalty values before dropping intermediates
    projections_df['Total'] = projections_df[[country + ' Royalty Value' for country in listener_geography_df['Country']]].sum(axis=1)
    
    # Remove intermediate columns
    projections_df.drop(columns=percentage_columns + columns_to_drop, inplace=True)
    
    return projections_df

def aggregate_into_yearly_periods(projections_df, first_year_months=9):
    """
    Aggregate monthly projections into yearly periods.
    
    Parameters
    ----------
    projections_df : pandas.DataFrame
        DataFrame containing monthly revenue projections
    first_year_months : int, optional
        Number of months to include in the first year (default: 9)
        
    Returns
    -------
    pandas.DataFrame
        DataFrame with yearly aggregated revenue projections
    """
    n_rows = len(projections_df)
    rows_per_period = 12
    
    # Initialize period pattern for aggregation
    period_pattern = []
    
    # First year (typically 9 months)
    period_pattern.extend([1] * first_year_months)
    
    # Calculate remaining rows after first period
    remaining_rows = n_rows - first_year_months
    
    # Assign remaining months to yearly periods (12 months per year)
    for period in range(2, (remaining_rows // rows_per_period) + 2):
        period_pattern.extend([period] * rows_per_period)

    # Ensure pattern length matches DataFrame rows
    if len(period_pattern) > n_rows:
        period_pattern = period_pattern[:n_rows]  # Trim if too long
    else:
        period_pattern.extend([period] * (n_rows - len(period_pattern)))  # Extend if too short

    # Assign periods to months
    projections_df['Period'] = period_pattern

    # Group data by period and aggregate
    aggregated_df = projections_df.groupby('Period').agg({
        'track_name': 'first',
        'month': 'first',  # First month in each period
        'DISC': 'sum'      # Sum discounted values
    }).reset_index(drop=True)

    # Rename for clarity
    aggregated_df.rename(columns={'month': 'Start_Month'}, inplace=True)

    # Add year numbers and drop start month
    aggregated_df['Year'] = range(1, len(aggregated_df) + 1)
    aggregated_df.drop(columns=['Start_Month'], inplace=True)
    
    return aggregated_df

def apply_ownership_adjustments(track_valuation_results_df, ownership_df):
    """
    Adjust track valuation results based on ownership percentages and MLC claims.
    
    Parameters:
    -----------
    track_valuation_results_df : pd.DataFrame
        DataFrame containing track valuation results
    ownership_df : pd.DataFrame
        DataFrame containing ownership and MLC claimed percentages
        
    Returns:
    --------
    pd.DataFrame
        Adjusted valuation results with ownership percentages applied
    """
    # Merge valuation results with ownership information
    adjusted_df = track_valuation_results_df.merge(
        ownership_df[['track_name', 'MLC Claimed(%)', 'Ownership(%)']], 
        on='track_name', 
        how='left'
    )

    # Ensure ownership percentages are properly formatted
    adjusted_df['MLC Claimed(%)'] = pd.to_numeric(adjusted_df['MLC Claimed(%)'], errors='coerce').fillna(0)
    adjusted_df['Ownership(%)'] = pd.to_numeric(adjusted_df['Ownership(%)'], errors='coerce').fillna(1)
    
    # Adjust historical value based on MLC claims and ownership percentage
    adjusted_df['historical_royalty_value_time_adjusted'] = adjusted_df.apply(
        lambda row: min(
            (1 - row['MLC Claimed(%)']) * row['historical_royalty_value_time_adjusted'], 
            row['Ownership(%)'] * row['historical_royalty_value_time_adjusted']
        ),
        axis=1
    )
    
    # Adjust forecast values based on ownership percentage
    adjusted_df['undiscounted_future_royalty_value'] = adjusted_df['undiscounted_future_royalty_value'].astype(float) * adjusted_df['Ownership(%)']
    adjusted_df['discounted_future_royalty_value'] = adjusted_df['discounted_future_royalty_value'].astype(float) * adjusted_df['Ownership(%)']
    adjusted_df['total_track_valuation'] = adjusted_df['discounted_future_royalty_value'] + adjusted_df['historical_royalty_value_time_adjusted']
    
    # Remove ownership columns after applying adjustments
    adjusted_df = adjusted_df.drop(columns=['Ownership(%)', 'MLC Claimed(%)'])
    
    return adjusted_df 