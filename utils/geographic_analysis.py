"""
Geographic analysis utility functions.

This module provides functions to analyze and visualize geographic
distribution of music streaming data and revenue projections.
"""

import pandas as pd
import matplotlib.pyplot as plt
from utils.ui_functions import create_country_distribution_chart, create_yearly_revenue_chart


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


def process_country_breakdown(listener_geography_df, monthly_track_revenue_projections_df):
    """
    Process country-specific revenue data and calculate distribution percentages.
    
    Parameters
    ----------
    listener_geography_df : pandas.DataFrame
        DataFrame containing the geographic distribution of listeners
    monthly_track_revenue_projections_df : pandas.DataFrame
        DataFrame containing monthly revenue projections with country breakdowns
        
    Returns
    -------
    pandas.DataFrame
        DataFrame with country revenue breakdown and percentage contributions
    """
    # Extract country-specific revenue data
    country_breakdown = []
    for index, row in listener_geography_df.iterrows():
        country = row['Country']
        forecast_no_disc_value = monthly_track_revenue_projections_df[country + ' Royalty Value'].sum() 
        country_breakdown.append({
            'Country': country,
            'forecast_no_disc': forecast_no_disc_value
        })
    
    # Process country breakdown data
    df_country_breakdown = pd.DataFrame(country_breakdown)
    df_country_breakdown['forecast_no_disc_numeric'] = df_country_breakdown['forecast_no_disc'].replace({'\$': '', ',': ''}, regex=True).astype(float)

    # Calculate total forecast value and country percentages
    total_forecast_no_disc_value = df_country_breakdown['forecast_no_disc_numeric'].sum()
    df_country_breakdown['Percentage'] = (df_country_breakdown['forecast_no_disc_numeric'] / total_forecast_no_disc_value) * 100
    
    return df_country_breakdown


def get_top_countries(country_breakdown_df, top_n=10):
    """
    Get the top countries by revenue contribution.
    
    Parameters
    ----------
    country_breakdown_df : pandas.DataFrame
        DataFrame with country revenue breakdown
    top_n : int, optional
        Number of top countries to return (default: 10)
        
    Returns
    -------
    tuple
        (pandas.DataFrame with top countries, float percentage sum)
    """
    top_countries = country_breakdown_df.sort_values(by='forecast_no_disc_numeric', ascending=False).head(top_n)
    top_n_percentage_sum = top_countries['Percentage'].sum()
    
    return top_countries, top_n_percentage_sum 