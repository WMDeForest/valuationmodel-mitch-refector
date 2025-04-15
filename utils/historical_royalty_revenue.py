"""
Historical Royalty Revenue calculation utilities.

This module provides functions to calculate the historical royalty revenue from a track's streaming history,
applying appropriate royalty rates and adjustment factors.
"""

def calculate_historical_royalty_revenue(
    total_historical_track_streams,
    mechanical_royalty_rates_df,
    date_range_mask,
    listener_percentage_usa,
    discount_rate,
    historical_value_time_adjustment,
    premium_stream_percentage=0.4,
    ad_supported_stream_percentage=0.6
):
    """
    Calculate the historical royalty revenue of a track based on its past streaming data.
    
    This function computes the estimated royalty revenue earned by a track from its release date
    until the valuation cutoff date. It accounts for:
    1. Different royalty rates for premium vs. ad-supported streams
    2. Geographic distribution of listeners (US vs. non-US royalty rates)
    3. Time value of money adjustment
    
    Parameters
    ----------
    total_historical_track_streams : int
        Total number of streams the track has accumulated since release
    mechanical_royalty_rates_df : pandas.DataFrame
        DataFrame containing historical mechanical royalty rates
    date_range_mask : pandas.Series
        Boolean mask for filtering the royalty rates DataFrame to the relevant date range
    listener_percentage_usa : float
        Percentage (0-1) of streams coming from USA listeners
    discount_rate : float
        Annual discount rate for time value adjustment (e.g., 0.045 for 4.5%)
    historical_value_time_adjustment : int
        Number of months to use in time value adjustment calculation
    premium_stream_percentage : float, optional
        Percentage (0-1) of streams from premium subscribers (default: 0.4)
    ad_supported_stream_percentage : float, optional
        Percentage (0-1) of streams from ad-supported users (default: 0.6)
        
    Returns
    -------
    float
        The calculated historical royalty revenue with time value adjustment applied
    """
    # Calculate average royalty rates for the specified date range
    avg_spotify_ad_supported_royalty_rate = mechanical_royalty_rates_df.loc[date_range_mask, 'Spotify_Ad-supported'].mean()
    avg_spotify_premium_royalty_rate = mechanical_royalty_rates_df.loc[date_range_mask, 'Spotify_Premium'].mean()
    
    # Calculate revenue from each stream type
    historical_ad_supported_stream_revenue = ad_supported_stream_percentage * total_historical_track_streams * avg_spotify_ad_supported_royalty_rate
    historical_premium_stream_revenue = premium_stream_percentage * total_historical_track_streams * avg_spotify_premium_royalty_rate
    
    # Calculate total historical royalty revenue with time value adjustment
    historical_royalty_revenue_time_adjusted = ((historical_ad_supported_stream_revenue + historical_premium_stream_revenue) * 
                                              listener_percentage_usa) / ((1 + discount_rate / 12) ** historical_value_time_adjustment)
    
    return historical_royalty_revenue_time_adjusted 