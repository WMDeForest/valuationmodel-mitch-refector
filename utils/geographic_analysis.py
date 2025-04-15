"""
Geographic analysis utility functions.

This module provides functions to analyze and visualize geographic
distribution of music streaming data and revenue projections.
"""

import pandas as pd
import matplotlib.pyplot as plt


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


def create_country_distribution_chart(top_countries, top_percentage_sum):
    """
    Create a horizontal bar chart showing top countries by revenue contribution.
    
    Parameters
    ----------
    top_countries : pandas.DataFrame
        DataFrame containing top countries by revenue
    top_percentage_sum : float
        Sum of percentage contributions from top countries
        
    Returns
    -------
    tuple
        (matplotlib.figure.Figure, matplotlib.axes.Axes)
    """
    # Create horizontal bar chart
    fig, ax = plt.subplots(facecolor='white')
    bar_color = 'teal'
    bars = ax.barh(top_countries['Country'], top_countries['forecast_no_disc_numeric'], color=bar_color)

    # Configure chart appearance
    ax.set_xlabel('% of Forecast Value')
    ax.set_title(f'Top {len(top_countries)} Countries Contribute {top_percentage_sum:.1f}% to Total Forecast Value')
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${int(x):,}"))
    max_value = top_countries['forecast_no_disc_numeric'].max()
    ax.set_xlim(0, max_value * 1.25)
    ax.set_xticks([])
    
    # Add percentage labels to bars
    for bar, percentage in zip(bars, top_countries['Percentage']):
        width = bar.get_width()
        ax.text(width + (width * 0.01), bar.get_y() + bar.get_height() / 2, 
                f'{percentage:.1f}%', va='center', ha='left', 
                fontsize=10, color='black')
    
    # Set white background
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    plt.tight_layout()
    
    return fig, ax


def create_yearly_revenue_chart(yearly_revenue_df):
    """
    Create a bar chart showing yearly revenue projections.
    
    Parameters
    ----------
    yearly_revenue_df : pandas.DataFrame
        DataFrame containing yearly revenue data with 'Year' and 'DISC' columns
        
    Returns
    -------
    tuple
        (matplotlib.figure.Figure, matplotlib.axes.Axes)
    """
    # Filter to first 10 years only
    yearly_revenue_df = yearly_revenue_df[yearly_revenue_df['Year'] <= 10].copy()
    
    # Create bar chart
    fig, ax = plt.subplots(facecolor='white', figsize=(10, 6))
    bar_color = 'teal'
    bars = ax.bar(yearly_revenue_df['Year'], yearly_revenue_df['DISC'], color=bar_color)

    # Configure chart appearance
    ax.set_xlabel('Year')
    ax.set_title('Income by Year (discounted)')
    ax.set_ylabel('')
    ax.yaxis.set_visible(False)
    
    # Set x-axis ticks to show years 1-10
    ax.set_xticks(range(1, 11))
    ax.set_xlim(0.5, 10.5)  # Add padding on both sides
    
    # Set y-axis limits with some headroom
    max_value = yearly_revenue_df['DISC'].max()
    ax.set_ylim(0, max_value * 1.25)

    # Add value labels to bars
    for bar, value in zip(bars, yearly_revenue_df['DISC']):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height, f'${int(value)}', 
                va='bottom', ha='center', fontsize=10, color='black')
    
    # Set white background
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    plt.tight_layout()
    
    return fig, ax


def detect_streaming_fraud(listener_geography_df, population_df, threshold_percentage=20):
    """
    Detect potential streaming fraud by checking if listener counts exceed
    a threshold percentage of a country's population.
    
    Parameters
    ----------
    listener_geography_df : pandas.DataFrame
        DataFrame containing geographical distribution of listeners
    population_df : pandas.DataFrame
        DataFrame containing country population data
    threshold_percentage : float, optional
        Threshold percentage of population to flag as suspicious (default: 20)
        
    Returns
    -------
    list
        List of countries with potential streaming fraud
    """
    # Cross-reference audience data with population data
    warning_df = pd.merge(listener_geography_df, population_df, on='Country', how='left')
    
    # Calculate threshold for suspicious activity
    warning_df['ThresholdPopulation'] = warning_df['Population'] * (threshold_percentage / 100)
    
    # Flag countries with abnormally high listener numbers
    warning_df['AboveThreshold'] = warning_df['Spotify Monthly Listeners'] > warning_df['ThresholdPopulation']
    warning_df['Alert'] = warning_df['AboveThreshold'].apply(lambda x: 1 if x else 0)
    
    # Get list of countries with potential streaming fraud
    alert_countries = warning_df[warning_df['Alert'] == 1]['Country'].tolist()
    
    return alert_countries 