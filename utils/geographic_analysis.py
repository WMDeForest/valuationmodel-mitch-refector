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
    fig, ax = plt.subplots()
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
    
    # Visual enhancements
    fig.patch.set_visible(False)
    ax.set_facecolor('none')
    ax.patch.set_alpha(0)
    plt.tight_layout()
    
    return fig, ax 