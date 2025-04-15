import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt


def display_track_selection_ui(track_catalog_df):
    """Display the track selection UI elements and return the selected songs"""
    st.write("Data Preview:")
    st.write(track_catalog_df)
    songs = sorted(track_catalog_df['track_name'].unique(), key=lambda x: x.lower())
    selected_songs = st.multiselect('Select Songs', songs, default=songs)
    return selected_songs 

def display_financial_parameters_ui():
    """Display the financial parameters input UI and return the values"""
    # Define default discount rate locally
    default_discount_rate = 4.50  # 4.5%
    discount_rate = st.number_input('Discount Rate (%)', 
                                   min_value=0.00, 
                                   max_value=10.00, 
                                   value=default_discount_rate,
                                   step=0.01, 
                                   format="%.2f") / 100
    return discount_rate 

def format_valuation_results(valuation_df):
    """
    Format valuation results for display in the UI with commas and currency symbols.
    
    Parameters:
    -----------
    valuation_df : pd.DataFrame
        DataFrame containing valuation results with numeric values
        
    Returns:
    --------
    pd.DataFrame
        Formatted DataFrame with values converted to strings with commas and currency symbols
    """
    # Create a copy to avoid modifying the original DataFrame
    formatted_df = valuation_df.copy()
    
    # Format stream counts with commas
    formatted_df['historical_streams'] = formatted_df['historical_streams'].astype(float).apply(lambda x: f"{int(round(x)):,}")
    formatted_df['forecast_streams'] = formatted_df['forecast_streams'].astype(float).apply(lambda x: f"{int(round(x)):,}")
    
    # Format monetary values with dollar signs and commas
    formatted_df['undiscounted_future_royalty'] = formatted_df['undiscounted_future_royalty'].astype(float).apply(lambda x: f"${int(round(x)):,}")
    formatted_df['discounted_future_royalty'] = formatted_df['discounted_future_royalty'].astype(float).apply(lambda x: f"${int(round(x)):,}")
    formatted_df['historical_royalty_value'] = formatted_df['historical_royalty_value'].astype(float).apply(lambda x: f"${int(round(x)):,}")
    formatted_df['total_track_valuation'] = formatted_df['total_track_valuation'].astype(float).apply(lambda x: f"${int(round(x)):,}")
    
    return formatted_df 

def display_valuation_results(valuation_df):
    """
    Format and display valuation results in the Streamlit UI.
    
    Parameters:
    -----------
    valuation_df : pd.DataFrame
        DataFrame containing valuation results with numeric values
    """
    # Format the valuation results
    formatted_df = format_valuation_results(valuation_df)
    
    # Display the formatted results
    st.write(formatted_df)
    return formatted_df 

def display_valuation_summary(valuation_df):
    """
    Calculate summed values across all tracks and display a summary table.
    
    Parameters:
    -----------
    valuation_df : pd.DataFrame
        Formatted DataFrame containing valuation results for all tracks
        
    Returns:
    --------
    pd.DataFrame
        Summary DataFrame with totals across all tracks in the catalog
    """
    # Extract numeric values from formatted strings
    numeric_df = valuation_df.copy()
    
    # Convert string currency values back to numbers for summing
    for col in ['historical_royalty_value', 'undiscounted_future_royalty', 
                'discounted_future_royalty', 'total_track_valuation']:
        numeric_df[col] = numeric_df[col].str.replace('$', '').str.replace(',', '').astype(float)
    
    # Calculate sums across all tracks in the catalog
    catalog_valuation_summary_df = pd.DataFrame({
        'Metric': ['Historical Value', 'Undiscounted Future Value', 'Discounted Future Value', 'Total Valuation'],
        'Sum': [
            numeric_df['historical_royalty_value'].sum(),
            numeric_df['undiscounted_future_royalty'].sum(),
            numeric_df['discounted_future_royalty'].sum(),
            numeric_df['total_track_valuation'].sum()
        ]
    })
    
    # Format sum values with dollar signs and commas
    catalog_valuation_summary_df['Sum'] = catalog_valuation_summary_df['Sum'].apply(lambda x: f"${int(round(x)):,}")
    
    # Display the summary table with a title
    st.write("Summed Values:")
    st.write(catalog_valuation_summary_df)
    
    return catalog_valuation_summary_df 

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