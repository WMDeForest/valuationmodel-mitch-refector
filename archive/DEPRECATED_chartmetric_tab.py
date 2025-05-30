"""
DEPRECATED: This module is no longer in use as of April 2024.
The ChartMetric functionality has been moved directly into streamlit_app.py.

This file is kept for reference purposes only.
DO NOT USE in new code.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import base64

# Import chartmetric service
from services.chartmetric_services import chartmetric_service as chartmetric
from services.chartmetric_services.dto import ArtistStateCampareListner, TrackSpotifyState, CountryListeners

# Import utility functions
from utils.data_processing import (
    convert_to_datetime,
    sample_data,
    select_columns,
    rename_columns,
    validate_columns,
    extract_earliest_date,
    calculate_period_streams,
    calculate_months_since_release,
    calculate_monthly_stream_averages,
    extract_track_metrics
)
from utils.decay_models import (
    exponential_decay,
    remove_anomalies,
    analyze_listener_decay,
    calculate_monthly_listener_decay_rate
)
from utils.population_utils.country_code_to_name import country_code_to_name

def get_artist_by_id(artist_id):
    """Get artist details by Chartmetric ID"""
    try:
        # Call the ChartMetric API to get artist details
        # This is a placeholder - would need to implement the actual API call
        # For now, return a simple dict with the ID and a placeholder name
        return {
            "id": artist_id,
            "name": f"Artist ID: {artist_id}",
            "image_url": "https://example.com/image.jpg"
        }
    except Exception as e:
        st.error(f"Error fetching artist details: {str(e)}")
        return None

def get_artist_monthly_listeners(artist_id):
    """Get artist monthly listeners from ChartMetric API"""
    try:
        # Call the ChartMetric API to get artist monthly listeners
        listeners_data = chartmetric.get_artist_spotify_stats(artist_id=artist_id)
        
        # Convert to DataFrame
        listeners_df = pd.DataFrame([
            {"Date": item.timestp, "Monthly Listeners": item.value}
            for item in listeners_data
        ])
        
        # Ensure Date is datetime format
        listeners_df['Date'] = pd.to_datetime(listeners_df['Date'])
        
        # Sort by date
        listeners_df = listeners_df.sort_values('Date')
        
        return listeners_df
    except Exception as e:
        st.error(f"Error fetching artist monthly listeners: {str(e)}")
        return pd.DataFrame(columns=['Date', 'Monthly Listeners'])

def get_audience_geography(artist_id):
    """Get audience geography data from ChartMetric API"""
    try:
        # Call the ChartMetric API to get audience geography
        geography_data = chartmetric.get_artist_track_where_people_listen(artist_id=artist_id)
        
        # Convert to DataFrame
        geography_df = pd.DataFrame([
            {
                "country_code": item.code2,
                "country": item.country_name,
                "listeners": item.listeners,
                "population": item.population
            }
            for item in geography_data
        ])
        
        # Calculate percentage
        total_listeners = geography_df['listeners'].sum()
        geography_df['percentage'] = geography_df['listeners'] / total_listeners * 100
        
        # Sort by percentage descending
        geography_df = geography_df.sort_values('percentage', ascending=False)
        
        # Calculate US percentage
        us_percentage = 0
        if 'US' in geography_df['country_code'].values:
            us_percentage = geography_df[geography_df['country_code'] == 'US']['percentage'].iloc[0]
        
        return geography_df, us_percentage
    except Exception as e:
        st.error(f"Error fetching audience geography: {str(e)}")
        # Return empty DataFrame with expected columns
        return pd.DataFrame(columns=['country_code', 'country', 'listeners', 'population', 'percentage']), 100

def render_chartmetric_tab():
    """Render the ChartMetric API tab"""
    st.header("ChartMetric API Search")
    
    # ===== TAB 1: SPOTIFY MONTHLY LISTENERS ANALYSIS =====
    # 1. DATA INPUT (ChartMetric ID instead of file upload)
    st.subheader("Search for an Artist by Chartmetric ID")
    artist_id_input = st.text_input("Artist ID", key="chartmetric_artist_id_input")
    search_id_button = st.button("Search", key="chartmetric_search_button")

    # Initialize session state for artist data
    if "artist_monthly_listeners_df" not in st.session_state:
        st.session_state.artist_monthly_listeners_df = None
        
    if "selected_artist" not in st.session_state:
        st.session_state.selected_artist = None

    # Search for artist by ID when button is clicked
    if search_id_button and artist_id_input:
        try:
            artist_id = int(artist_id_input)
            with st.spinner(f"Fetching artist with ID: {artist_id}..."):
                artist_result = get_artist_by_id(artist_id)
                if artist_result:
                    st.session_state.selected_artist = artist_result
                    st.success(f"Found artist: {artist_result['name']}")
                    
                    # ===== DATA PROCESSING SECTION =====
                    # 1. DATA LOADING: Automatically fetch monthly listeners data
                    with st.spinner("Loading monthly listeners data..."):
                        artist_monthly_listeners_df = get_artist_monthly_listeners(artist_id)
                        
                        if not artist_monthly_listeners_df.empty:
                            st.success(f"Successfully retrieved {len(artist_monthly_listeners_df)} data points")
                            
                            # Save the dataframe to session state for later use
                            st.session_state.artist_monthly_listeners_df = artist_monthly_listeners_df
                            
                            # Save to CSV if wanted
                            if st.button("Download Monthly Listeners Data as CSV", key="chartmetric_download_button"):
                                csv = artist_monthly_listeners_df.to_csv(index=False)
                                st.download_button(
                                    label="Download CSV",
                                    data=csv,
                                    file_name=f"artist_{artist_id}_monthly_listeners.csv",
                                    mime="text/csv",
                                    key="chartmetric_download_csv_button"
                                )
                        else:
                            st.warning("No monthly listener data available for this artist")
                else:
                    st.error(f"No artist found with ID {artist_id}")
        except ValueError:
            st.error("Please enter a valid numeric ID")
    
    # If we have artist data, process and display it
    if st.session_state.artist_monthly_listeners_df is not None and not st.session_state.artist_monthly_listeners_df.empty:
        artist_monthly_listeners_df = st.session_state.artist_monthly_listeners_df
        
        # 2. DATA VALIDATION
        # Check if required columns exist
        if validate_columns(artist_monthly_listeners_df, ['Date', 'Monthly Listeners']):
            # 3. DATA SELECTION
            # Keep only required columns
            columns_to_keep = ['Date', 'Monthly Listeners']
            artist_monthly_listeners_df = select_columns(artist_monthly_listeners_df, columns_to_keep)
            
            # 4. DATE CONVERSION
            # Convert 'Date' column to datetime format with error handling
            artist_monthly_listeners_df, date_issues = convert_to_datetime(artist_monthly_listeners_df, 'Date', dayfirst=True)
            
            # Display any issues with date conversion
            for issue in date_issues:
                if "Failed to convert" in issue:
                    st.error(issue)
                else:
                    st.warning(f"{issue} Please check your data.")
            
            # 5. INITIAL DECAY ANALYSIS
            # Calculate decay rates using our dedicated function - just like file_uploader_tab
            mldr = calculate_monthly_listener_decay_rate(artist_monthly_listeners_df)
            
            # For visualization and UI, we still need the full analysis
            decay_analysis = analyze_listener_decay(artist_monthly_listeners_df)

            # ===== UI COMPONENTS SECTION =====
            # Get normalized dates for consistent month-based calculations
            min_date = decay_analysis['normalized_start_date'].date()  # Convert to datetime.date
            max_date = decay_analysis['normalized_end_date'].date()    # Convert to datetime.date
            
            # 6. DATE RANGE SELECTION
            st.write("Select Date Range:")
            start_date, end_date = st.slider(
                "Select date range",
                min_value=min_date,
                max_value=max_date,
                value=(min_date, max_date),
                format="YYYY-MM-DD",
                key="chartmetric_date_slider"  # Add a unique key to prevent duplicate ID error
            )

            # Convert slider values to Timestamp objects
            start_date = pd.Timestamp(start_date)
            end_date = pd.Timestamp(end_date)

            # Update the decay calculation if the user changes the date range
            if start_date != decay_analysis['normalized_start_date'] or end_date != decay_analysis['normalized_end_date']:
                mldr = calculate_monthly_listener_decay_rate(artist_monthly_listeners_df, start_date, end_date)
                decay_analysis = analyze_listener_decay(artist_monthly_listeners_df, start_date, end_date)
            
            # Extract required data from the analysis for visualization
            date_filtered_listener_data = decay_analysis['date_filtered_listener_data']
            fitted_decay_parameters = decay_analysis['fitted_decay_parameters']
            normalized_start_date = decay_analysis['normalized_start_date']
            normalized_end_date = decay_analysis['normalized_end_date']
            
            # ===== RESULTS DISPLAY SECTION =====
            # 8. SHOW METRICS
            st.write(f'Exponential decay rate: {mldr}')
            
            # Always show normalized dates for transparency
            st.write(f'Normalized date range used: {normalized_start_date.strftime("%Y-%m-%d")} to {normalized_end_date.strftime("%Y-%m-%d")}')
            
            # 9. VISUALIZATION
            fig, ax = plt.subplots(figsize=(10, 4))
            # Plot the moving average
            ax.plot(date_filtered_listener_data['Date'], date_filtered_listener_data['4_Week_MA'], label='Moving Average', color='tab:blue', linewidth=2)
            # Plot the fitted decay curve using pre-calculated parameters
            ax.plot(date_filtered_listener_data['Date'], exponential_decay(date_filtered_listener_data['Months'], *fitted_decay_parameters), 
                   label='Fitted Decay Curve', color='red', linestyle='--')
            
            # Plot formatting and styling
            ax.set_xlabel('Date', fontsize=12)
            ax.set_ylabel('Monthly Listeners', fontsize=12)
            ax.set_title(f'Moving Average and Exponential Decay', fontsize=14, weight='bold')
            ax.legend()
            ax.set_ylim(bottom=0)
            plt.xticks(rotation=45)
            
            # Visual enhancements
            fig.patch.set_visible(False)
            ax.set_facecolor('none')
            ax.patch.set_alpha(0)
            plt.tight_layout()
            
            # Display the plot
            st.pyplot(fig)
        else:
            st.error("The API data does not contain the required columns 'Date' and 'Monthly Listeners'.")
    
    # Show a message if no artist is selected yet
    if st.session_state.artist_monthly_listeners_df is None or st.session_state.artist_monthly_listeners_df.empty:
        st.info("Enter a Chartmetric artist ID to begin analysis")
    
    return 