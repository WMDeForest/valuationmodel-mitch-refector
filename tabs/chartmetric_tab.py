import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Import chartmetric service
from services.chartmetric_services import chartmetric_service as chartmetric
from services.chartmetric_services.dto import ArtistStateCampareListner, TrackSpotifyState, CountryListeners

# Import utility functions
from utils.data_processing import (
    calculate_period_streams,
    calculate_months_since_release,
    calculate_monthly_stream_averages,
    extract_track_metrics
)
from utils.decay_models import (
    exponential_decay,
    analyze_listener_decay,
    calculate_monthly_listener_decay_rate
)
from utils.population_utils.country_code_to_name import country_code_to_name

def search_artist(query):
    """Search for an artist using ChartMetric API"""
    # This is a placeholder - would need to implement ChartMetric search API
    st.warning("Artist search API not yet implemented")
    # Return demo artist for now
    return {
        "id": 12345, 
        "name": "Sample Artist",
        "image_url": "https://example.com/image.jpg"
    }

def search_tracks(artist_id):
    """Get tracks for an artist using ChartMetric API"""
    # This is a placeholder - would need to implement ChartMetric artist tracks API
    st.warning("Track search API not yet implemented")
    # Return demo tracks for now
    return [
        {"id": 54321, "name": "Sample Track 1"},
        {"id": 54322, "name": "Sample Track 2"}
    ]

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

def get_track_stream_data(track_id):
    """Get track streaming data from ChartMetric API"""
    try:
        # Get track details first
        track_details = chartmetric.get_track_detail(track_id=track_id)
        
        # Get track streaming data
        streams_data = chartmetric.get_track_sp_streams_campare(track_id=track_id)
        
        # Convert to DataFrame
        streams_df = pd.DataFrame([
            {"Date": item.timestp, "CumulativeStreams": item.value}
            for item in streams_data
        ])
        
        # Ensure Date is datetime format
        streams_df['Date'] = pd.to_datetime(streams_df['Date'])
        
        # Sort by date
        streams_df = streams_df.sort_values('Date')
        
        return streams_df, track_details
    except Exception as e:
        st.error(f"Error fetching track streaming data: {str(e)}")
        return pd.DataFrame(columns=['Date', 'CumulativeStreams']), None

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
    
    # Artist search section
    st.subheader("Search for an Artist")
    artist_query = st.text_input("Artist Name")
    search_button = st.button("Search")
    
    # Initialize session state for selected artist and tracks
    if "selected_artist" not in st.session_state:
        st.session_state.selected_artist = None
    
    if "selected_tracks" not in st.session_state:
        st.session_state.selected_tracks = []
    
    # Search for artist when button is clicked
    if search_button and artist_query:
        with st.spinner("Searching for artist..."):
            # This would call the ChartMetric API to search for the artist
            artist_result = search_artist(artist_query)
            if artist_result:
                st.session_state.selected_artist = artist_result
                st.success(f"Found artist: {artist_result['name']}")
            else:
                st.error(f"No artists found matching '{artist_query}'")
    
    # If an artist is selected, show their monthly listeners and tracks
    if st.session_state.selected_artist:
        artist = st.session_state.selected_artist
        st.subheader(f"Artist: {artist['name']}")
        
        # Get and display monthly listeners
        with st.spinner("Loading artist monthly listeners..."):
            listeners_df = get_artist_monthly_listeners(artist['id'])
            
            if not listeners_df.empty:
                # Calculate the decay rate directly
                mldr = calculate_monthly_listener_decay_rate(listeners_df)
                
                # Get the full analysis for visualization
                decay_analysis = analyze_listener_decay(listeners_df)
                
                # Show metrics
                st.write(f"Exponential decay rate: {mldr}")
                
                # Display chart
                fig, ax = plt.subplots(figsize=(10, 4))
                ax.plot(decay_analysis['date_filtered_listener_data']['Date'], 
                        decay_analysis['date_filtered_listener_data']['4_Week_MA'], 
                        label='Moving Average', color='tab:blue', linewidth=2)
                
                ax.plot(decay_analysis['date_filtered_listener_data']['Date'], 
                        exponential_decay(decay_analysis['date_filtered_listener_data']['Months'], 
                                          *decay_analysis['fitted_decay_parameters']), 
                        label='Fitted Decay Curve', color='red', linestyle='--')
                
                # Plot formatting
                ax.set_xlabel('Date', fontsize=12)
                ax.set_ylabel('Monthly Listeners', fontsize=12)
                ax.set_title(f'Moving Average and Exponential Decay', fontsize=14, weight='bold')
                ax.legend()
                ax.set_ylim(bottom=0)
                plt.xticks(rotation=45)
                plt.tight_layout()
                
                # Display the plot
                st.pyplot(fig)
            else:
                st.warning("No monthly listener data available for this artist")
        
        # Get and display tracks
        with st.spinner("Loading artist tracks..."):
            # This would call the ChartMetric API to get tracks for the artist
            tracks = search_tracks(artist['id'])
            if tracks:
                # Let user select tracks to analyze
                track_options = {f"{track['name']} (ID: {track['id']})": track for track in tracks}
                selected_track_names = st.multiselect("Select Tracks to Analyze", list(track_options.keys()))
                
                # Store selected tracks
                st.session_state.selected_tracks = [track_options[name] for name in selected_track_names]
            else:
                st.warning("No tracks found for this artist")
        
        # Get audience geography if artist is selected
        with st.spinner("Loading audience geography..."):
            geography_df, us_percentage = get_audience_geography(artist['id'])
            
            if not geography_df.empty:
                st.subheader("Audience Geography")
                st.write(f"US Listeners: {us_percentage:.2f}%")
                
                # Display top countries
                st.dataframe(geography_df.head(10))
            else:
                st.warning("No audience geography data available for this artist")
        
        # Track analysis section
        if st.session_state.selected_tracks:
            st.subheader("Track Analysis")
            
            # Analyze each selected track
            for track in st.session_state.selected_tracks:
                st.write(f"### {track['name']}")
                
                # Get track streaming data
                with st.spinner(f"Loading data for {track['name']}..."):
                    streams_df, track_details = get_track_stream_data(track['id'])
                    
                    if not streams_df.empty and track_details:
                        # Extract track metrics
                        track_metrics = extract_track_metrics(
                            track_data_df=streams_df,
                            track_name=track_details.name
                        )
                        
                        # Display track metrics
                        st.write(f"Release Date: {track_details.release_date}")
                        st.write(f"Total Streams: {track_metrics['total_historical_track_streams']:,}")
                        st.write(f"Last 30 Days: {track_metrics['track_streams_last_30days']:,}")
                        st.write(f"Last 90 Days: {track_metrics['track_streams_last_90days']:,}")
                        st.write(f"Last 365 Days: {track_metrics['track_streams_last_365days']:,}")
                        
                        # Display stream chart
                        fig, ax = plt.subplots(figsize=(10, 4))
                        ax.plot(streams_df['Date'], streams_df['CumulativeStreams'], color='tab:blue')
                        ax.set_xlabel('Date', fontsize=12)
                        ax.set_ylabel('Cumulative Streams', fontsize=12)
                        ax.set_title(f'Cumulative Streams for {track_details.name}', fontsize=14, weight='bold')
                        plt.xticks(rotation=45)
                        plt.tight_layout()
                        
                        # Display the plot
                        st.pyplot(fig)
                    else:
                        st.warning(f"No streaming data available for {track['name']}")
    
    # Show a message if no artist is selected yet
    if not st.session_state.selected_artist:
        st.info("Search for an artist to begin analysis")
    
    return 