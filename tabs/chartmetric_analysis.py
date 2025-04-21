import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import io

# Import ChartMetric service
from services.chartmetric_services import chartmetric_service as chartmetric

# Import process_and_visualize_track_data function from tab 1
from tabs.process_visualize_track_data import process_and_visualize_track_data

def show_chartmetric_analysis():
    """
    Tab 2: ChartMetric Analysis Tab
    
    Fetches data from ChartMetric API based on artist ID and track ID inputs,
    then passes the formatted data to process_and_visualize_track_data function.
    """
    st.title("ChartMetric Analysis")
    st.write("Use ChartMetric API to analyze artist and track data")
    
    # === INPUT SECTION ===
    st.header("Input ChartMetric IDs")
    
    col1, col2 = st.columns(2)
    with col1:
        artist_id = st.text_input("ChartMetric Artist ID", help="Enter the ChartMetric ID for the artist")
    with col2:
        track_id = st.text_input("ChartMetric Track ID", help="Enter the ChartMetric ID for the track")
    
    # Run button
    if st.button("Fetch Data and Run Analysis"):
        if not artist_id or not track_id:
            st.error("Please enter both Artist ID and Track ID")
            return
        
        with st.spinner("Fetching data from ChartMetric API..."):
            try:
                # === FETCH DATA FROM CHARTMETRIC API ===
                
                # 1. Get artist monthly listeners data
                artist_monthly_listeners = fetch_artist_monthly_listeners(artist_id)
                
                # 2. Get track streaming data
                track_streaming_data = fetch_track_streaming_data(track_id)
                
                # 3. Get audience geography data
                audience_geography = fetch_audience_geography(artist_id)
                
                # Check if we have the minimum required data
                if artist_monthly_listeners is None or track_streaming_data is None:
                    st.error("Failed to fetch essential data. Please check the IDs and try again.")
                    return
                
                # Convert data into the format expected by process_and_visualize_track_data
                artist_monthly_listeners_df = format_artist_monthly_listeners(artist_monthly_listeners)
                catalog_file_data = format_track_streaming_data(track_streaming_data, track_id)
                audience_geography_data = format_audience_geography(audience_geography)
                
                # Get track name for display
                try:
                    track_details = chartmetric.get_track_detail(track_id=int(track_id))
                    track_name = track_details.name if hasattr(track_details, 'name') else f"Track ID: {track_id}"
                except Exception:
                    track_name = f"Track ID: {track_id}"
                
                # Display success message with fetched data summary
                st.success(f"Successfully fetched data for track: {track_name}")
                
                # Pass the formatted data to process_and_visualize_track_data
                process_and_visualize_track_data(
                    artist_monthly_listeners_df=artist_monthly_listeners_df,
                    catalog_file_data=catalog_file_data,
                    audience_geography_data=audience_geography_data,
                    ownership_data=None  # No ownership data from API
                )
                
            except Exception as e:
                st.error(f"Error fetching data: {str(e)}")
                st.error("Please check your API credentials and input IDs.")

def fetch_artist_monthly_listeners(artist_id):
    """
    Fetch artist monthly listeners data from ChartMetric API
    
    Parameters:
    -----------
    artist_id : str
        ChartMetric artist ID
    
    Returns:
    --------
    list
        List of monthly listener data points from ChartMetric API
    """
    try:
        # Get Spotify stats (monthly listeners) for the artist
        monthly_listeners_data = chartmetric.get_artist_spotify_stats(artist_id=int(artist_id))
        
        if not monthly_listeners_data:
            st.warning("No monthly listeners data found for this artist.")
            return None
            
        return monthly_listeners_data
    except Exception as e:
        st.error(f"Error fetching artist monthly listeners: {str(e)}")
        return None

def fetch_track_streaming_data(track_id):
    """
    Fetch track streaming data from ChartMetric API
    
    Parameters:
    -----------
    track_id : str
        ChartMetric track ID
    
    Returns:
    --------
    list
        List of streaming data points from ChartMetric API
    """
    try:
        # Get Spotify streams data for the track
        streaming_data = chartmetric.get_track_sp_streams_campare(track_id=int(track_id))
        
        if not streaming_data:
            st.warning("No streaming data found for this track.")
            return None
            
        return streaming_data
    except Exception as e:
        st.error(f"Error fetching track streaming data: {str(e)}")
        return None

def fetch_audience_geography(artist_id):
    """
    Fetch audience geography data from ChartMetric API
    
    Parameters:
    -----------
    artist_id : str
        ChartMetric artist ID
    
    Returns:
    --------
    list
        List of country data with code2 and listeners only
    """
    try:
        # Create params dictionary with required parameters
        params = {
            # Required parameter - use a date from before Aug 12, 2024 to get top 50 cities
            'since': '2024-08-11',
            # Get maximum countries
            'limit': 50
        }
        
        # Use the fixed API method
        geography_data = chartmetric.get_artist_track_where_people_listen(
            artist_id=int(artist_id),
            params=params
        )
        
        if geography_data and len(geography_data) > 0:
            st.success(f"Successfully fetched audience geography data for {len(geography_data)} countries")
            return geography_data
        else:
            st.warning("No country data found in the API response")
            return None
            
    except Exception as e:
        st.error(f"Error fetching audience geography data: {str(e)}")
        # Return None on error, format_audience_geography will handle it
        return None

def format_artist_monthly_listeners(monthly_listeners_data):
    """
    Format artist monthly listeners data to match the format expected by process_and_visualize_track_data
    
    Parameters:
    -----------
    monthly_listeners_data : list
        List of monthly listener data points from ChartMetric API
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame with columns:
        - Date: in DD/MM/YYYY format
        - Monthly Listeners: count of monthly listeners per date
    """
    # Convert API data to DataFrame
    df = pd.DataFrame(monthly_listeners_data)
    
    # Rename columns to match expected format
    df = df.rename(columns={
        'timestp': 'Date',
        'value': 'Monthly Listeners'
    })
    
    # Convert date format from 'YYYY-MM-DD' to 'DD/MM/YYYY'
    df['Date'] = pd.to_datetime(df['Date']).dt.strftime('%d/%m/%Y')
    
    return df

def format_track_streaming_data(streaming_data, track_id):
    """
    Format track streaming data to match the format expected by process_and_visualize_track_data
    
    Parameters:
    -----------
    streaming_data : list
        List of streaming data points from ChartMetric API
    track_id : str
        ChartMetric track ID
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame with columns:
        - Date: in DD/MM/YYYY format
        - Value/CumulativeStreams: total cumulative streams as of each date
        - Track Name: name identifier for each track
    """
    # Convert API data to DataFrame
    df = pd.DataFrame(streaming_data)
    
    # Rename columns to match expected format
    df = df.rename(columns={
        'timestp': 'Date',
        'value': 'Value'
    })
    
    # Get track details to get the name
    try:
        track_details = chartmetric.get_track_detail(track_id=int(track_id))
        track_name = track_details.get('name', f"Track ID: {track_id}")
    except:
        track_name = f"Track ID: {track_id}"
    
    # Add track name column
    df['Track Name'] = track_name
    
    # Convert date format from 'YYYY-MM-DD' to 'DD/MM/YYYY'
    df['Date'] = pd.to_datetime(df['Date']).dt.strftime('%d/%m/%Y')
    
    # Sort by date to ensure correct cumulative calculation
    df = df.sort_values('Date')
    
    # Convert streaming data to cumulative streams if it's not already
    # Check if the values are already cumulative by seeing if they always increase
    is_cumulative = (df['CumulativeStreams'].diff().dropna() >= 0).all()
    
    if not is_cumulative:
        # Convert to cumulative by calculating cumulative sum
        df['CumulativeStreams'] = df['CumulativeStreams'].cumsum()
    
    return df

def format_audience_geography(geography_data):
    """
    Format audience geography data to match the format expected by process_and_visualize_track_data
    
    Parameters:
    -----------
    geography_data : list
        List of dictionary items with code2 and listeners
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame with columns:
        - Country: country code (ISO 2-letter format)
        - Listeners: number of listeners per country
    """
    # If data is None, return a default DataFrame
    if geography_data is None:
        default_data = [
            {"Country": "US", "Listeners": 100000},
            {"Country": "GB", "Listeners": 50000}
        ]
        st.warning("Using default country distribution - no geography data available")
        return pd.DataFrame(default_data)
    
    # Convert the list of dictionaries to DataFrame
    df = pd.DataFrame(geography_data)
    
    # Rename columns to match expected format
    df = df.rename(columns={
        'code2': 'Country',
        'listeners': 'Listeners'
    })
    
    # Ensure we have the required columns
    if 'Country' not in df.columns or 'Listeners' not in df.columns:
        default_data = [
            {"Country": "US", "Listeners": 100000},
            {"Country": "GB", "Listeners": 50000}
        ]
        st.warning("Invalid data format. Using default country distribution.")
        return pd.DataFrame(default_data)
    
    return df 