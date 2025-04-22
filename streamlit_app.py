#!/usr/bin/env python
# coding: utf-8

import streamlit as st
import pandas as pd
import base64
from datetime import datetime, timedelta
import io

# Import core functionality
from utils.process_visualize_track_data import process_and_visualize_track_data

# Import ChartMetric service
from services.chartmetric_services import chartmetric_service as chartmetric

# Import country code to name utility
from utils.population_utils.country_code_to_name import country_code_to_name

# ===== APP INTERFACE SETUP =====
st.title('Valuation Model_mitch_refactor')
st.write("Analyze track valuation using historical streaming data")

# ===== DATA SOURCE SELECTION =====
data_source = st.radio(
    "Select Data Source:",
    ["CSV Upload", "ChartMetric API"],
    horizontal=True,
    key="data_source_selector"
)

# Initialize containers for data
artist_monthly_listeners_df = None
catalog_file_data = None
audience_geography_data = None
ownership_data = None
analysis_ready = False

# ===== CSV UPLOAD SECTION =====
if data_source == "CSV Upload":
    st.header("Upload Your Data Files")
    
    # 1. Artist Monthly Listeners Data
    uploaded_file = st.file_uploader("Artist Monthly Spotify Listeners", type="csv", key="csv_artist_upload")
    if uploaded_file is not None:
        artist_monthly_listeners_df = pd.read_csv(uploaded_file)
        st.success(f"Successfully loaded artist data with {len(artist_monthly_listeners_df)} entries")
    
    # 2. Track Catalog Data
    uploaded_catalog_file = st.file_uploader("Track Catalog CSV", type=["csv"], 
                                           help="Upload a single CSV containing data for multiple tracks.",
                                           key="csv_catalog_upload")
    if uploaded_catalog_file is not None:
        catalog_file_data = uploaded_catalog_file
        st.success("Successfully loaded catalog file")
    
    # 3. Audience Geography Data
    uploaded_file_audience_geography = st.file_uploader("Audience Geography", type=["csv"], 
                                                      key="csv_geo_upload")
    if uploaded_file_audience_geography is not None:
        audience_geography_data = uploaded_file_audience_geography
        st.success("Successfully loaded audience geography data")
    
    # 4. Ownership Data
    uploaded_file_ownership = st.file_uploader("MLC Claimed and Song Ownership", type="csv",
                                             key="csv_ownership_upload")
    if uploaded_file_ownership is not None:
        ownership_data = uploaded_file_ownership
        st.success("Successfully loaded ownership data")
    
    # Check if minimum required data is available
    if artist_monthly_listeners_df is not None and catalog_file_data is not None:
        analysis_ready = True
    
# ===== CHARTMETRIC API SECTION =====
elif data_source == "ChartMetric API":
    st.header("Enter ChartMetric IDs")
    
    col1, col2 = st.columns(2)
    with col1:
        artist_id = st.text_input("ChartMetric Artist ID", help="Enter the ChartMetric ID for the artist", key="api_artist_id")
    with col2:
        track_id = st.text_input("ChartMetric Track ID", help="Enter the ChartMetric ID for the track", key="api_track_id")
    
    # Fetch button for API data
    if st.button("Fetch Data from ChartMetric", key="fetch_api_button"):
        if not artist_id or not track_id:
            st.error("Please enter both Artist ID and Track ID")
        else:
            with st.spinner("Fetching data from ChartMetric API..."):
                try:
                    # === FETCH DATA FROM CHARTMETRIC API ===
                    
                    # Helper functions to fetch and format data from ChartMetric
                    def fetch_artist_monthly_listeners(artist_id):
                        """Fetch artist monthly listeners data from ChartMetric API"""
                        try:
                            monthly_listeners_data = chartmetric.get_artist_spotify_stats(artist_id=int(artist_id))
                            if not monthly_listeners_data:
                                st.warning("No monthly listeners data found for this artist.")
                                return None
                            return monthly_listeners_data
                        except Exception as e:
                            st.error(f"Error fetching artist monthly listeners: {str(e)}")
                            return None

                    def fetch_track_streaming_data(track_id):
                        """Fetch track streaming data from ChartMetric API"""
                        try:
                            streaming_data = chartmetric.get_track_sp_streams_campare(track_id=int(track_id))
                            if not streaming_data:
                                st.warning("No streaming data found for this track.")
                                return None
                                
                            # Display the first 15 lines of raw track data from the API
                            st.subheader("First 15 lines of track data from ChartMetric API:")
                            
                            # Create a formatted display of the first 15 records
                            data_sample = streaming_data[:15]
                            
                            # Create a table for better readability
                            df_sample = pd.DataFrame([
                                {
                                    'Date': item.timestp,
                                    'Value': item.value,
                                    'Daily Diff': item.daily_diff,
                                    'Interpolated': item.interpolated
                                }
                                for item in data_sample
                            ])
                            
                            st.dataframe(df_sample)
                            st.write(f"Total records: {len(streaming_data)}")
                                
                            return streaming_data
                        except Exception as e:
                            st.error(f"Error fetching track streaming data: {str(e)}")
                            return None

                    def fetch_audience_geography(artist_id):
                        """Fetch audience geography data from ChartMetric API"""
                        try:
                            # Create params dictionary with required parameters
                            params = {
                                # Use a date from before Aug 12, 2024 to get top 50 cities
                                'since': '2024-08-11',
                                # Get maximum countries
                                'limit': 50
                            }
                            
                            # Use the API method
                            geography_data = chartmetric.get_artist_track_where_people_listen(
                                artist_id=int(artist_id),
                                params=params
                            )
                            
                            if geography_data and len(geography_data) > 0:
                                return geography_data
                            else:
                                st.warning("No country data found in the API response")
                                return None
                        except Exception as e:
                            st.error(f"Error fetching audience geography data: {str(e)}")
                            return None

                    # Standardized date conversion function to avoid redundancy
                    def convert_date_format(df, date_column='timestp'):
                        """Standardized date conversion to consistent format"""
                        # First ensure dates are in datetime format for proper sorting
                        df[date_column] = pd.to_datetime(df[date_column])
                        # Sort by the datetime column
                        df = df.sort_values(date_column)
                        # Then convert to required string format after sorting
                        df[date_column] = df[date_column].dt.strftime('%d/%m/%Y')
                        return df

                    # Formatting functions
                    def format_artist_monthly_listeners(monthly_listeners_data):
                        """Format artist monthly listeners data to match expected format"""
                        # Convert API data to DataFrame
                        df = pd.DataFrame(monthly_listeners_data)
                        
                        # Sort and convert dates first
                        df = convert_date_format(df)
                        
                        # Rename columns to match expected format
                        df = df.rename(columns={
                            'timestp': 'Date',
                            'value': 'Monthly Listeners'
                        })
                        
                        return df

                    def format_track_streaming_data(streaming_data, track_id):
                        """Format track streaming data to match expected format"""
                        # Convert API data to DataFrame
                        df = pd.DataFrame(streaming_data)
                        
                        # Debug: Show DataFrame at initial creation
                        st.write("DEBUG - Initial DataFrame from API data:")
                        st.write(df.head(3))
                        st.write(f"Value column stats - Min: {df['value'].min()}, Max: {df['value'].max()}, Mean: {df['value'].mean():.2f}")
                        
                        # Sort and convert dates first
                        df = convert_date_format(df)
                        
                        # Rename columns to match expected format
                        df = df.rename(columns={
                            'timestp': 'Date',
                            'value': 'CumulativeStreams'
                        })
                        
                        # Get track details to get the name and release date
                        track_release_date = None
                        try:
                            track_details = chartmetric.get_track_detail(track_id=int(track_id))
                            track_name = track_details.name
                            
                            # Store the track ID in session state
                            st.session_state.current_chartmetric_track_id = track_id
                            
                            # Get and store the actual release date if available
                            if track_details.release_date:
                                # Store the original API release date
                                release_date = track_details.release_date
                                
                                # Format for our application (DD/MM/YYYY)
                                release_date_dt = pd.to_datetime(release_date)
                                track_release_date = release_date_dt.strftime("%d/%m/%Y")
                                
                                # Store in session state for historical value calculation
                                st.session_state.track_release_date = track_release_date
                                
                                # Log for debugging
                                st.write(f"Track release date from API: {release_date} (formatted: {track_release_date})")
                                
                        except Exception as e:
                            track_name = f"Track ID: {track_id}"
                            st.warning(f"Could not fetch complete track details: {str(e)}")
                        
                        # Add track name column
                        df['Track Name'] = track_name
                        
                        # Debug: Show DataFrame after formatting
                        st.write("DEBUG - Final DataFrame after formatting:")
                        st.write(df.head(3))
                        st.write(f"Value column stats - Min: {df['CumulativeStreams'].min()}, Max: {df['CumulativeStreams'].max()}, Mean: {df['CumulativeStreams'].mean():.2f}")
                        
                        return df

                    def format_audience_geography(geography_data):
                        """Format audience geography data to match expected format"""
                        # If data is None, return a default DataFrame
                        if geography_data is None:
                            default_data = [
                                {"Country": "United States", "Listeners": 100000},
                                {"Country": "United Kingdom", "Listeners": 50000}
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
                                {"Country": "United States", "Listeners": 100000},
                                {"Country": "United Kingdom", "Listeners": 50000}
                            ]
                            st.warning("Invalid data format. Using default country distribution.")
                            return pd.DataFrame(default_data)
                        
                        # Convert two-letter country codes to full country names
                        df['Country'] = df['Country'].apply(country_code_to_name)
                        
                        # Debug output to verify country code conversion
                        st.write("DEBUG - Audience geography after country code conversion:")
                        st.write(df.head(5))
                        
                        return df
                    
                    # 1. Get artist monthly listeners data
                    artist_monthly_listeners = fetch_artist_monthly_listeners(artist_id)
                    
                    # 2. Get track streaming data
                    track_streaming_data = fetch_track_streaming_data(track_id)
                    
                    # 3. Get audience geography data
                    audience_geography = fetch_audience_geography(artist_id)
                    
                    # Check if we have the minimum required data
                    if artist_monthly_listeners is None or track_streaming_data is None:
                        st.error("Failed to fetch essential data. Please check the IDs and try again.")
                    else:
                        # Convert data into the format expected by process_and_visualize_track_data
                        artist_monthly_listeners_df = format_artist_monthly_listeners(artist_monthly_listeners)
                        track_df = format_track_streaming_data(track_streaming_data, track_id)
                        audience_geography_df = format_audience_geography(audience_geography)
                        
                        # Debug: Examine the DataFrame before processing
                        st.write("DEBUG - Track DataFrame before processing:")
                        st.write(track_df.head(3))
                        st.write(f"Shape: {track_df.shape}, Columns: {track_df.columns.tolist()}")
                        
                        # Add more diagnostic data to match CSV display for comparison
                        st.write("----------------------------------------------------")
                        st.write("🔍 CHARTMETRIC API DATA DIAGNOSTIC INFO:")
                        st.write(f"  • Track Name: {track_df['Track Name'].iloc[0] if 'Track Name' in track_df.columns else 'Unknown'}")
                        st.write(f"  • Total Historical Streams: {track_df['CumulativeStreams'].iloc[-1]:,}")
                        
                        # Show both release date and earliest data date
                        try:
                            track_details = chartmetric.get_track_detail(track_id=int(track_id))
                            if track_details.release_date:
                                release_date = track_details.release_date
                                st.write(f"  • Actual Track Release Date (API): {release_date}")
                        except Exception as e:
                            pass
                            
                        st.write(f"  • Earliest Data Date (in dataset): {track_df['Date'].iloc[0]}")
                        
                        # Show audience geography info
                        if isinstance(audience_geography_df, pd.DataFrame) and not audience_geography_df.empty:
                            total_listeners = audience_geography_df['Listeners'].sum()
                            us_listeners = audience_geography_df[audience_geography_df['Country'] == 'United States']['Listeners'].sum() if 'United States' in audience_geography_df['Country'].values else 0
                            us_percentage = (us_listeners / total_listeners) if total_listeners > 0 else 0
                            st.write(f"  • US Listener Percentage: {us_percentage:.2%}")
                            st.write(f"  • Total Countries: {len(audience_geography_df)}")
                        else:
                            st.write("  • No audience geography data available, using default US distribution")
                        st.write("----------------------------------------------------")
                        
                        # Store formatted DataFrames directly in session state
                        st.session_state.artist_monthly_listeners_df = artist_monthly_listeners_df
                        st.session_state.catalog_df = track_df
                        st.session_state.audience_geo_df = audience_geography_df
                        
                        # Log what's in session state for debugging
                        st.write("---")
                        st.write("📊 SESSION STATE DATA STORED FOR VALUATION:")
                        st.write(f"• Track ID: {st.session_state.get('current_chartmetric_track_id')}")
                        
                        # List all release date keys
                        release_date_keys = [key for key in st.session_state.keys() if key.startswith('track_release_date_')]
                        for key in release_date_keys:
                            st.write(f"• {key}: {st.session_state[key]}")
                        st.write("---")
                        
                        # For processing, convert DataFrame to BytesIO only when needed
                        catalog_file_data = io.BytesIO(track_df.to_csv(index=False).encode())
                        
                        if isinstance(audience_geography_df, pd.DataFrame):
                            audience_geography_data = io.BytesIO(audience_geography_df.to_csv(index=False).encode())
                        
                        # Pass a custom data processor that will override the earliest_track_date
                        # when extract_track_metrics is called
                        def custom_extract_track_metrics(track_data_df, track_name=None):
                            """Custom wrapper for extract_track_metrics that uses the actual release date"""
                            from utils.data_processing import extract_track_metrics as original_extract_track_metrics
                            
                            # Get the original metrics
                            metrics = original_extract_track_metrics(track_data_df, track_name)
                            
                            # Get track ID from session state (set during API data fetch)
                            track_id = st.session_state.get('current_chartmetric_track_id')
                            
                            if track_id:
                                release_date_key = f"track_release_date_{track_id}"
                                
                                # If we have a corresponding release date, use it
                                if release_date_key in st.session_state:
                                    release_date = st.session_state[release_date_key]
                                    st.write(f"Overriding earliest track date from {metrics['earliest_track_date']} to {release_date} (using stored release date)")
                                    metrics['earliest_track_date'] = release_date
                                    
                                    # Recalculate months_since_release_total based on the new date
                                    from utils.data_processing import calculate_months_since_release
                                    metrics['months_since_release_total'] = calculate_months_since_release(release_date)
                            
                            return metrics
                        
                        # Store the custom function in session state
                        st.session_state.custom_extract_track_metrics = custom_extract_track_metrics
                        
                        # Get track name for display
                        track_name = track_df['Track Name'].iloc[0] if 'Track Name' in track_df.columns else f"Track ID: {track_id}"
                        
                        # Set flag to indicate data is ready
                        st.session_state.chartmetric_data_ready = True
                        
                        # Display success message 
                        st.success(f"Successfully fetched data for track: {track_name}")
                        analysis_ready = True
                        
                except Exception as e:
                    st.error(f"Error fetching data: {str(e)}")
    
    # Check if we have data in session state from a previous fetch
    if 'chartmetric_data_ready' in st.session_state and st.session_state.chartmetric_data_ready:
        # Retrieve data from session state
        if 'artist_monthly_listeners_df' in st.session_state:
            artist_monthly_listeners_df = st.session_state.artist_monthly_listeners_df
        
        if 'catalog_df' in st.session_state:
            # Convert DataFrame to BytesIO only when needed for processing
            catalog_df = st.session_state.catalog_df
            catalog_file_data = io.BytesIO(catalog_df.to_csv(index=False).encode())
        
        if 'audience_geo_df' in st.session_state:
            audience_geo_df = st.session_state.audience_geo_df
            audience_geography_data = io.BytesIO(audience_geo_df.to_csv(index=False).encode())
        
        # Set analysis ready flag
        analysis_ready = True

# ===== ANALYSIS SECTION =====
# Only show the Run Analysis button if we have the minimum required data
if analysis_ready:
    st.header("Analysis")
    
    # Automatically run the analysis when data is ready
    if data_source == "ChartMetric API" and 'custom_extract_track_metrics' in st.session_state:
        # Use our custom function that will override the earliest track date with the real release date
        from utils.data_processing import extract_track_metrics
        
        # Temporarily replace the extract_track_metrics function with our custom one
        import utils.data_processing
        original_extract_track_metrics = utils.data_processing.extract_track_metrics
        utils.data_processing.extract_track_metrics = st.session_state.custom_extract_track_metrics
        
        # Process with our modified function
        process_and_visualize_track_data(
            artist_monthly_listeners_df=artist_monthly_listeners_df,
            catalog_file_data=catalog_file_data,
            audience_geography_data=audience_geography_data,
            ownership_data=ownership_data
        )
        
        # Restore the original function
        utils.data_processing.extract_track_metrics = original_extract_track_metrics
    else:
        # Use the standard function for CSV data
        process_and_visualize_track_data(
            artist_monthly_listeners_df=artist_monthly_listeners_df,
            catalog_file_data=catalog_file_data,
            audience_geography_data=audience_geography_data,
            ownership_data=ownership_data
        )
else:
    # If not ready for analysis, show a message
    if data_source == "CSV Upload":
        st.info("Please upload both Artist Monthly Listeners and Track Catalog data to run analysis.")
    else:  # ChartMetric API
        if not st.session_state.get('fetch_api_button', False):
            st.info("Enter Artist ID and Track ID, then click 'Fetch Data from ChartMetric'.")

