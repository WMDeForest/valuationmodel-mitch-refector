#!/usr/bin/env python
# coding: utf-8

import streamlit as st
import pandas as pd
import json
from services.chartmetric_services import chartmetric_service as chartmetric

def calculate_usa_percentage(df):
    """Calculate the percentage of USA listeners"""
    if df.empty:
        return 0.0
    total_listeners = df['listeners'].sum()
    usa_listeners = df[df['code2'] == 'US']['listeners'].sum() if 'US' in df['code2'].values else 0
    return (usa_listeners / total_listeners) * 100 if total_listeners > 0 else 0.0

def process_location_data(location_dict):
    """Process nested location data into a flat list of records"""
    flat_data = []
    if not location_dict:
        return flat_data
        
    for location_name, data_list in location_dict.items():
        if isinstance(data_list, list) and data_list:
            # Take the first entry for each location
            record = data_list[0].copy()
            record['name'] = location_name  # Add the location name
            flat_data.append(record)
    return flat_data

def display_data_section(response_data, title):
    """Display both city and country data with expandable sections"""
    if not response_data:
        st.warning(f"No {title} found in the API response")
        return

    # Show raw response for debugging
    with st.expander("Debug - View Raw API Response"):
        st.json(response_data)

    try:
        # Extract the obj containing cities and countries
        data = response_data.get('obj', {}) if isinstance(response_data, dict) else {}
        
        if not data:
            st.error("No 'obj' data found in the API response")
            return
            
        # Check if cities and countries exist
        has_cities = 'cities' in data and data['cities']
        has_countries = 'countries' in data and data['countries']
        
        if not has_cities and not has_countries:
            st.error("No cities or countries data found in the API response")
            return
        
        # Process cities data
        cities_dict = data.get('cities', {})
        cities_data = process_location_data(cities_dict)
        
        # Process countries data
        countries_dict = data.get('countries', {})
        countries_data = process_location_data(countries_dict)
        
        # Convert to DataFrames
        cities_df = pd.DataFrame(cities_data) if cities_data else pd.DataFrame()
        countries_df = pd.DataFrame(countries_data) if countries_data else pd.DataFrame()
        
        # Display Cities Data
        st.write("\n📍 Cities Data")
        if not cities_df.empty:
            st.write(f"Total cities: {len(cities_df)}")
            total_city_listeners = cities_df['listeners'].sum() if 'listeners' in cities_df.columns else 0
            st.write(f"Total city listeners: {total_city_listeners:,}")
            
            # Create expandable section for raw cities data
            with st.expander("View Cities Data"):
                # Select relevant columns for cities
                city_columns = ['name', 'region', 'code2', 'listeners', 'population', 
                              'lat', 'lng', 'artist_city_rank']
                display_cols = [col for col in city_columns if col in cities_df.columns]
                
                # Display top 15 cities
                st.write("\nTop 15 Cities by Listeners:")
                top_cities = cities_df.sort_values('listeners', ascending=False).head(15)
                st.dataframe(top_cities[display_cols])
                
                # Show full data
                st.write("\nAll Cities Data:")
                st.dataframe(cities_df[display_cols])
        else:
            st.warning("No cities data available")
        
        # Display Countries Data
        st.write("\n🌎 Countries Data")
        if not countries_df.empty:
            st.write(f"Total countries: {len(countries_df)}")
            total_country_listeners = countries_df['listeners'].sum() if 'listeners' in countries_df.columns else 0
            st.write(f"Total country listeners: {total_country_listeners:,}")
            
            # Calculate and display USA percentage
            usa_percentage = calculate_usa_percentage(countries_df)
            st.write(f"USA Listener Percentage: {usa_percentage:.2f}%")
            
            # Create expandable section for raw countries data
            with st.expander("View Countries Data"):
                # Select relevant columns for countries
                country_columns = ['name', 'code2', 'listeners', 'population', 'region']
                display_cols = [col for col in country_columns if col in countries_df.columns]
                
                # Sort by listeners and display all countries
                st.write("\nAll Countries by Listeners (sorted):")
                sorted_countries = countries_df.sort_values('listeners', ascending=False)
                st.dataframe(sorted_countries[display_cols])
        else:
            st.warning("No countries data available")
            
    except Exception as e:
        st.error(f"Error processing data: {str(e)}")
        # Show the raw data structure for debugging
        with st.expander("Debug - Raw Response Details"):
            st.json(response_data)

def fetch_and_display_audience_geography(artist_id):
    """
    Fetch and display audience geography data for a given artist ID.
    
    Parameters:
    -----------
    artist_id : int
        The ChartMetric artist ID
    """
    # Create tabs for different date parameters
    tab1, tab2, tab3 = st.tabs(["Latest Data", "Historical Data (date)", "Historical Data (since)"])
    
    with tab1:
        st.subheader("Latest Data")
        
        latest_params = {
            'latest': True,
            'limit': 50
        }
        
        st.write("Parameters used:")
        st.write(latest_params)
        
        try:
            # Fetch latest data
            with st.spinner("Fetching latest data..."):
                latest_data = chartmetric.get_artist_track_where_people_listen(
                    artist_id=int(artist_id),
                    params=latest_params
                )
            display_data_section(latest_data, "latest data")
        except Exception as e:
            st.error(f"Error fetching latest data: {str(e)}")
    
    with tab2:
        st.subheader("Historical Data using 'date' parameter")
        
        # Date input for historical data
        selected_date = st.date_input("Select Date", value=pd.to_datetime('2022-08-12'))
        date_str = selected_date.strftime('%Y-%m-%d')
        
        date_params = {
            'date': date_str,
            'limit': 50
        }
        
        st.write("Parameters used:")
        st.write(date_params)
        
        if st.button("Fetch Data with 'date' parameter"):
            try:
                # Fetch historical data using date
                with st.spinner(f"Fetching data for {date_str}..."):
                    historical_data = chartmetric.get_artist_track_where_people_listen(
                        artist_id=int(artist_id),
                        params=date_params
                    )
                display_data_section(historical_data, f"data for {date_str}")
            except Exception as e:
                st.error(f"Error fetching historical data: {str(e)}")
    
    with tab3:
        st.subheader("Historical Data using 'since' parameter")
        
        # Date input for historical data
        since_date = st.date_input("Select Since Date", value=pd.to_datetime('2022-08-12'), key="since_date")
        since_str = since_date.strftime('%Y-%m-%d')
        
        since_params = {
            'since': since_str,
            'limit': 50
        }
        
        st.write("Parameters used:")
        st.write(since_params)
        
        if st.button("Fetch Data with 'since' parameter"):
            try:
                # Fetch historical data using since
                with st.spinner(f"Fetching data since {since_str}..."):
                    since_data = chartmetric.get_artist_track_where_people_listen(
                        artist_id=int(artist_id),
                        params=since_params
                    )
                display_data_section(since_data, f"data since {since_str}")
            except Exception as e:
                st.error(f"Error fetching historical data: {str(e)}")

# Streamlit interface
st.title("ChartMetric Audience Geography Explorer")
st.write("This app explores the ChartMetric audience geography endpoint with different parameters.")

# Input for artist ID
artist_id = st.text_input("Enter ChartMetric Artist ID:", value="4276517")  # Using the user's artist ID

if artist_id:
    if artist_id.isdigit():
        fetch_and_display_audience_geography(int(artist_id))
    else:
        st.error("Please enter a valid numeric artist ID") 