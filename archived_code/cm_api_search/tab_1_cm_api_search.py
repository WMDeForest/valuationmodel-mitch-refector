"""
Archived tab1 implementation code from streamlit_app.py.
This code implemented the "API Search" tab which allowed users to search for artist and track data
using the ChartMetric API. It provided form controls for entering artist/track IDs, displayed artist
listener data, calculated decay rates, and showed geographic streaming distribution.
The code formed the complete implementation of the first tab in the original application.
This code is not active in the current application and is preserved for reference only.
"""

// ... existing code ... 
with tab1:
    st.title("Artist and Track ID Form")
    
    # Initialize session state variables if they don't exist
    if 'artist_data' not in st.session_state:
        st.session_state.artist_data = None
    if 'monthly_data' not in st.session_state:
        st.session_state.monthly_data = None
    if 'date_range' not in st.session_state:
        st.session_state.date_range = {
            'min_date': None,
            'max_date': None,
            'start_date': None,
            'end_date': None
        }
    if 'mldr' not in st.session_state:
        st.session_state.mldr = None
    if 'track_summary_list' not in st.session_state:
        st.session_state.track_summary_list = []
    
    # Function to load test data
    def load_test_data():
        st.session_state['artist_id'] = "4276517"  # Thymes
        st.session_state['track_id'] = "60300793"  # Free as a bird
        
    # Initialize session state for input fields if not already done
    if 'artist_id' not in st.session_state:
        st.session_state['artist_id'] = ""
    if 'track_id' not in st.session_state:
        st.session_state['track_id'] = ""
        
    with st.form(key="artist_track_form"):
        st.subheader("Enter Details")
        artist_id = st.text_input("Artist ID", placeholder="Enter Artist ID", value=st.session_state['artist_id'], key="artist_id_input")
        track_ids_input = st.text_input("Track ID", placeholder="Enter Track ID", value=st.session_state['track_id'], key="track_id_input")
        
        col1, col2 = st.columns([1, 1])
        with col1:
            submit_button = st.form_submit_button("Submit")
        with col2:
            load_test_button = st.form_submit_button("Load Test Data", help="Load test data with Artist ID: Thymes (4276517) and Track ID: Free as a bird (60300793)")
            
    # Handle the load test button click
    if load_test_button:
        load_test_data()
        st.rerun()  # Rerun the app to reflect the changes in the form fields

    if submit_button:
        start_time = time.time()
        st.session_state.track_summary_list = []  # Reset the list when form is submitted

        data = chartmetric.get_artist_spotify_stats(artist_id)
        df = pd.DataFrame(data)
        df.rename(columns={'timestp': 'Date', 'value': 'Monthly Listeners', 'diff': 'Difference'}, inplace=True)
        df['Date'] = pd.to_datetime(df['Date']).dt.strftime('%b %d, %Y')
        df['Difference'] = df['Difference'].fillna(0).astype(int)

        if 'Date' in df.columns:
            try:
                df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
            except Exception as e:
                st.error(f"Failed to convert Date column: {e}")

        if df['Date'].isna().any():
            st.warning("Some dates couldn't be parsed and have been set to 'NaT'. Please check your data.")

        df = df[['Date', 'Monthly Listeners']].rename(columns={'Monthly Listeners': 'Streams'})
        df = df.iloc[::7, :]
        df = df.sort_values(by='Date')
        
        # Remove anomalies and store in session state
        st.session_state.monthly_data = remove_anomalies(df)
        st.session_state.artist_data = df
        
        # Update date range in session state
        st.session_state.date_range = {
            'min_date': st.session_state.monthly_data['Date'].min().to_pydatetime(),
            'max_date': st.session_state.monthly_data['Date'].max().to_pydatetime(),
            'start_date': st.session_state.monthly_data['Date'].min().to_pydatetime(),
            'end_date': st.session_state.monthly_data['Date'].max().to_pydatetime()
        }

    # Only show date range slider and plot if we have data
    if st.session_state.monthly_data is not None:
        st.write("Select Date Range:")
        try:
            selected_start, selected_end = st.slider(
                "Select date range",
                min_value=st.session_state.date_range['min_date'],
                max_value=st.session_state.date_range['max_date'],
                value=(st.session_state.date_range['start_date'], 
                      st.session_state.date_range['end_date']),
                format="YYYY-MM-DD",
                key='date_slider'
            )
            
            # Update session state with new selection
            st.session_state.date_range['start_date'] = selected_start
            st.session_state.date_range['end_date'] = selected_end
            
            # Filter data based on selection
            mask = (st.session_state.monthly_data['Date'] >= selected_start) & (st.session_state.monthly_data['Date'] <= selected_end)
            subset_df = st.session_state.monthly_data[mask].copy()
            
            if len(subset_df) > 0:
                # Calculate months for decay rate
                min_date = subset_df['Date'].min()
                subset_df['Months'] = subset_df['Date'].apply(
                    lambda x: (x.year - min_date.year) * 12 + x.month - min_date.month
                )
                
                # Calculate decay rate and store in session state
                mldr, popt = calculate_decay_rate(subset_df)
                st.session_state.mldr = mldr
                st.write(f'Exponential decay ratemldr: {mldr}')
                
                # Create plot
                fig, ax = plt.subplots(figsize=(10, 4))
                ax.plot(subset_df['Date'], subset_df['4_Week_MA'], 
                       label='Moving Average', color='tab:blue', linewidth=2)
                ax.plot(subset_df['Date'], 
                       exponential_decay(subset_df['Months'], *popt), 
                       label='Fitted Decay Curve', color='red', linestyle='--')
                
                ax.set_xlabel('Date', fontsize=12)
                ax.set_ylabel('Monthly Listeners', fontsize=12)  # Changed from 'Streams' to 'Monthly Listeners'
                ax.set_title('Moving Average and Exponential Decay', 
                           fontsize=14, weight='bold')
                ax.legend()
                ax.set_ylim(bottom=0)
                plt.xticks(rotation=45)
                
                fig.patch.set_visible(False)
                ax.set_facecolor('none')
                ax.patch.set_alpha(0)
                plt.tight_layout()
                st.pyplot(fig)
            else:
                st.warning("No data available for the selected date range.")
        except Exception as e:
            st.error(f"Error updating date range: {str(e)}")
            st.write("Please try adjusting the date range again.")

        track_ids = track_ids_input.split(',')
        for track_id in track_ids:
            track_id = track_id.strip()
            df_track_data_unique = chartmetric.get_track_sp_streams_campare(track_id=track_id)
            time.sleep(1)
            track_detail = chartmetric.get_track_detail(track_id=track_id)
            time.sleep(1)

            df_track_data_unique = pd.DataFrame(df_track_data_unique)
            df_track_data_unique.rename(columns={'timestp': 'Date', 'value': 'Value'}, inplace=True)
            
            # Display date range information
            start_date = pd.to_datetime(df_track_data_unique['Date'].min())
            end_date = pd.to_datetime(df_track_data_unique['Date'].max())
            st.write(f"Track: {track_detail.name}")
            st.write(f"Data range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
            st.write(f"Total days of data: {(end_date - start_date).days}")
            
            df_track_data_unique['Date'] = pd.to_datetime(df_track_data_unique['Date']).dt.strftime('%b %d, %Y')
            release_date_unique = pd.to_datetime(df_track_data_unique['Date'].iloc[0], format='%b %d, %Y').strftime('%d/%m/%Y')
            total_value_unique = df_track_data_unique['Value'].iloc[-1]
            
            if len(df_track_data_unique) > 30:
                spotify_streams_1m_unique = total_value_unique - df_track_data_unique['Value'].iloc[-31]
            else:
                spotify_streams_1m_unique = total_value_unique
            
            if len(df_track_data_unique) > 90:
                spotify_streams_3m_unique = total_value_unique - df_track_data_unique['Value'].iloc[-91]
            else:
                spotify_streams_3m_unique = total_value_unique
            
            if len(df_track_data_unique) > 365:
                spotify_streams_12m_unique = total_value_unique - df_track_data_unique['Value'].iloc[-366]
            else:
                spotify_streams_12m_unique = total_value_unique
            
            track_name_unique = track_detail.name

            st.session_state.track_summary_list.append({
                'Track': track_name_unique,
                'Release date': release_date_unique,
                'Spotify Streams 1m': spotify_streams_1m_unique,
                'Spotify Streams 3m': spotify_streams_3m_unique,
                'Spotify Streams 12m': spotify_streams_12m_unique,
                'Spotify Streams Total': total_value_unique
            })

        track_summary_df_unique = pd.DataFrame(st.session_state.track_summary_list)
        df = track_summary_df_unique
        ownership_df = pd.DataFrame({
            'Track': df['Track'],
            'Ownership(%)': [None] * len(df),
            'MLC Claimed(%)': [None] * len(df)
        })

        df_additional['Date'] = pd.to_datetime(df_additional['Date'], format='%b-%y')

        if 'Spotify Streams 1m' in df.columns:
            df['streams_last_month'] = df['Spotify Streams 1m']
        if 'Spotify Streams 3m' in df.columns:
            df['total_streams_3_months'] = df['Spotify Streams 3m']
        if 'Spotify Streams 12m' in df.columns:
            df['total_streams_12_months'] = df['Spotify Streams 12m']
        if 'Spotify Streams Total' in df.columns:
            df['historical'] = df['Spotify Streams Total']
        if 'Release date' in df.columns:
            df['release_date'] = df['Release date']

        columns_to_drop = ["Release date", "Spotify Streams 1m", "Spotify Streams 3m", "Spotify Streams 12m", "Spotify Streams Total"]
        df.drop(columns=columns_to_drop, inplace=True)

        # Stream influence factor (formerly called sp_playlist_reach)
        stream_influence_factor = 1000
        forecast_periods = 400
        current_date = datetime.today()

        st.write("Data Preview:")
        st.write(df)

        # Third Endpoint Calls
        # Use a specific date (August 12, 2024) as mentioned in the documentation
        api_params = {
            'limit': 50,       # Request the maximum number of cities (50)
            'date': '2024-08-12'  # Specific date snapshot (August 12, 2024)
        }
        
        # Call the API directly to get the raw response
        response = chartmetric._ChartMetricService__get_artist_track_where_people_listen_request(artist_id, api_params)
        
        # Check if the API response has the expected structure
        if "obj" in response and "cities" in response["obj"]:
            raw_obj = response["obj"]
            
            # Get city data
            cities_data = []
            for city, entries in raw_obj["cities"].items():
                for entry in entries:
                    city_entry = {
                        'City': city,
                        'Country': entry.get('code2', ''),
                        'Spotify Monthly Listeners': entry.get('listeners', 0),
                        'Date': api_params['date']
                    }
                    cities_data.append(city_entry)
            
            # Create DataFrame with city data
            cities_df = pd.DataFrame(cities_data)
            
            st.write(f"### Top Cities for {api_params['date']}: Found {len(cities_data)} cities")
            st.write(cities_df)
            
            # Group by Country (similar to Tab 2 approach)
            country_df = cities_df.groupby('Country')['Spotify Monthly Listeners'].sum().reset_index()
            
            st.write("### Country Totals (from city data):")
            st.write(country_df)
            
            # Download link for the data
            csv = cities_df.to_csv(index=False)
            b64 = base64.b64encode(csv.encode()).decode()
            href = f'<a href="data:file/csv;base64,{b64}" download="city_data_{api_params["date"]}.csv">Download city data as CSV</a>'
            st.markdown(href, unsafe_allow_html=True)
            
            # Use the city-based country data for further processing
            audience_df = country_df
            
            # Convert country codes to full names to match GLOBAL dataset
            audience_df['Original_Country'] = audience_df['Country']  # Save original code
            audience_df['Country'] = audience_df['Country'].apply(country_code_to_name)
            
            # Process the country data
            total_listeners = audience_df['Spotify Monthly Listeners'].sum()
            audience_df['Spotify monthly listeners (%)'] = (audience_df['Spotify Monthly Listeners'] / total_listeners) * 100
            
            st.write("### Final Processed Data (with country names):")
            st.write(audience_df)
            
            audience_df["Spotify monthly listeners (%)"] = pd.to_numeric(audience_df["Spotify monthly listeners (%)"], errors='coerce')
            audience_df["Spotify monthly listeners (%)"] = audience_df["Spotify monthly listeners (%)"] / 100
            
            # Check if United States exists before trying to access it
            if "United States" in audience_df["Country"].values:
                percentage_usa = audience_df.loc[audience_df["Country"] == "United States", "Spotify monthly listeners (%)"].values[0]
            else:
                st.error("No United States data found - this will cause an error in further processing")
                # This will error out if United States is not found, which is what we want for debugging
                percentage_usa = audience_df.loc[audience_df["Country"] == "United States", "Spotify monthly listeners (%)"].values[0]
        else:
            st.error("Could not retrieve city data for the specified date. API response:")
            st.write(response)
            st.stop()
            
        component_handler(df)