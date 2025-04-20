import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import base64
from datetime import datetime, timedelta

# Import data loading functions from utils module
from utils.data_loader import get_mech_data, get_rates_data, load_local_csv
from utils.population_utils.population_data import get_population_data
from utils.population_utils.country_code_to_name import country_code_to_name
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
    extract_track_metrics,
    parse_catalog_file
)
from utils.decay_rates import (
    ranges_sp,
    sp_range,
    SP_REACH_DATA,
    SP_REACH,
    fitted_params,
    fitted_params_df,
    track_lifecycle_segment_boundaries,
    DEFAULT_STREAM_INFLUENCE_FACTOR,
    DEFAULT_TRACK_STREAMS_FORECAST_PERIOD,
    DEFAULT_VALUATION_CALCULATION_YEARS
)

# Import decay model functions from the new modules
from utils.decay_models import (
    exponential_decay,
    remove_anomalies,
    fit_decay_curve,
    analyze_listener_decay,
    calculate_monthly_listener_decay_rate
)

# Import track forecasting functions (now including parameter update functions)
from utils.track_stream_forecasting import (
    build_complete_track_forecast, 
    get_decay_parameters, 
    update_fitted_params, 
    fit_segment,
    generate_track_decay_rates_by_month,
    create_decay_rate_dataframe,
    adjust_track_decay_rates,
    calculate_track_decay_rates_by_segment,
    calculate_monthly_stream_projections,
    prepare_decay_rate_fitting_data,
    piecewise_exp_decay
)

# Import UI functions
from utils.ui_functions import (
    display_financial_parameters_ui, 
    display_valuation_results,
    display_valuation_summary,
    create_country_distribution_chart,
    create_yearly_revenue_chart
)

# Import financial parameters
from utils.financial_parameters import (
    PREMIUM_STREAM_PERCENTAGE, 
    AD_SUPPORTED_STREAM_PERCENTAGE,
    HISTORICAL_VALUE_TIME_ADJUSTMENT
)

# Import historical value calculation function
from utils.historical_royalty_revenue import calculate_historical_royalty_revenue, HISTORICAL_VALUATION_CUTOFF

# Import forecast projections functions
from utils.forecast_projections import (
    create_monthly_track_revenue_projections,
    aggregate_into_yearly_periods,
    apply_ownership_adjustments
)

# Import geographic analysis functions
from utils.geographic_analysis import (
    process_country_breakdown,
    get_top_countries,
    process_audience_geography
)

# Import ownership data functions
from utils.ownership_data import process_ownership_data

# Import fraud detection
from utils.fraud_detection import detect_streaming_fraud

def render_file_uploader_tab():
    """Render the File Uploader tab with all the necessary UI components and processing"""
    
    # ===== FILE UPLOAD SECTION =====
    # 1. Artist Monthly Listeners Data
    uploaded_file = st.file_uploader("Artist Monthly Spotify Listeners", type="csv")
    
    # Process artist data if uploaded
    mldr = 0.05  # Default value if no artist data
    if uploaded_file is not None:
        # Read the uploaded CSV file
        artist_monthly_listeners_df = pd.read_csv(uploaded_file)
        
        # Calculate decay rates using our dedicated function
        try:
            # Get the full decay analysis (which contains the MLDR and all visualization data)
            decay_analysis = analyze_listener_decay(artist_monthly_listeners_df, dayfirst=True)
            
            # Extract the MLDR (Monthly Listener Decay Rate) from the analysis results
            mldr = decay_analysis['mldr']
            
            # Get normalized dates for consistent month-based calculations
            min_date = decay_analysis['normalized_start_date'].date()  # Convert to datetime.date
            max_date = decay_analysis['normalized_end_date'].date()    # Convert to datetime.date
            
            # Date Range Selection UI
            st.write("Select Date Range:")
            start_date, end_date = st.slider(
                "Select date range",
                min_value=min_date,
                max_value=max_date,
                value=(min_date, max_date),
                format="YYYY-MM-DD"
            )
            
            # Convert slider values to Timestamp objects
            start_date = pd.Timestamp(start_date)
            end_date = pd.Timestamp(end_date)

            # Update the decay calculation if the user changes the date range
            if start_date != decay_analysis['normalized_start_date'] or end_date != decay_analysis['normalized_end_date']:
                try:
                    # Recalculate the decay analysis with the new date range
                    decay_analysis = analyze_listener_decay(artist_monthly_listeners_df, start_date, end_date, dayfirst=True)
                    
                    # Extract the updated MLDR
                    mldr = decay_analysis['mldr']
                except ValueError as e:
                    st.error(f"Error updating date range: {str(e)}")
            
            # Display Results
            st.write(f'Exponential decay rate: {mldr}')
            
            # Always show normalized dates for transparency
            st.write(f'Normalized date range used: {decay_analysis["normalized_start_date"].strftime("%Y-%m-%d")} to {decay_analysis["normalized_end_date"].strftime("%Y-%m-%d")}')
            
            # Visualization
            fig, ax = plt.subplots(figsize=(10, 4))
            # Plot the moving average
            ax.plot(decay_analysis['date_filtered_listener_data']['Date'], decay_analysis['date_filtered_listener_data']['4_Week_MA'], label='Moving Average', color='tab:blue', linewidth=2)
            # Plot the fitted decay curve using pre-calculated parameters
            ax.plot(decay_analysis['date_filtered_listener_data']['Date'], exponential_decay(decay_analysis['date_filtered_listener_data']['Months'], *decay_analysis['fitted_decay_parameters']), 
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
            
        except ValueError as e:
            st.error(f"Error processing artist data: {str(e)}")
    else:
        st.warning("No artist monthly listeners data provided. This may impact decay rate calculations.")
    
    # Track Catalog Data
    uploaded_catalog_file = st.file_uploader("Track Catalog CSV", type=["csv"], 
                                             help="Upload a single CSV containing data for multiple tracks.")
    
    uploaded_file_audience_geography = st.file_uploader("Audience Geography", type=["csv"])
    uploaded_file_ownership = st.file_uploader("MLC Claimed and Song Ownership", type="csv")

    # Parse the catalog file to extract individual track data
    track_data_map, track_names, parse_errors = parse_catalog_file(uploaded_catalog_file)
    
    # Display any parsing errors
    for error in parse_errors:
        st.error(error)
    
    # Display the number of tracks found in the catalog
    if track_names:
        st.success(f"Found {len(track_names)} tracks in the catalog file.")

    # ===== UI DISPLAY AND TRACK SELECTION =====
    # Set default selection to all tracks
    selected_songs = st.multiselect("Select Songs to Analyze", track_names, default=track_names)
    
    # ===== FINANCIAL PARAMETERS =====
    # The discount rate is used to:
    #  1. Convert future projected royalty earnings to present value
    #  2. Adjust historical value calculations for time value
    #  3. Account for risk and opportunity cost in the valuation model
    # The default of 4.5% represents a moderate risk profile for music royalty assets
    discount_rate = display_financial_parameters_ui()

    # ===== RUN BUTTON =====
    if st.button('Run All'):
        # Load required global datasets
        population_df = get_population_data()
        mechanical_royalty_rates_df = get_mech_data()
        worldwide_royalty_rates_df = get_rates_data()
        
        # Check if data loading succeeded
        if mechanical_royalty_rates_df is None:
            st.error("Failed to load mechanical royalties data")
            return
        
        if worldwide_royalty_rates_df is None:
            st.error("Failed to load worldwide rates data")
            return
            
        # Initialize data structures to store results
        track_yearly_revenue_collection = []
        export_track_streams_forecast = pd.DataFrame()
        track_valuation_summaries = []
        track_catalog_df = pd.DataFrame() #DataFrame that will store data for all tracks (our catalog)
        track_forecast_results = {}  # Dictionary to store all forecast results
        
        # Verify we have at least one track selected and the necessary data
        if not track_names:
            st.error("No tracks found in the catalog file. Please upload a valid catalog file.")
            return
            
        if not selected_songs:
            st.error("No songs selected for analysis. Please select at least one song.")
            return
        
        # Process each selected song's data from our track_data_map
        for selected_song in selected_songs:
            if selected_song in track_data_map:
                # Get the dataframe for this track
                df_track_data_unique = track_data_map[selected_song]
                track_name_unique = selected_song  # We already have the track name
                
                # Make column names more descriptive - 'Value' becomes 'CumulativeStreams'
                df_track_data_unique = rename_columns(df_track_data_unique, {'Value': 'CumulativeStreams'})
                
                # Verify we have the required data
                if 'Date' not in df_track_data_unique.columns or 'CumulativeStreams' not in df_track_data_unique.columns:
                    st.error(f"Missing required columns for track '{selected_song}'. Skipping...")
                    continue

                # Step 1: Extract track metrics
                try:
                    track_metrics = extract_track_metrics(
                        track_data_df=df_track_data_unique,
                        track_name=track_name_unique
                    )
                
                    # Step 2: Generate track stream forecast using the metrics
                    forecast_result = build_complete_track_forecast(
                        track_metrics=track_metrics,
                        mldr=mldr,  # From artist-level analysis done earlier
                        fitted_params_df=fitted_params_df,
                        stream_influence_factor=DEFAULT_STREAM_INFLUENCE_FACTOR,
                        sp_range=sp_range,
                        sp_reach=SP_REACH,
                        track_lifecycle_segment_boundaries=track_lifecycle_segment_boundaries,
                        forecast_periods=DEFAULT_TRACK_STREAMS_FORECAST_PERIOD
                    )
                
                    # Combine track metrics and forecast results for compatibility with existing code
                    combined_result = {**track_metrics, **forecast_result}
                
                    # Store the combined results
                    track_forecast_results[track_name_unique] = combined_result
                
                    # Create a single row DataFrame for this track's metrics (for compatibility with existing code)
                    track_data = pd.DataFrame({
                        'track_name': [track_name_unique],
                        'earliest_track_date': [track_metrics['earliest_track_date']],
                        'track_streams_last_30days': [track_metrics['track_streams_last_30days']],
                        'track_streams_last_90days': [track_metrics['track_streams_last_90days']],
                        'track_streams_last_365days': [track_metrics['track_streams_last_365days']],
                        'total_historical_track_streams': [track_metrics['total_historical_track_streams']],
                        'months_since_release_total': [track_metrics['months_since_release_total']],
                        'months_since_release': [track_metrics['months_since_release'].tolist() if hasattr(track_metrics['months_since_release'], 'tolist') else track_metrics['months_since_release']],
                        'monthly_averages': [track_metrics['monthly_averages'].tolist() if hasattr(track_metrics['monthly_averages'], 'tolist') else track_metrics['monthly_averages']]
                    })
                
                    # Add to catalog DataFrame for compatibility with existing code
                    track_catalog_df = pd.concat([track_catalog_df, track_data], ignore_index=True)
                
                    # Add forecast data to export DataFrame
                    forecast_df_copy = forecast_result['forecast_df'].copy()
                    forecast_df_copy['track_name'] = track_name_unique
                    export_track_streams_forecast = pd.concat([export_track_streams_forecast, forecast_df_copy], ignore_index=True)
                    
                    st.success(f"Successfully processed track: {selected_song}")
                
                except Exception as e:
                    st.error(f"Error processing track '{selected_song}': {str(e)}")
                    continue
            else:
                st.error(f"Data for track '{selected_song}' not found in the catalog. Skipping...")

        # ===== AUDIENCE GEOGRAPHY PROCESSING =====
        # Process the audience geography data to determine geographic distribution of listeners
        # This data is used to apply country-specific royalty rates in revenue projections
        # If no geography data is provided, assume 100% US market for royalty calculations
        listener_geography_df, listener_percentage_usa = process_audience_geography(uploaded_file_audience_geography)

        # ===== OWNERSHIP DATA PROCESSING =====
        # Process ownership and MLC claim information to accurately calculate revenue shares
        # This ensures all calculations account for partial ownership and existing royalty claims
        # If no ownership data is provided, assume 100% ownership and 0% MLC claims
        ownership_df = process_ownership_data(uploaded_file_ownership, track_names)

        # Process each selected song
        for selected_song in selected_songs:
            # Check if we have forecast data for this song
            if selected_song not in track_forecast_results:
                st.error(f"Data for {selected_song} not processed properly. Skipping...")
                continue
                
            # Get forecast results
            forecast_result = track_forecast_results[selected_song]
            track_streams_forecast_df = forecast_result['forecast_df']
            
            # Calculate total predicted streams for valuation
            track_valuation_months = DEFAULT_VALUATION_CALCULATION_YEARS * 12
            total_track_streams_forecast = track_streams_forecast_df.loc[:track_valuation_months, 'predicted_streams_for_month'].sum()
            
            # ===== 8. CALCULATE HISTORICAL VALUE =====
            # Determine the end date for royalty rate calculations
            # For older tracks, use the valuation cutoff date
            # For newer tracks, use the latest available data
            
            # Convert date from "DD/MM/YYYY" to "YYYY-MM" format for mechanical royalty rate comparison
            earliest_track_date_formatted = datetime.strptime(forecast_result['earliest_track_date'], "%d/%m/%Y").strftime('%Y-%m')
            
            if earliest_track_date_formatted >= HISTORICAL_VALUATION_CUTOFF:
                royalty_calculation_end_date = mechanical_royalty_rates_df['Date'].max()
            else:
                royalty_calculation_end_date = HISTORICAL_VALUATION_CUTOFF
                
            # Filter mechanical royalty data for relevant date range
            # Note: MECHv2_fixed.csv dates are already in 'YYYY-MM' format, so no conversion needed
            mask = (mechanical_royalty_rates_df['Date'] >= earliest_track_date_formatted) & (mechanical_royalty_rates_df['Date'] <= royalty_calculation_end_date)
            
            # Calculate historical royalty value using our dedicated function
            historical_royalty_value_time_adjusted = calculate_historical_royalty_revenue(
                total_historical_track_streams=forecast_result['total_historical_track_streams'],
                mechanical_royalty_rates_df=mechanical_royalty_rates_df,
                date_range_mask=mask,
                listener_percentage_usa=listener_percentage_usa,
                discount_rate=discount_rate,
                historical_value_time_adjustment=HISTORICAL_VALUE_TIME_ADJUSTMENT,
                premium_stream_percentage=PREMIUM_STREAM_PERCENTAGE,
                ad_supported_stream_percentage=AD_SUPPORTED_STREAM_PERCENTAGE
            )

            # ===== 9. PREPARE MONTHLY FORECAST DATA =====
            # Use utility function to create projections DataFrame and apply geographic distribution
            monthly_track_revenue_projections_df = create_monthly_track_revenue_projections(
                track_name=selected_song,
                track_streams_forecast_df=track_streams_forecast_df,
                listener_geography_df=listener_geography_df,
                worldwide_royalty_rates_df=worldwide_royalty_rates_df,
                discount_rate=discount_rate
            )
            
            # Calculate total discounted and non-discounted values
            discounted_future_royalty_value = monthly_track_revenue_projections_df['DISC'].sum()
            undiscounted_future_royalty_value = monthly_track_revenue_projections_df['Total'].sum()
            total_track_valuation = discounted_future_royalty_value + historical_royalty_value_time_adjusted

            # ===== 12. STORE FORECAST SUMMARY =====
            track_valuation_summaries.append({
                'track_name': selected_song,
                'total_historical_track_streams': forecast_result['total_historical_track_streams'],
                'total_track_streams_forecast': total_track_streams_forecast,
                'historical_royalty_value_time_adjusted': historical_royalty_value_time_adjusted,
                'undiscounted_future_royalty_value': undiscounted_future_royalty_value,
                'discounted_future_royalty_value': discounted_future_royalty_value,
                'total_track_valuation': total_track_valuation,
            })

            # ===== 13. AGGREGATE MONTHLY DATA INTO YEARLY PERIODS =====
            # Use utility function to aggregate monthly projections into yearly periods
            yearly_track_revenue_df = aggregate_into_yearly_periods(monthly_track_revenue_projections_df)
            
            # Store yearly aggregated data for plotting
            track_yearly_revenue_collection.append(yearly_track_revenue_df)

        # ===== 14. DATA EXPORT AND AGGREGATION =====
        # Create downloadable CSV file of track stream forecasts
        csv = export_track_streams_forecast.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        download_link = f'<a href="data:file/csv;base64,{b64}" download="track_streams_forecast.csv">Download Track Streams Forecast</a>'
        st.markdown(download_link, unsafe_allow_html=True)
        
        # Combine track revenue data across all tracks and summarize by year
        yearly_revenue_combined_df = pd.concat(track_yearly_revenue_collection)
        yearly_total_by_year_df = yearly_revenue_combined_df.groupby('Year')['DISC'].sum().reset_index()
        
        # Convert track valuation summaries to DataFrame for display
        track_valuation_results_df = pd.DataFrame(track_valuation_summaries)

        # ===== 15. OWNERSHIP ADJUSTMENTS =====
        # Apply ownership adjustments to valuation results
        ownership_adjusted_valuation_df = apply_ownership_adjustments(track_valuation_results_df, ownership_df)
        
        # ===== 16. DISPLAY FORMATTING =====
        # Format and display valuation results
        final_valuation_display_df = display_valuation_results(ownership_adjusted_valuation_df)

        # ===== 17. SUMMARY STATISTICS =====
        # Calculate and display summary statistics across all tracks in the catalog
        catalog_valuation_summary_df = display_valuation_summary(final_valuation_display_df)

        # ===== 18. GEOGRAPHIC DISTRIBUTION ANALYSIS =====
        # Process country-specific revenue data and get top countries
        df_country_breakdown = process_country_breakdown(listener_geography_df, monthly_track_revenue_projections_df)
        top_countries, top_10_percentage_sum = get_top_countries(df_country_breakdown)
        
        # ===== 19. VISUALIZATION: TOP COUNTRIES =====
        # Create and display country distribution chart
        fig, ax = create_country_distribution_chart(top_countries, top_10_percentage_sum)
        st.pyplot(fig)

        # ===== 20. VISUALIZATION: YEARLY INCOME =====
        # Create and display yearly revenue chart
        fig, ax = create_yearly_revenue_chart(yearly_total_by_year_df)
        st.pyplot(fig)

        # ===== 21. FRAUD DETECTION =====
        # Detect potential streaming fraud
        alert_countries = detect_streaming_fraud(listener_geography_df, population_df)
        
        # Display fraud alerts if any are detected
        if alert_countries:
            st.write("Fraud Alert. This artist has unusually high streams from these countries:")
            for country in alert_countries:
                st.write(country)
    
    return 