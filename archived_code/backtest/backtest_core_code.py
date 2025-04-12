with tab2:
    st.title("Backtest Model Accuracy")
    
    # First file upload - Artist Listener History
    uploaded_file = st.file_uploader("Artist Listener History", type="csv", key="spotify_streams_tab3")
    
    # Second file upload - Historical Track Streaming Data
    uploaded_files_unique = st.file_uploader("Historical Track Streaming Data", type=['csv'], accept_multiple_files=True, key="track_data_tab3")
    
    # Process the data only if both types of files are uploaded
    if uploaded_file is not None and uploaded_files_unique:
        # Process Artist Listener History
        listener_df = pd.read_csv(uploaded_file)
        
        # Basic data validation for listener history
        if 'Date' not in listener_df.columns or 'Monthly Listeners' not in listener_df.columns:
            st.error("The Artist Listener History file must contain 'Date' and 'Monthly Listeners' columns.")
            st.stop()
            
        # Split listener history data
        listener_train_df, listener_val_df = split_listener_history_for_backtesting(listener_df)
        
        if listener_train_df is None:
            st.error("Insufficient data in Artist Listener History for backtesting. Need at least 27 months of historical data.")
            st.stop()

        # Calculate mldr from listener history data
        listener_train_df['Date'] = pd.to_datetime(listener_train_df['Date'])
        listener_train_df['YearMonth'] = listener_train_df['Date'].dt.strftime('%Y-%m')
        monthly_listeners = listener_train_df.groupby('YearMonth').agg({
            'Monthly Listeners': 'mean',
            'Date': 'first'
        }).reset_index()
        
        # Calculate months since first date
        min_date = monthly_listeners['Date'].min()
        monthly_listeners['Months'] = monthly_listeners['Date'].apply(
            lambda x: (x.year - min_date.year) * 12 + x.month - min_date.month
        )
        
        # Fit exponential decay model
        x_data = monthly_listeners['Months']
        y_data = monthly_listeners['Monthly Listeners']
        popt, _ = curve_fit(exponential_decay, x_data, y_data, p0=(max(y_data), 0.1))
        mldr = popt[1]  # This is the decay rate
        
        # Display the Artist MLDR prominently
        st.write("## Artist MLDR (Decay Rate)")
        st.info(f"**Artist MLDR: {mldr:.6f}**")
        st.write("This exponential decay rate will be used in the model calculations.")
            
        # Display information about the listener history split
        st.write("Artist Listener History Data Split:")
        st.write(f"Training period: {listener_train_df['Date'].min().strftime('%Y-%m-%d')} to {listener_train_df['Date'].max().strftime('%Y-%m-%d')}")
        st.write(f"Validation period: {listener_val_df['Date'].min().strftime('%Y-%m-%d')} to {listener_val_df['Date'].max().strftime('%Y-%m-%d')}")
        
        # Process Track Streaming Data
        track_data_splits = []
        track_summary_list = []  # New list for track summaries
        
        for file in uploaded_files_unique:
            track_df = pd.read_csv(file)
            
            # Basic data validation for track streaming data
            if 'Date' not in track_df.columns or 'Value' not in track_df.columns:
                st.error(f"File {file.name} must contain 'Date' and 'Value' columns.")
                continue
            
            # Calculate streaming metrics as in tab2
            total_value = track_df['Value'].iloc[-1]  # Total historical streams
            streams_last_month = total_value - track_df['Value'].iloc[-31] if len(track_df) > 30 else total_value
            streams_3_months = total_value - track_df['Value'].iloc[-91] if len(track_df) > 90 else total_value
            streams_12_months = total_value - track_df['Value'].iloc[-366] if len(track_df) > 365 else total_value
            release_date = pd.to_datetime(track_df['Date'].iloc[0]).strftime('%d/%m/%Y')
            
            track_name = file.name.split(' - ')[1].strip()
            
            # Add to summary list
            track_summary_list.append({
                'Track': track_name,
                'streams_last_month': streams_last_month,
                'total_streams_3_months': streams_3_months,
                'total_streams_12_months': streams_12_months,
                'historical': total_value,
                'release_date': release_date
            })
                
            # Split track streaming data
            track_train_df, track_val_df = split_track_streaming_for_backtesting(track_df)
            
            if track_train_df is None:
                st.warning(f"Insufficient data in {file.name} for backtesting. Skipping this track.")
                continue
            
            track_data_splits.append({
                'track_name': track_name,
                'training_df': track_train_df,
                'validation_df': track_val_df
            })
        
        if not track_data_splits:
            st.error("No valid track data available for backtesting.")
            st.stop()
        
        # Display track summary data
        st.write("\nData Preview:")
        track_summary_df = pd.DataFrame(track_summary_list)
        st.write(track_summary_df)
        
        # Display track data splits information
        st.write("\nTrack Streaming Data Splits:")
        for track_split in track_data_splits:
            st.write(f"\nTrack: {track_split['track_name']}")
            st.write(f"Training period: {track_split['training_df']['Date'].min().strftime('%Y-%m-%d')} to {track_split['training_df']['Date'].max().strftime('%Y-%m-%d')}")
            st.write(f"Validation period: {track_split['validation_df']['Date'].min().strftime('%Y-%m-%d')} to {track_split['validation_df']['Date'].max().strftime('%Y-%m-%d')}")
        
        if st.button("Run Backtest"):
            st.write("Running backtest analysis...")
            
            track_metrics_list = []
            
            for track_split in track_data_splits:
                st.write(f"\nAnalyzing track: {track_split['track_name']}")
                
                # Prepare training data for forecasting
                training_df = track_split['training_df']
                validation_df = track_split['validation_df']
                
                # Calculate key metrics from training data
                # Get the last month's streams (new streams, not cumulative)
                if len(training_df) > 30:
                    last_month_streams = training_df['Value'].iloc[-1] - training_df['Value'].iloc[-31]
                else:
                    last_month_streams = training_df['Value'].iloc[-1]

                # Calculate 3-month streams (new streams)
                if len(training_df) > 90:
                    last_3_months = training_df['Value'].iloc[-1] - training_df['Value'].iloc[-91]
                else:
                    last_3_months = training_df['Value'].iloc[-1]

                # Calculate 12-month streams (new streams)
                if len(training_df) > 365:
                    last_12_months = training_df['Value'].iloc[-1] - training_df['Value'].iloc[-366]
                else:
                    last_12_months = training_df['Value'].iloc[-1]

                historical_total = training_df['Value'].iloc[-1]  # This should still be cumulative
                release_date = training_df['Date'].min()

                # Create forecast input data structure
                forecast_input = pd.DataFrame({
                    'Track': [track_split['track_name']],
                    'streams_last_month': [last_month_streams],
                    'total_streams_3_months': [last_3_months],
                    'total_streams_12_months': [last_12_months],
                    'historical': [historical_total],
                    'release_date': [release_date.strftime('%d/%m/%Y')]
                })

                # Prepare monthly actual values from validation data
                validation_df['Date'] = pd.to_datetime(validation_df['Date'])
                validation_df['YearMonth'] = validation_df['Date'].dt.strftime('%Y-%m')
                monthly_actual = validation_df.groupby('YearMonth').agg({
                    'Value': lambda x: x.iloc[-1] - x.iloc[0],  # Calculate new streams in each month
                    'Date': 'first'
                }).reset_index()

                # Generate forecasts using training data
                # Stream influence factor (formerly called sp_playlist_reach)
                stream_influence_factor = 1000  # Default value as in tab2
                current_date = training_df['Date'].max()
                
                # Update fitted parameters based on playlist reach
                updated_fitted_params_df = update_fitted_params(fitted_params_df, stream_influence_factor, sp_range, SP_REACH)
                
                # Calculate months since release for forecasting
                months_since_release_total = (current_date - release_date).days // 30
                
                # Calculate monthly averages
                monthly_avg_3_months = (last_3_months - last_month_streams) / (2 if months_since_release_total > 2 else 1)
                monthly_avg_last_month = last_month_streams
                monthly_avg_12_months = (last_12_months - last_3_months) / (9 if months_since_release_total > 11 else (months_since_release_total - 3))
                
                # Prepare data for decay rate calculation
                months_since_release = np.array([
                    max((months_since_release_total - 11), 0),
                    max((months_since_release_total - 2), 0),
                    months_since_release_total - 0
                ])
                monthly_averages = np.array([monthly_avg_12_months, monthly_avg_3_months, monthly_avg_last_month])
                
                # Calculate decay rates
                params = fit_segment(months_since_release, monthly_averages)
                S0, k = params
                
                # Generate forecasts for validation period
                validation_months = len(validation_df)
                consolidated_df = pd.DataFrame({
                    'segment': range(1, len(breakpoints)),
                    'k': [k] * (len(breakpoints) - 1)  # Using the calculated decay rate for all segments
                })
                
                forecasts = forecast_values(consolidated_df, last_month_streams, months_since_release_total, validation_months)
                forecast_df = pd.DataFrame(forecasts)
                
                # Convert month numbers to actual dates starting from the end of training period
                last_training_date = training_df['Date'].max()
                forecast_df['Date'] = [last_training_date + pd.DateOffset(months=i+1) for i in range(len(forecast_df))]
                
                # Prepare monthly actual values
                validation_df['Date'] = pd.to_datetime(validation_df['Date'])
                validation_df['YearMonth'] = validation_df['Date'].dt.strftime('%Y-%m')
                monthly_actual = validation_df.groupby('YearMonth').agg({
                    'Value': lambda x: x.iloc[-1] - x.iloc[0],  # Calculate new streams in each month
                    'Date': 'first'
                }).reset_index()

                # Generate forecasts for validation period
                validation_months = len(validation_df['Date'].dt.to_period('M').unique())
                
                # Calculate months since release for all periods
                months_since_release_all = list(range(1, 500))
                decay_rate_list = []

                # Get initial decay rates for each month
                for month in months_since_release_all:
                    for i in range(len(breakpoints) - 1):
                        if breakpoints[i] <= month < breakpoints[i + 1]:
                            decay_rate = updated_fitted_params_df.loc[i, 'k']
                            decay_rate_list.append(decay_rate)
                            break

                # Create final_df with initial decay rates
                final_df = pd.DataFrame({
                    'months_since_release': months_since_release_all,
                    'decay_rate': decay_rate_list
                })

                # Add mldr to relevant months
                start_month = min(months_since_release)
                end_month = max(months_since_release)
                final_df.loc[(final_df['months_since_release'] >= start_month) & 
                            (final_df['months_since_release'] <= end_month), 'mldr'] = mldr

                # Calculate percent change and adjust decay rates
                final_df['percent_change'] = ((final_df['mldr'] - final_df['decay_rate']) / final_df['decay_rate']) * 100
                average_percent_change = final_df['percent_change'].mean()
                
                if average_percent_change > 0:
                    weight = min(1, max(0, average_percent_change / 100))
                else:
                    weight = 0
                    
                final_df['adjusted_decay_rate'] = final_df['decay_rate'] * (1 + (average_percent_change * weight) / 100)
                
                # Add new decay rate and calculate final adjustment
                final_df.loc[(final_df['months_since_release'] >= start_month) & 
                            (final_df['months_since_release'] <= end_month), 'new_decay_rate'] = k

                final_df['percent_change_new_vs_adjusted'] = ((final_df['new_decay_rate'] - final_df['adjusted_decay_rate']) / final_df['adjusted_decay_rate']) * 100
                average_percent_change_new_vs_adjusted = final_df['percent_change_new_vs_adjusted'].mean()

                weight_new = 1 if average_percent_change_new_vs_adjusted > 0 else 0
                final_df['final_adjusted_decay_rate'] = final_df['adjusted_decay_rate'] * (1 + (average_percent_change_new_vs_adjusted * weight_new) / 100)

                # Calculate decay rates using the same method as tab2
                segments = []
                avg_decay_rates = []

                for i in range(len(breakpoints) - 1):
                    start = breakpoints[i]
                    end = breakpoints[i + 1] - 1
                    segment_data = final_df[(final_df['months_since_release'] >= start) & (final_df['months_since_release'] <= end)]
                    avg_decay_rate = segment_data['final_adjusted_decay_rate'].mean()
                    segments.append(i + 1)
                    avg_decay_rates.append(avg_decay_rate)

                consolidated_df = pd.DataFrame({
                    'segment': segments,
                    'k': avg_decay_rates
                })

                # Generate forecasts using the same method as tab2
                forecasts = forecast_values(consolidated_df, last_month_streams, months_since_release_total, validation_months)
                forecast_df = pd.DataFrame(forecasts)
                
                # Convert month numbers to actual dates
                last_training_date = training_df['Date'].max()
                forecast_df['Date'] = [last_training_date + pd.DateOffset(months=i+1) for i in range(len(forecast_df))]
                forecast_df['YearMonth'] = forecast_df['Date'].dt.strftime('%Y-%m')
                forecast_df['forecasted_value'] = forecast_df['forecasted_value']  # These are already monthly values

                # Ensure we have matching months between actual and predicted
                monthly_actual = monthly_actual.set_index('YearMonth')
                monthly_predicted = forecast_df.set_index('YearMonth')

                # Get common months
                common_months = monthly_actual.index.intersection(monthly_predicted.index)

                # Filter both DataFrames to only include common months
                monthly_actual = monthly_actual.loc[common_months].reset_index()
                monthly_predicted = monthly_predicted.loc[common_months].reset_index()

                # Calculate accuracy metrics using monthly data
                actual_values = monthly_actual['Value'].values
                predicted_values = monthly_predicted['forecasted_value'].values
                
                # MAPE calculation
                mape = np.mean(np.abs((actual_values - predicted_values) / actual_values)) * 100
                
                # MAE calculation
                mae = np.mean(np.abs(actual_values - predicted_values))
                
                # R-squared calculation
                ss_res = np.sum((actual_values - predicted_values) ** 2)
                ss_tot = np.sum((actual_values - np.mean(actual_values)) ** 2)
                r2 = 1 - (ss_res / ss_tot)
                
                # Monthly variance calculation
                monthly_variance = np.std(np.abs((actual_values - predicted_values) / actual_values)) * 100
                
                # Display results
                st.write("\nAccuracy Metrics:")
                st.write(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
                st.write(f"Mean Absolute Error (MAE): {mae:.2f}")
                st.write(f"R-squared Value: {r2:.4f}")
                st.write(f"Monthly Variance: {monthly_variance:.2f}%")
                
                # Create visualization of actual vs predicted values (monthly)
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.plot(monthly_actual['Date'], monthly_actual['Value'], 
                       label='Actual', color='blue', marker='o')
                ax.plot(monthly_predicted['Date'], monthly_predicted['forecasted_value'], 
                       label='Predicted', color='red', linestyle='--', marker='o')
                ax.set_title(f'Monthly Actual vs Predicted Streams - {track_split["track_name"]}')
                ax.set_xlabel('Month')
                ax.set_ylabel('Monthly Streams')
                ax.legend()
                plt.xticks(rotation=45)
                plt.tight_layout()
                st.pyplot(fig)
                
                # Create monthly comparison table
                comparison_df = pd.DataFrame({
                    'Month': monthly_actual['Date'],
                    'Actual Monthly Streams': monthly_actual['Value'],
                    'Predicted Monthly Streams': monthly_predicted['forecasted_value']
                })
                comparison_df['Absolute Error'] = np.abs(comparison_df['Actual Monthly Streams'] - comparison_df['Predicted Monthly Streams'])
                comparison_df['Percentage Error'] = (comparison_df['Absolute Error'] / comparison_df['Actual Monthly Streams']) * 100

                # Calculate all accuracy metrics
                mape = np.mean(np.abs((actual_values - predicted_values) / actual_values)) * 100
                mae = np.mean(np.abs(actual_values - predicted_values))
                r2 = r2_score(actual_values, predicted_values)
                monthly_variance = np.std(np.abs((actual_values - predicted_values) / actual_values)) * 100
                
                # Store track-specific metrics (all numeric, no formatting)
                track_metrics = {
                    'Track': track_split['track_name'],
                    'MAPE': mape,  # Store raw numeric value
                    'MAE': mae,
                    'R-squared': r2,
                    'Monthly Variance': monthly_variance,
                    'Mean Monthly Streams (Actual)': np.mean(actual_values),
                    'Mean Monthly Streams (Predicted)': np.mean(predicted_values),
                    'Total Streams (Actual)': np.sum(actual_values),
                    'Total Streams (Predicted)': np.sum(predicted_values)
                }
                
                # Add metrics to list for overall calculations
                track_metrics_list.append(track_metrics)
                
                # Create formatted version for display only
                display_metrics = {
                    'Track': track_split['track_name'],
                    'MAPE (%)': f"{mape:.2f}%",
                    'MAE': f"{int(mae):,}",
                    'R-squared': f"{r2:.4f}",
                    'Monthly Variance (%)': f"{monthly_variance:.2f}%",
                    'Mean Monthly Streams (Actual)': f"{int(np.mean(actual_values)):,}",
                    'Mean Monthly Streams (Predicted)': f"{int(np.mean(predicted_values)):,}",
                    'Total Streams (Actual)': f"{int(np.sum(actual_values)):,}",
                    'Total Streams (Predicted)': f"{int(np.sum(predicted_values)):,}"
                }

                # Display track-specific metrics
                st.write(f"\nAccuracy Metrics for {track_split['track_name']}:")
                metrics_df = pd.DataFrame([display_metrics]).set_index('Track')
                st.write(metrics_df)

                # Format numbers with commas in comparison table
                comparison_df['Actual Monthly Streams'] = comparison_df['Actual Monthly Streams'].apply(lambda x: f"{int(x):,}")
                comparison_df['Predicted Monthly Streams'] = comparison_df['Predicted Monthly Streams'].apply(lambda x: f"{int(x):,}")
                comparison_df['Absolute Error'] = comparison_df['Absolute Error'].apply(lambda x: f"{int(x):,}")
                comparison_df['Percentage Error'] = comparison_df['Percentage Error'].apply(lambda x: f"{x:.2f}%")
                
                # Display monthly comparison
                st.write("\nMonthly Comparison:")
                comparison_df.set_index('Month', inplace=True)
                st.write(comparison_df)
                
                # Create visualization
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
                
                # Plot 1: Actual vs Predicted
                ax1.plot(monthly_actual['Date'], actual_values, 
                        label='Actual', color='blue', marker='o')
                ax1.plot(monthly_predicted['Date'], predicted_values, 
                        label='Predicted', color='red', linestyle='--', marker='o')
                ax1.set_title(f'Monthly Actual vs Predicted Streams - {track_split["track_name"]}')
                ax1.set_xlabel('Month')
                ax1.set_ylabel('Monthly Streams')
                ax1.legend()
                plt.setp(ax1.xaxis.get_ticklabels(), rotation=45)
                
                # Plot 2: Percentage Error Over Time
                percentage_errors = (np.abs(actual_values - predicted_values) / actual_values) * 100
                ax2.plot(monthly_actual['Date'], percentage_errors, 
                        color='purple', marker='o')
                ax2.set_title('Prediction Error Over Time')
                ax2.set_xlabel('Month')
                ax2.set_ylabel('Percentage Error (%)')
                plt.setp(ax2.xaxis.get_ticklabels(), rotation=45)
                
                plt.tight_layout()
                st.pyplot(fig)

            # After processing all tracks, calculate and display overall metrics
            if track_metrics_list:
                st.write("\nOverall Model Performance:")
                overall_metrics = pd.DataFrame(track_metrics_list)
                
                overall_summary = pd.DataFrame({
                    'Metric': [
                        'Average MAPE',
                        'Average MAE',
                        'Average R-squared',
                        'Average Monthly Variance',
                        'Number of Tracks Analyzed'
                    ],
                    'Value': [
                        f"{overall_metrics['MAPE'].mean():.2f}%",  # Using new column name
                        f"{int(overall_metrics['MAE'].mean()):,}",
                        f"{overall_metrics['R-squared'].mean():.4f}",
                        f"{overall_metrics['Monthly Variance'].mean():.2f}%",  # Using new column name
                        f"{len(track_metrics_list)}"
                    ]
                })
                st.write(overall_summary)
