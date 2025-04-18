# Track Forecasting Pipeline

This directory contains scripts for generating and managing track stream forecasts using historical data stored in the database.

## Key Files

- `forecast_streams.py`: Main script that connects to the database, retrieves training data, generates forecasts, and stores results
- `test_forecast.py`: Testing script for validating the forecasting process on a single track

## Database Tables

The scripts interact with the following database tables:

1. **backtest_track_daily_training_data**: Source of historical track streaming data
   - `id`: Unique identifier
   - `cm_track_id`: ChartMetric track identifier
   - `cm_artist_id`: ChartMetric artist identifier
   - `date`: Date of the streaming data
   - `daily_streams`: Number of streams on that date
   - `days_from_release`: Days since the track was released
   - `created_at`: Timestamp when the record was created

2. **backtest_artist_mldr**: Source of artist Monthly Listener Decay Rate values
   - `id`: Unique identifier
   - `cm_artist_id`: ChartMetric artist identifier
   - `mldr`: Monthly Listener Decay Rate value
   - `created_at`: Timestamp when the record was created
   - `backtest_artist_daily_training_data_id`: Reference to training data

3. **backtest_track_streams_forecast**: Destination for forecast results
   - `id`: Unique identifier
   - `month`: Month number in the forecast (1, 2, 3, etc.)
   - `forecasted_value`: Predicted number of streams for that month
   - `segment_used`: Lifecycle segment used for this forecast (1, 2, 3, etc.)
   - `time_used`: Time period used in forecast calculation
   - `cm_track_id`: ChartMetric track identifier
   - `cm_artist_id`: ChartMetric artist identifier
   - `training_data_id`: Reference to the training data used
   - `created_at`: Timestamp when the forecast was created

## Forecasting Process

The forecasting process follows these steps:

1. Connect to the database and retrieve tracks that need forecasting
2. For each track:
   - Retrieve the track's daily streaming data
   - Calculate key metrics (last 30 days streams, months since release)
   - Get the artist's MLDR value from the artist table
   - Generate decay parameters and apply adjustments
   - Calculate segmented decay rates for different lifecycle phases
   - Generate stream forecasts for each future month
   - Store the forecasts in the database

## Usage

### Running the Main Forecasting Script

```bash
# From the project root directory
python backtest_pipeline/track_forecasts/forecast_streams.py
```

This will process all tracks that don't yet have forecasts in the database.

### Testing a Single Track

```bash
# From the project root directory
python backtest_pipeline/track_forecasts/test_forecast.py
```

This will:
1. Find a track with available data that hasn't been forecasted yet
2. Generate forecasts for that track
3. Output diagnostic information
4. Save the forecasts to a CSV file for inspection

## Notes

- The script uses the same forecasting algorithm as the Streamlit app, ensuring consistency
- Artist MLDR values are critical for accurate forecasting - tracks without an artist MLDR will be skipped
- The forecasting period is set to 24 months (2 years) for backtest pipeline, while the Streamlit app uses 400 months
- Track decay rates are segmented based on `track_lifecycle_segment_boundaries` from the decay_rates module

## Performance Optimization

The forecasting process has been heavily optimized for performance. Key optimizations include:

### Batch Processing
- **Original approach**: Processed one track at a time (2 database queries per track)
- **Optimized approach**: Processes tracks in batches of 100 (2 queries per batch of 100 tracks)
- **Improvement**: 50x reduction in database queries

### Parallel Processing
- **Original approach**: Single-threaded, sequential processing
- **Optimized approach**: Multi-threaded using Python's `concurrent.futures` module
- **CPU utilization**: Automatically scales to use all available CPU cores
- **Improvement**: Near-linear scaling with CPU cores

### Data Size Reduction
- **Original approach**: Used floating-point numbers with many decimal places
- **Optimized approach**: Rounded stream forecasts to integers (matching database column type)
- **Improvement**: Reduced memory usage and conversion overhead

### Reduced Type Conversions
- **Original approach**: Multiple redundant type checks and conversions
- **Optimized approach**: Single conversion at data source, direct use afterward
- **Improvement**: Eliminated thousands of redundant operations per batch

### Performance Results

Recent benchmark (April 2025):

| Metric | Original | Optimized | Improvement |
|--------|----------|-----------|-------------|
| Total processing time | ~7.4 hours | 12 minutes | 37x faster |
| Tracks per second | 0.32 | 11.7 | 37x improvement |
| Average time per track | 3.16 seconds | 0.09 seconds | 35x faster |
| Database queries | ~17,000 | ~170 | 100x reduction |
| Success rate | N/A | 99.8% (8,443/8,459) | N/A |

These optimizations make the backtest pipeline practical for daily use and rapid iteration, dramatically reducing the time needed to generate forecasts for the entire catalog.
