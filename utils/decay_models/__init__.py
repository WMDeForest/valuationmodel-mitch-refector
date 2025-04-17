"""
Decay Models Package
-------------------

This package contains functions and modules for analyzing and modeling
the decay of monthly listener counts over time.

Core analysis workflow:
1. Clean and preprocess streaming data with preprocessing.remove_anomalies()
2. Calculate artist-level decay rate using fitting.fit_decay_curve()
3. Fine-tune decay model with parameter_updates.update_fitted_params()
4. Generate future forecasts with forecasting.forecast_track_streams()

The central concept is the Monthly Listener Decay Rate (MLDR), which 
quantifies how quickly a track's streams decrease over time.

Recommended Usage:
-----------------
For most applications, use the high-level analyze_listener_decay() function,
which handles the entire workflow in one call:

```python
from utils.decay_models import analyze_listener_decay

# Process data from any source (CSV, API, database)
results = analyze_listener_decay(df_monthly_listeners)

# Access results
mldr = results['mldr']  # Monthly Listener Decay Rate
popt = results['popt']  # Fitted parameters
plot_data = results['subset_df']  # DataFrame with processed data for visualization
```

Advanced Usage:
-------------
For more fine-grained control, you can use the individual functions:

```python
from utils.decay_models import remove_anomalies, fit_decay_curve, update_fitted_params, forecast_track_streams

# Clean data and handle anomalies
monthly_data = remove_anomalies(streaming_df)

# Calculate decay rate
mldr, params = fit_decay_curve(monthly_data)

# Generate forecasts
forecasts = forecast_track_streams(params_df, initial_value, start_period, periods)
```
"""

from utils.decay_models.core import piecewise_exp_decay, exponential_decay
from utils.data_processing import remove_anomalies
from utils.decay_models.fitting import (
    fit_decay_curve, 
    fit_segment, 
    analyze_listener_decay,
    prepare_decay_rate_fitting_data
)
from utils.decay_models.parameter_updates import (
    update_fitted_params, 
    get_decay_parameters,
    generate_track_decay_rates_by_month,
    create_decay_rate_dataframe,
    adjust_track_decay_rates,
    calculate_track_decay_rates_by_segment
)
from utils.decay_models.forecasting import calculate_monthly_stream_projections

# For backward compatibility
from utils.decay_models.fitting import calculate_decay_rate, calculate_monthly_listener_decay

__all__ = [
    'piecewise_exp_decay',
    'exponential_decay',
    'remove_anomalies',
    'fit_decay_curve',
    'analyze_listener_decay',
    'fit_segment',
    'prepare_decay_rate_fitting_data',
    'update_fitted_params',
    'get_decay_parameters',
    'calculate_monthly_stream_projections',
    'generate_track_decay_rates_by_month',
    'create_decay_rate_dataframe',
    'adjust_track_decay_rates',
    'calculate_track_decay_rates_by_segment',
    # For backward compatibility
    'calculate_decay_rate',
    'calculate_monthly_listener_decay',
]

# For backward compatibility
forecast_track_streams = calculate_monthly_stream_projections 