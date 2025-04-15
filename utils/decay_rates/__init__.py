"""
This module provides a comprehensive interface for accessing decay rate data and parameters
used in the streaming prediction model. It combines three main components:

1. Volume Ranges (from volume_ranges.py):
   - Defines streaming volume categories (0-10,000, 10,000-30,000, etc.)
   - Used to categorize tracks based on their current streaming volume

2. SP_REACH Data (from sp_reach.py):
   - Contains decay rates for different streaming volume ranges
   - Organized by time segments (1-12 months, 3-5 months, etc.)
   - Empirically derived from historical streaming data

3. Fitted Parameters (from fitted_params.py):
   - Contains S0 (initial value) and k (decay rate) for each time segment
   - Includes breakpoints defining the boundaries between segments
   - Used in the exponential decay model for predictions

4. Configuration Parameters (from config.py):
   - Default parameter values used in the forecasting model
   - Includes stream influence factor and forecast periods

The module provides a clean interface for accessing all these components, making it easy
to use the decay rate model in other parts of the application.
"""

from .volume_ranges import ranges_sp, sp_range
from .sp_reach import SP_REACH_DATA, SP_REACH
from .fitted_params import fitted_params, fitted_params_df, track_lifecycle_segment_boundaries
from .config import DEFAULT_STREAM_INFLUENCE_FACTOR, DEFAULT_FORECAST_PERIODS, DEFAULT_FORECAST_YEARS

__all__ = [
    'ranges_sp',
    'sp_range',
    'SP_REACH_DATA',
    'SP_REACH',
    'fitted_params',
    'fitted_params_df',
    'track_lifecycle_segment_boundaries',
    'DEFAULT_STREAM_INFLUENCE_FACTOR',
    'DEFAULT_FORECAST_PERIODS',
    'DEFAULT_FORECAST_YEARS'
] 