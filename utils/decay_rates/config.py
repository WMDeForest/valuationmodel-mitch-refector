"""
Configuration parameters for decay rate modeling and forecasting.

This module centralizes key configuration values that control the behavior
of the decay model forecasting system, separating them from application code.
"""

# Default stream influence factor - controls decay rate adjustment
# Higher values generally result in slower decay rates
# Range: Should be between 0-1,000,000, with 1000 being in lowest range (0-10,000)
DEFAULT_STREAM_INFLUENCE_FACTOR = 1000

# Number of months to forecast track streams into the future (for full decay curves)
# 400 months = ~33 years of track stream forecasting
DEFAULT_TRACK_STREAMS_FORECAST_PERIOD = 400

# Default number of years to use for valuation calculations
# Industry standard is typically 20 years for music asset valuation
DEFAULT_VALUATION_CALCULATION_YEARS = 20 