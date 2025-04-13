"""
Decay models package for music streaming valuation models.

This package provides a comprehensive framework for modeling, analyzing, and
forecasting music streaming decay patterns over time. It enables accurate
valuation of music assets based on historical streaming data and decay rates.

Overview of Components:
-----------------------
- core: Fundamental mathematical decay functions that form the basis of the model
- preprocessing: Data cleaning and anomaly detection functions
- fitting: Functions for estimating decay rates from historical data
- parameter_updates: Logic for adjusting decay rates based on external factors
- forecasting: Functions for generating future stream predictions

Typical Workflow:
----------------
1. Clean raw streaming data using preprocessing.remove_anomalies()
2. Calculate artist-level decay rate using fitting.calculate_decay_rate()
3. Adjust decay parameters based on playlist reach with parameter_updates.update_fitted_params()
4. Generate stream forecasts using forecasting.forecast_values()
5. Convert stream forecasts to financial projections (handled in main app)

This module architecture separates concerns to allow for easier testing,
maintenance, and future enhancements to the valuation model.
"""

from utils.decay_models.core import piecewise_exp_decay, exponential_decay
from utils.decay_models.preprocessing import remove_anomalies
from utils.decay_models.fitting import calculate_decay_rate, fit_segment, calculate_monthly_listener_decay
from utils.decay_models.parameter_updates import update_fitted_params
from utils.decay_models.forecasting import forecast_values

# Expose all relevant functions for easy importing
__all__ = [
    'piecewise_exp_decay',
    'exponential_decay',
    'remove_anomalies',
    'calculate_decay_rate',
    'calculate_monthly_listener_decay',
    'fit_segment',
    'update_fitted_params',
    'forecast_values',
] 