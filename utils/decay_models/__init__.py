"""
Decay models package for music streaming valuation models.
This package contains modules for decay rate estimation, forecasting, 
and other utilities for modeling streaming decay patterns.
"""

from utils.decay_models.core import piecewise_exp_decay, exponential_decay
from utils.decay_models.preprocessing import remove_anomalies
from utils.decay_models.fitting import calculate_decay_rate, fit_segment
from utils.decay_models.parameter_updates import update_fitted_params
from utils.decay_models.forecasting import forecast_values

# Expose all relevant functions for easy importing
__all__ = [
    'piecewise_exp_decay',
    'exponential_decay',
    'remove_anomalies',
    'calculate_decay_rate',
    'fit_segment',
    'update_fitted_params',
    'forecast_values',
] 