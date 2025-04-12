"""
Core mathematical functions for decay modeling.

This module contains the fundamental decay functions used to model
the pattern of music streaming decay over time. These functions serve as
the mathematical foundation for the entire valuation model.
"""
import numpy as np

def piecewise_exp_decay(x, S0, k):
    """
    Core piecewise exponential decay function that models stream decay over time.
    
    This function implements the standard exponential decay formula S(t) = S0 * e^(-kt),
    where S0 is the initial value and k is the decay rate constant. The "piecewise" aspect 
    comes from using different k values for different time segments after release, which is 
    handled in the forecasting module.
    
    Args:
        x: Time variable (typically months since release)
        S0: Initial value (starting number of streams)
        k: Decay rate constant (higher values = faster decay)
        
    Returns:
        Decayed value at time x (predicted streams at month x)
    
    Example:
        >>> piecewise_exp_decay(3, 100000, 0.1)  # After 3 months with decay rate 0.1
        74081.82234034616  # Streams in month 3
    """
    return S0 * np.exp(-k * x)

def exponential_decay(x, a, b):
    """
    Simpler exponential decay function used for initial fitting and MLDR calculation.
    
    This function has the same mathematical form as piecewise_exp_decay but uses
    different parameter names (a, b) to distinguish it as the function used for the
    initial decay rate estimation across the entire dataset, rather than segmented
    decay rates.
    
    Args:
        x: Time variable (typically months since release)
        a: Initial value (equivalent to S0)
        b: Decay rate constant (equivalent to k)
        
    Returns:
        Decayed value at time x
        
    Note:
        The difference between this and piecewise_exp_decay is primarily semantic;
        this function is typically used with curve_fit to find the overall decay rate
        of an artist's catalog, while piecewise_exp_decay is used with segment-specific
        parameters for forecasting.
    """
    return a * np.exp(-b * x) 