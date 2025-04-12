"""
Core mathematical functions for decay modeling.
"""
import numpy as np

def piecewise_exp_decay(x, S0, k):
    """
    Core piecewise exponential decay function.
    
    Args:
        x: Time variable (typically months since release)
        S0: Initial value
        k: Decay rate constant
        
    Returns:
        Decayed value at time x
    """
    return S0 * np.exp(-k * x)

def exponential_decay(x, a, b):
    """
    Simpler exponential decay function used for initial fitting.
    
    Args:
        x: Time variable (typically months since release)
        a: Initial value
        b: Decay rate constant
        
    Returns:
        Decayed value at time x
    """
    return a * np.exp(-b * x) 