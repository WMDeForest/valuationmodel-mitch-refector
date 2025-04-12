"""
This module defines the streaming volume ranges used in the decay rate calculations.
These ranges are used to categorize tracks based on their streaming volume, which affects
how their decay rates are calculated.

The ranges are structured as follows:
- Column 1: Range identifier (1-10)
- RangeStart: Lower bound of each streaming volume range
- RangeEnd: Upper bound of each streaming volume range

For example:
- Range 1: 0-10,000 streams
- Range 2: 10,000-30,000 streams
- Range 3: 30,000-50,000 streams
...
- Range 10: 950,000+ streams

These ranges are used in conjunction with SP_REACH data to determine appropriate decay rates
for tracks based on their streaming volume. The ranges help create a more nuanced prediction
model that accounts for different streaming patterns at various volume levels.
"""

import pandas as pd

# Define streaming volume ranges
ranges_sp = {
    'Column 1': list(range(1, 11)),
    'RangeStart': [0, 10000, 30000, 50000, 75000, 110000, 160000, 250000, 410000, 950000],
    'RangeEnd': [10000, 30000, 50000, 75000, 110000, 160000, 250000, 410000, 950000, 1E18]
}

# Create DataFrame from ranges
sp_range = pd.DataFrame(ranges_sp) 