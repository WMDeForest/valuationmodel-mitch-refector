# ChartMetric API Documentation

This directory contains documentation for the ChartMetric API integration used in this application.

## Contents

- [General Usage](general_usage.md) - Authentication, rate limits, and common patterns
- [Artist Endpoints](artist_endpoints.md) - API endpoints for artist data
- [Track Endpoints](track_endpoints.md) - API endpoints for track data
- [Geography Endpoints](geography_endpoints.md) - API endpoints for geographical audience data

## Quick Start

```python
from services.chartmetric_services import chartmetric_service as chartmetric

# Get artist monthly listeners
listeners_data = chartmetric.get_artist_spotify_stats(artist_id=12345)

# Get track streaming data
streams_data = chartmetric.get_track_sp_streams_campare(track_id=54321)

# Get audience geography data
geography_data = chartmetric.get_artist_track_where_people_listen(artist_id=12345)
```

## Implementation Details

The ChartMetric service is implemented as a wrapper around the ChartMetric API. It handles:

1. Authentication
2. Rate limiting
3. Error handling
4. Data type conversion

See the [chartmetric.py](../chartmetric.py) file for implementation details.

## Data Types

The service uses data transfer objects (DTOs) defined in [dto.py](../dto.py) to provide type safety for the API responses.

## Related Files

- [chartmetric.py](../chartmetric.py) - Main service implementation
- [dto.py](../dto.py) - Data transfer objects
- [exceptions.py](../exceptions.py) - Custom exceptions
- [http_client.py](../http_client.py) - HTTP client for API requests
- [urlpatterns.py](../urlpatterns.py) - API URL patterns 