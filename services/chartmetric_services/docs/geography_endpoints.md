# ChartMetric API - Geography Endpoints

This document contains the API documentation for retrieving geographical audience data from the ChartMetric API.

## Get Artist Audience Geography

Retrieves information about where an artist's audience is located.

### Endpoint
```
GET /artist/{artist_id}/where-people-listen
```

### Parameters
- `artist_id` (integer, required): The ChartMetric ID of the artist

### Example Usage
```python
from services.chartmetric_services import chartmetric_service as chartmetric

# Get audience geography data
geography_data = chartmetric.get_artist_track_where_people_listen(artist_id=12345)
```

### Example Response
```json
[
  {
    "code2": "US",
    "country_name": "United States",
    "listeners": 500000,
    "population": 331900000
  },
  {
    "code2": "GB",
    "country_name": "United Kingdom",
    "listeners": 250000,
    "population": 67220000
  }
]
```

## Get Track Audience Geography

Retrieves information about where a track's audience is located.

### Endpoint
```
GET /track/{track_id}/where-people-listen
```

### Parameters
- `track_id` (integer, required): The ChartMetric ID of the track

### Example Usage
```python
from services.chartmetric_services import chartmetric_service as chartmetric

# Get track audience geography data
track_geography = chartmetric.get_track_where_people_listen(track_id=54321)
```

### Example Response
```json
[
  {
    "code2": "US",
    "country_name": "United States",
    "listeners": 100000,
    "population": 331900000
  },
  {
    "code2": "GB",
    "country_name": "United Kingdom",
    "listeners": 50000,
    "population": 67220000
  }
]
```

## Get City-Level Audience Data

Retrieves more granular city-level information about an artist's audience.

### Endpoint
```
GET /artist/{artist_id}/cities
```

### Parameters
- `artist_id` (integer, required): The ChartMetric ID of the artist
- `country_code` (string, optional): Filter results by two-letter country code

### Example Usage
```python
from services.chartmetric_services import chartmetric_service as chartmetric

# Get city-level audience data
city_data = chartmetric.get_artist_cities(artist_id=12345, country_code="US")
```

### Example Response
```json
[
  {
    "city": "New York",
    "listeners": 50000,
    "population": 8804190
  },
  {
    "city": "Los Angeles",
    "listeners": 35000,
    "population": 3898747
  }
]
``` 