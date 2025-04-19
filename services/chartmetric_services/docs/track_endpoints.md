# ChartMetric API - Track Endpoints

This document contains the API documentation for retrieving track data from the ChartMetric API.

## Get Track Details

Retrieves detailed information about a track.

### Endpoint
```
GET /track/{track_id}
```

### Parameters
- `track_id` (integer, required): The ChartMetric ID of the track

### Example Usage
```python
from services.chartmetric_services import chartmetric_service as chartmetric

# Get track details
track_details = chartmetric.get_track_detail(track_id=54321)
```

### Example Response
```json
{
  "id": 54321,
  "name": "Track Name",
  "isrc": "USABC1234567",
  "release_date": "2023-01-15",
  "duration_ms": 216000,
  "album": {
    "id": 9876,
    "name": "Album Name"
  }
}
```

## Get Track Spotify Streams Comparison

Retrieves streaming data for a track on Spotify.

### Endpoint
```
GET /track/{track_id}/spotify/streams
```

### Parameters
- `track_id` (integer, required): The ChartMetric ID of the track
- `start_date` (string, optional): Start date for data in format YYYY-MM-DD
- `end_date` (string, optional): End date for data in format YYYY-MM-DD

### Example Usage
```python
from services.chartmetric_services import chartmetric_service as chartmetric

# Get track streaming data
streams_data = chartmetric.get_track_sp_streams_campare(track_id=54321)
```

### Example Response
```json
[
  {
    "timestp": "2023-01-16",
    "value": 10000
  },
  {
    "timestp": "2023-01-17",
    "value": 25000
  }
]
```

## Search Tracks

Searches for tracks by name or ISRC.

### Endpoint
```
GET /search/track
```

### Parameters
- `query` (string, required): The search term
- `search_type` (string, optional): Type of search. Options: "name", "isrc"
- `limit` (integer, optional): Number of results to return (default: 10)

### Example Usage
```python
from services.chartmetric_services import chartmetric_service as chartmetric

# Search for a track
results = chartmetric.search_track(query="Track Name", search_type="name", limit=5)
```

### Example Response
```json
[
  {
    "id": 54321,
    "name": "Track Name",
    "isrc": "USABC1234567",
    "artist": {
      "id": 12345,
      "name": "Artist Name"
    },
    "release_date": "2023-01-15"
  }
]
```

## Get Track Stats

Returns various statistics for a track across different platforms.

### Endpoint
```
GET /track/:id/:platform/stats/:mode
```

### Parameters

| Field | Type | Description |
|-------|------|-------------|
| id | Integer | Chartmetric track ID when isDomainId=false, platform ID when isDomainId=true |
| platform | String | Streaming platform to retrieve data for.<br>Allowed values: `chartmetric`, `spotify`, `youtube`, `shazam`, `tiktok`, `genius`, `soundcloud` |
| mode | String | Defines the algorithm used to select tracks when more than one option is available.<br>`highest-playcounts`: selects the track with the highest average stat in the last 3 months.<br>`most-history`: selects the track with the longest historical data.<br>Allowed values: `highest-playcounts`, `most-history` |
| since | String | (Optional) Start of date range in ISO date format (e.g. 2017-03-25). Default: 180 days ago |
| until | String | (Optional) End of date range in ISO format (e.g. 2017-03-25). Default: today |
| latest | Boolean | (Optional) If set to true, returns latest data point available regardless of date |
| interpolated | Boolean | (Optional) If true, returns interpolated data for missing dates. Default: false |
| type | String | (Optional) Specifies the type of statistic to return (applicable only when platform=spotify or platform=chartmetric)<br>Allowed values when platform=spotify:<br>- `popularity` (default) — the Spotify popularity value for this track<br>- `streams` — the number of streams on Spotify for this track<br>Allowed values when platform=chartmetric:<br>- `score` (default) — the Chartmetric score for this track (rounded; multiple tracks may have the same score)<br>- `rank` — the Chartmetric rank for this track (unique; only one track is assigned to a rank value) |
| isDomainId | Boolean | (Optional) If true, the id passed in the request parameter will be considered the platform ID. If false the :id will be considered Chartmetric's track ID. Default: false |

### Example Usage
```
# Spotify, highest-playcounts
curl -H 'Authorization: Bearer [ACCESS KEY]' https://api.chartmetric.com/api/track/15611361/spotify/stats/highest-playcounts?since=2022-01-01

# Spotify with isDomainId
curl -H 'Authorization: Bearer [ACCESS KEY]' https://api.chartmetric.com/api/track/15611361/spotify/stats/highest-playcounts?isDomainId=true
```

### Example Usage in Python
```python
from services.chartmetric_services import chartmetric_service as chartmetric

# Get Spotify track stats with highest playcounts
spotify_stats = chartmetric.get_track_stats(
    track_id=15611361,
    platform="spotify",
    mode="highest-playcounts",
    since="2022-01-01"
)

# Get TikTok track stats
tiktok_stats = chartmetric.get_track_stats(
    track_id=15611361,
    platform="tiktok",
    mode="most-history"
)
```

### Response Format

| Field | Type | Description |
|-------|------|-------------|
| domain | String | Domain |
| track_domain_id | String | Track domain ID |
| data | Object[] | Array of stat objects for each date |
| » value | Integer | Statistic from given service. Either popularity or streams for Spotify, views for YouTube, play count for Shazam and SoundCloud, posts count for TikTok, page views for Genius, and track score or rank for Chartmetric |
| » timestp | Date | The time this statistic was achieved |
| » daily_diff | Integer | The difference between current and previous statistic (not applicable for Spotify). This field is called diff if interpolated=false |
| » interpolated | Boolean | (Optional) If true, the value field did not come directly from the service, but is rather an interpolated value to create a smooth curve (not applicable for Spotify). Default: false | 