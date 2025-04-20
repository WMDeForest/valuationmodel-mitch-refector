# ChartMetric API - General Usage

This document contains general information about using the ChartMetric API, including authentication, rate limits, and common patterns.

## Authentication

The ChartMetric API uses token-based authentication. You need to authenticate with your credentials to receive a token, which is then used for subsequent requests.

### Getting an Access Token

```python
from services.chartmetric_services import chartmetric_service as chartmetric

# Authentication is handled automatically when making API calls
# But you can also manually refresh the token
chartmetric.refresh_token()
```

### Token Lifespan

Tokens are valid for 24 hours. Our service automatically handles token refreshes when needed.

## Rate Limits

ChartMetric API has rate limits to prevent abuse:

- 100 requests per minute
- 1000 requests per day

If you exceed these limits, the API will return a 429 (Too Many Requests) error. Our service implements exponential backoff retry logic for these cases.

## Search - ChartMetric Search Engine

Search tracks, albums, artists, curators, playlists, and more with one single query. The search API also supports searching by URLs.

### Endpoint
```
GET /search
```

### Parameters

| Field | Type | Description |
|-------|------|-------------|
| q | String | Search query. Can also be a URL such as:<br>- `https://open.spotify.com/artist/7ENzCHnmJUr20nUjoZ0zZ1`<br>- `https://itunes.apple.com/us/artist/snarky-puppy/152987454`<br>- `https://www.deezer.com/artist/471526` |
| limit | Number | (Optional) The number of entries to be returned. Default: 10 |
| offset | Number | (Optional) The offset of entries to be returned. Default: 0 |
| type | String | (Optional) The type of search for the query. Default: all<br>Allowed values: `all`, `artists`, `tracks`, `playlists`, `curators`, `albums`, `stations`, `cities`, `songwriters` |
| triggerCitiesOnly | Boolean | (Optional) Only return trigger cities. Available for both beta=true and beta=false requests. Default: false |
| beta | Boolean | (Optional) Enable the improved beta search engine for higher relevance and accuracy. Default: false |
| platforms | String[] | (Optional) Platforms to search. Available only for beta=true requests.<br>Allowed values: `cm`, `spotify`, `itunes`, `applemusic`, `deezer`, `amazon`, `youtube`, `tidal`, `soundcloud`, `acr`<br>Default: [] |

### Example Usage
```
# Basic search
curl -H 'Authorization: Bearer [ACCESS KEY]' https://api.chartmetric.com/api/search?q=Ariana&limit=10

# Using beta search
curl -H 'Authorization: Bearer [ACCESS KEY]' https://api.chartmetric.com/api/search?q=Ariana&beta=true

# Search for trigger cities only
curl -H 'Authorization: Bearer [ACCESS KEY]' https://api.chartmetric.com/api/search?q=New&triggerCitiesOnly=true
```

### Example Usage in Python
```python
from services.chartmetric_services import chartmetric_service as chartmetric

# Basic search
search_results = chartmetric.search(query="Ariana Grande", limit=10)

# Search for specific type
artist_results = chartmetric.search(
    query="Ariana Grande", 
    type="artists", 
    limit=5
)

# Search using beta engine with specific platforms
beta_results = chartmetric.search(
    query="https://open.spotify.com/artist/7ENzCHnmJUr20nUjoZ0zZ1",
    beta=True,
    platforms=["spotify", "youtube"]
)
```

### Response Format

| Field | Type | Description |
|-------|------|-------------|
| artists | Object[] | List of artists |
| » id | Number | Chartmetric artist ID |
| » name | String | Artist name |
| » image_url | String | Image URL |
| » isni | String | Artist ISNI |
| » sp_monthly_listeners | Number | Number of Spotify Monthly Listeners |
| » sp_followers | Number | Number of Spotify Followers |
| » cm_artist_score | Number | Chartmetric Artist Score |
| playlists | Object | List of playlists by platform |
| » spotify | Object[] | List of Spotify playlists |
| »» id | Number | Chartmetric playlist ID |
| »» name | String | Playlist name |
| »» image_url | String | Image URL |
| »» owner_name | String | Curator's name |
| » applemusic | Object[] | List of Apple Music playlists |
| » deezer | Object[] | List of Deezer playlists |
| » amazon | Object[] | List of Amazon playlists |
| » youtube | Object[] | List of YouTube playlists |
| tracks | Object[] | List of tracks |
| » id | String | Chartmetric track ID |
| » name | String | Track name |
| » image_url | String | Image URL |
| » isrc | String | Track ISRC |
| » artist_names | String[] | Artists names |
| curators | Object | List of curators by platform |
| » spotify | Object[] | List of Spotify curators |
| » applemusic | Object[] | List of Apple Music curators |
| » deezer | Object[] | List of Deezer curators |
| albums | Object[] | List of albums |
| » id | Number | Chartmetric album ID |
| » name | String | Album name |
| » image_url | String | Image URL |
| » label | String | Music label |
| stations | Object[] | List of stations |
| » id | Number | Chartmetric station ID |
| » name | String | Station name |
| » genre | String | Genre of the station |
| » image_url | String | Image URL |
| » station_city | String | City of the station |
| » country | String | Country of the station |
| labels | Array | List of labels |
| cities | Object[] | List of cities |
| » id | Number | Chartmetric city ID |
| » name | String | City name |
| » name_ascii | String | City name (ASCII) |
| » population | Number | City population |
| » country | String | Country of the city |
| » code2 | String | Country code of the city |
| » province | String | State or province of the city |
| » trigger_city | Boolean | Whether the city is considered a Trigger City |
| songwriters | Object[] | List of songwriters |
| » name | String | Songwriter name |
| » doc_id | Number | Chartmetric songwriter ID |
| » artistName | String | Songwriter's name as an artist |
| » image_url | String | Image URL |
| suggestions | Object[] | List of search results (beta=true only) |
| match_strength | Float | Match strength of the search result (beta=true only) |
| target | String | Target of the search result (beta=true only) |
| platform | String | Platform of the search result (beta=true only) |
| name | String | Name of the search result (beta=true only) |
| image_url | String | Image URL of the search result (beta=true only) |

## Error Handling

The ChartMetric service wrapper handles common errors:

- 401: Authentication errors (automatically refreshes token)
- 404: Resource not found
- 429: Rate limit exceeded (implements backoff)
- 500: Server errors

Example error handling:

```python
from services.chartmetric_services import chartmetric_service as chartmetric
from services.chartmetric_services.exceptions import ChartMetricApiError

try:
    data = chartmetric.get_artist_spotify_stats(artist_id=12345)
except ChartMetricApiError as e:
    print(f"API Error: {e}")
```

## Data Types

The ChartMetric service uses data transfer objects (DTOs) to provide type safety:

```python
from services.chartmetric_services.dto import ArtistStateCampareListner, TrackSpotifyState, CountryListeners

# These are the return types from the API methods
```

## Common Parameters

Many API endpoints share common parameters:

- `start_date`: First date to include in results (format: YYYY-MM-DD)
- `end_date`: Last date to include in results (format: YYYY-MM-DD)
- `limit`: Number of results to return (default varies by endpoint)
- `offset`: Number of results to skip (for pagination)

## API Base URL

```
https://api.chartmetric.com/api
```

This base URL is configured in the service and should not need to be changed. 