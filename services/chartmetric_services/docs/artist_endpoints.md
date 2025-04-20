# ChartMetric API - Artist Endpoints

This document contains the API documentation for retrieving artist data from the ChartMetric API.

## Get Artists

Get an artists snapshot of sample artists based on a given metric.

Given the metric filter and score range, get a list of artist ids and all relevant metrics for the given range of the respective metric.

This endpoint is useful for artist profiling based on specific filters such as Spotify performance, YouTube performance and social media performance.

### Endpoint
```
GET /artist/:type/list
```

### Parameters

| Field | Type | Description |
|-------|------|-------------|
| type | String | Platform and metric to filter by. <br>Allowed values: `sp_followers`, `sp_monthly_listeners`, `sp_popularity`, `sp_listeners_to_followers_ratio`, `sp_followers_to_listeners_ratio`, `deezer_fans`, `fs_likes`, `fs_talks`, `ins_followers`, `ts_followers`, `ts_retweets`, `ycs_views`, `ycs_subscribers`, `youtube_daily_video_views`, `youtube_monthly_video_views`, `ws_views`, `soundcloud_followers`, `bs_followers`, `tiktok_followers`, `tiktok_likes`, `cm_artist_rank` |
| min | Integer | Minimum filter threshold (e.g. min 5,000 for Spotify followers) |
| max | Integer | Maximum filter threshold (e.g. max 10,000 for Spotify followers) |
| code2 | String | (Optional) ISO code for the artist country, such as US, JP, IN, etc |
| genreId | Integer | (Optional) The genre ID to filter the results |
| subGenreId | Integer | (Optional) The sub genre ID to filter the results |
| city | String | (Optional) City name to filter the top listeners |
| unsigned | Boolean | (Optional) Whether to return only unsigned artists. Default: false |
| limit | Integer | (Optional) The number of entries to be returned. Max value: 200. Default: 10 |
| offset | Integer | (Optional) Each request returns only certain number of entries. Use offset option to get more results. Max value: 100000. Default: 0 |

### Example Usage
```
curl -H 'Authorization: Bearer [ACCESS KEY]' https://api.chartmetric.com/api/artist/sp_followers/list?min=500&max=10000&offset=0&code2=US&city=Chicago
```

### Example Usage in Python
```python
from services.chartmetric_services import chartmetric_service as chartmetric

# Get artists with Spotify followers between 500-10000 in Chicago, US
artists = chartmetric.get_artists(
    type="sp_followers", 
    min=500, 
    max=10000, 
    code2="US", 
    city="Chicago"
)
```

### Response Format

| Field | Type | Description |
|-------|------|-------------|
| length | Integer | Number of items in the data array |
| data | Array | List of results |
| » chartmetric_artist_id | Integer | Chartmetric artist ID |
| » name | String | Artist name |
| » code2 | String | Country codes |
| » genres | String | Genres |
| » cpp_rank | Integer | Cross-Platform Performance rank |
| » rank_eg | Integer | Engagement Rank |
| » rank_fb | Integer | Fan base Rank |
| » spotify_popularity | Integer | Spotify popularity |
| » spotify_followers | Integer | Spotify followers |
| » spotify_monthly_listeners | Integer | Spotify monthly listeners |
| » spotify_listeners_to_followers_ratio | Float | Spotify listeners to followers ratio |
| » spotify_followers_to_listeners_ratio | Float | Spotify followers to listeners ratio |
| » deezer_fans | Integer | Deezer fans |
| » facebook_likes | Integer | Facebook likes |
| » facebook_talks | Integer | Facebook talks |
| » twitter_followers | Integer | Twitter followers |
| » twitter_retweets | Integer | Twitter retweets |
| » instagram_followers | Integer | Instagram followers |
| » youtube_channel_views | Integer | YouTube channel views |
| » youtube_subscribers | Integer | YouTube subscribers |
| » youtube_daily_video_views | Integer | YouTube daily video views |
| » youtube_monthly_video_views | Integer | YouTube monthly video views |
| » wikipedia_views | Integer | Wikipedia views |
| » soundcloud_followers | Integer | Soundcloud followers |
| » bandsintown_followers | Integer | Bandsintown followers |
| » tiktok_followers | Integer | TikTok followers |
| » tiktok_likes | Integer | TikTok likes |
| » sp_where_people_listen | Array | Spotify listener count and location |
| »» code2 | String | Country code of where people listen |
| »» listeners | String | Number of listeners of the country where people listen |
| »» name | String | Name of the city where people listen |
| » signed | Boolean | Whether the artist has ever been signed to a label |

## Get Artist Spotify Stats

Retrieves monthly listener statistics for an artist on Spotify.

### Endpoint
```
GET /artist/{artist_id}/spotify-stats
```

### Parameters
- `artist_id` (integer, required): The ChartMetric ID of the artist

### Response Format
Returns an array of data points with timestamps and monthly listener values.

### Example Usage
```python
from services.chartmetric_services import chartmetric_service as chartmetric

# Get artist monthly listeners
listeners_data = chartmetric.get_artist_spotify_stats(artist_id=12345)
```

### Example Response
```json
[
  {
    "timestp": "2023-01-01",
    "value": 1500000
  },
  {
    "timestp": "2023-02-01",
    "value": 1550000
  }
]
```

## Search Artist

Searches for artists by name.

### Endpoint
```
GET /search/artist
```

### Parameters
- `query` (string, required): The search term
- `limit` (integer, optional): Number of results to return (default: 10)

### Example Usage
```python
from services.chartmetric_services import chartmetric_service as chartmetric

# Search for an artist
results = chartmetric.search_artist(query="Artist Name", limit=5)
```

### Example Response
```json
[
  {
    "id": 12345,
    "name": "Artist Name",
    "image_url": "https://example.com/image.jpg",
    "followers": 1500000
  }
]
```

## Get Artist Details

Retrieves detailed information about an artist.

### Endpoint
```
GET /artist/{artist_id}
```

### Parameters
- `artist_id` (integer, required): The ChartMetric ID of the artist

### Example Usage
```python
from services.chartmetric_services import chartmetric_service as chartmetric

# Get artist details
artist_details = chartmetric.get_artist_details(artist_id=12345)
```

## Get Artist Tracks

Get the tracks by an artist. This includes tracks where the artist is the main artist as well as tracks where they are featured.

### Endpoint
```
GET /artist/:id/tracks
```

### Parameters

| Field | Type | Description |
|-------|------|-------------|
| id | Integer | Chartmetric artist ID |
| offset | Integer | (Optional) The offset of entries to be returned. Default: 0 |
| limit | Integer | (Optional) The number of entries to be returned. Default: 10 |
| artist_type | String | (Optional) Type of artist: 'main' or 'featured'. Omitting this parameter will return both types of artists. |

### Example Usage
```
curl -H 'Authorization: Bearer [ACCESS KEY]' https://api.chartmetric.com/api/artist/206557/tracks
```

### Example Usage in Python
```python
from services.chartmetric_services import chartmetric_service as chartmetric

# Get tracks for an artist
tracks = chartmetric.get_artist_tracks(
    artist_id=206557, 
    limit=20, 
    offset=0,
    artist_type="main"
)
```

### Response Format

| Field | Type | Description |
|-------|------|-------------|
| total | Integer | Total number of tracks for the artist in Chartmetric's database |
| artist_type | String | Type of artist, main or featured |
| album_ids | Integer[] | Album IDs |
| album_label | String[] | Album labels |
| album_names | String[] | Album names |
| album_upc | String[] | Album UPC |
| amazon_album_ids | String[] | Amazon album ID's |
| amazon_track_ids | String[] | Amazon track ID's |
| artist_covers | String[] | Artist cover image URLs |
| artist_images | String[] | Artist image URLs |
| artist_names | String[] | Artist names |
| cm_artist | Integer | Chartmetric Artist ID's |
| cm_track | Integer | Chartmetric track ID |
| code2s | String[] | Country codes |
| created_at | String | Date this entry was created |
| deezer_album_ids | Integer[] | Deezer album ID's |
| deezer_duration | Integer | Deezer track duration |
| deezer_track_ids | String[] | Deezer track ID's |
| description | String | Track description (deprecated, returns null) |
| id | Integer | Chartmetric's Track ID (deprecated, use cm_track instead) |
| image_url | String | Track image URL |
| isrc | String | Track ISRC |
| itunes_album_ids | String[] | iTunes album ID's |
| itunes_track_ids | String[] | iTunes track ID's |
| modified_at | Date | Date this entry was modified (deprecated, returns null) |
| name | String | Track's name |
| release_dates | Date[] | Track release dates |
| spotify_album_ids | String[] | Spotify album ID's |
| spotify_duration_ms | Integer | Spotify track duration |
| spotify_track_ids | String[] | Spotify track ID's |
| storefronts | String[] | Storefronts list |
| tags | String | Comma separated list of genres for the track |
| cm_statistics | Object | Statistics about the track |
| » cm_track | Integer | Chartmetric's Track ID (deprecated, use root cm_track instead) |
| » de_playlist_total_reach | Integer | Deezer playlist total reach |
| » num_am_editorial_playlists | Integer | Apple music editorial playlists count |
| » num_am_playlists | Integer | Apple music playlists count |
| » num_az_editorial_playlists | Integer | Amazon editorial playlists count |
| » num_az_playlists | Integer | Amazon playlists count |
| » num_de_editorial_playlists | Integer | Deezer editorial playlists count |
| » num_de_playlists | Integer | Deezer playlists count |
| » num_sp_editorial_playlists | Integer | Spotify editorial playlists count |
| » num_sp_playlists | Integer | Spotify playlists count |
| » num_tt_videos | Integer | TikTok videos count |
| » num_yt_editorial_playlists | Integer | YouTube editorial playlists count |
| » num_yt_playlists | Integer | YouTube playlists count |
| » sp_playlist_total_reach | Integer | Spotify playlist total reach |
| » sp_popularity | Integer | Spotify popularity |
| » sp_streams | Integer | Spotify streams (not available for every track) |
| » yt_playlist_total_reach | String | YouTube playlist total reach |
| cm_audio_features | Object | Chartmetric's Audio features (deprecated, fields return 0) |
| » acousticness | Integer | Acousticness |
| » danceability | Integer | Danceability |
| » energy | Integer | Energy |
| » instrumentalness | Integer | Instrumentalness |
| » key | Integer | Key |
| » liveness | Integer | Liveness |
| » loudness | Integer | Loudness |
| » mode | Integer | Mode |
| » speechiness | Integer | Speechiness |
| » tempo | Integer | Tempo |
| » valence | Integer | Valence |

## Get Artist Spotify Monthly Listeners by City

Spotify's "Where people listen" stats, showing cities on each day. This endpoint provides geographic data about where an artist's audience is located.

**Note:** 
- Spotify data for top 50 cities available prior to Aug 12, 2024
- As of 03/17/2025, this endpoint is limited to 10 requests per-second per-user to maintain system stability.

### Endpoint
```
GET /artist/:id/where-people-listen
```

### Parameters

| Field | Type | Description |
|-------|------|-------------|
| id | Integer | Chartmetric artist ID |
| since | Date | Beginning date in ISO date format (e.g. "2017-03-25"). This parameter is required. |
| until | Date | (Optional) End date in ISO date format (e.g. "2017-03-25"). Default: today |
| date | Date | (Optional) Date in ISO date format (e.g. "2017-03-25"). This parameter can be passed in instead of since and until, to fetch a snapshot for a specific date. This will include top 50 cities prior to Aug 12, 2024. |
| limit | Integer | (Optional) The number of entries to be returned (default 10, max 50). Default: 10 |
| offset | Integer | (Optional) The offset of entries to be returned. Default: 0 |
| latest | Boolean | (Optional) If set to true, returns latest data point available regardless of date. If this is true, since/until parameters will be ignored. |

### Example Usage
```
curl -H 'Authorization: Bearer [ACCESS KEY]' https://api.chartmetric.com/api/artist/2762/where-people-listen?latest=true
```

### Example Usage in Python
```python
from services.chartmetric_services import chartmetric_service as chartmetric

# Get latest geographic data for an artist
geographic_data = chartmetric.get_artist_track_where_people_listen(
    artist_id=2762,
    latest=True
)

# Get geographic data for a specific time period
geographic_data = chartmetric.get_artist_track_where_people_listen(
    artist_id=2762,
    since="2023-01-01",
    until="2023-03-31",
    limit=50
)
```

### Response Format

#### Cities Data

| Field | Type | Description |
|-------|------|-------------|
| cities | Object | List of cities |
| » timestp | Date | Time stamp |
| » code2 | String | Country code |
| » spotify_artist_id | String | Spotify artist ID |
| » lat | Number | Latitude of the city |
| » lng | Number | Longitude of the city |
| » region | String | Region |
| » city_id | Integer | Unique city ID |
| » spotify_artist_insights_location | String | Spotify artist insights location (null if not available) |
| » current_max_count | Integer | Current maximum count |
| » market_max_cm_artist_id | Integer | Chartmetric artist ID for the artist with the maximum listeners in the market |
| » market_max_artist_name | String | Name of the artist with the maximum listeners in the market |
| » listeners | Integer | Current listeners count |
| » prev_listeners | Integer | Previous listeners count |
| » artist_city_rank | Integer | Artist rank in the city (null if not available) |
| » max_listeners | Integer | Maximum listeners count |
| » population | Integer | City population |
| » city_affinity | String | Affinity to the city (null if not available) |
| » is_estimate | Boolean | Whether the data is an estimate |

#### Countries Data

| Field | Type | Description |
|-------|------|-------------|
| countries | Object | List of countries |
| » timestp | Date | Time stamp |
| » code2 | String | Country code |
| » spotify_artist_id | String | Spotify artist ID |
| » lat | Number | Latitude of the country |
| » lng | Number | Longitude of the country |
| » region | String | Region |
| » market_max_cm_artist_id | Integer | Chartmetric artist ID for the artist with the maximum listeners in the country (null if not applicable) |
| » market_max_artist_name | String | Name of the artist with the maximum listeners in the country (null if not applicable) |
| » listeners | Integer | Current listeners count |
| » prev_listeners | Integer | Previous listeners count |
| » artist_city_rank | Integer | Artist rank in the country (null if not available) |
| » max_listeners | Integer | Maximum listeners count |
| » population | Integer | Country population |
| » is_estimate | Boolean | Whether the data is an estimate | 