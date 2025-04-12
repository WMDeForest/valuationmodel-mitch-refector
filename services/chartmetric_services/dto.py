from dataclasses import dataclass
from datetime import date, datetime
from typing import List, Dict, Optional


@dataclass
class RefreshToken:
    token: str
    expires_in: int
    refresh_token: str
    scope: str


@dataclass
class ArtistStateCampareListner:
    weekly_diff: int
    weekly_diff_percent: int
    monthly_diff: int
    monthly_diff_percent: float
    value: int
    timestp: datetime
    diff: str | None

    def __init__(
        self,
        weekly_diff: int,
        weekly_diff_percent: float,
        monthly_diff: int,
        monthly_diff_percent: float,
        value: int,
        timestp: str,
        diff: Optional[int],
    ):
        self.weekly_diff = weekly_diff
        self.weekly_diff_percent = weekly_diff_percent
        self.monthly_diff = monthly_diff
        self.monthly_diff_percent = monthly_diff_percent
        self.value = value
        self.diff = diff
        self.timestp = self.format_date(timestp)

    @staticmethod
    def format_date(iso_date: str) -> str:
        return datetime.fromisoformat(iso_date.replace("Z", "")).strftime("%Y-%m-%d")


@dataclass
class CountryListeners:
    country_name: str
    timestp: str
    code2: str
    spotify_artist_id: str
    lat: float
    lng: float
    region: str
    city_id: int
    spotify_artist_insights_location: int
    listeners: int
    prev_listeners: int
    population: int
    city_affinity: int
    current_max_count: int
    market_max_cm_artist_id: int
    market_max_artist_name: str
    max_listeners: int
    artist_city_rank: int
    is_estimate: bool


# @dataclass
# class CountryListeners:
#     countries: Dict[str, List[CountryData]]


@dataclass
class Mood:
    id: Optional[int]
    name: str


@dataclass
class Artist:
    id: int
    name: str
    image_url: Optional[str]
    cover_url: Optional[str]
    code2: Optional[str]
    gender: Optional[str]
    isni: Optional[str]
    description: Optional[str]
    created_at: str
    hometown_city: Optional[int]
    current_city: Optional[int]
    date_of_birth: Optional[str]
    date_of_death: Optional[str]
    modified_at: str
    is_duplicate: bool
    label: Optional[str]
    booking_agent: Optional[str]
    record_label: Optional[str]
    press_contact: Optional[str]
    general_manager: Optional[str]
    band: bool
    is_non_artist: bool


@dataclass
class Album:
    id: int
    name: str
    upc: str
    release_date: str
    label: str
    image_url: str
    popularity: int


@dataclass
class Statistics:
    timestamp: str
    score: int
    cm_track: int
    num_sp_playlists: Optional[int]
    num_sp_editorial_playlists: Optional[int]
    num_am_playlists: Optional[int]
    num_am_editorial_playlists: Optional[int]
    sp_playlist_total_reach: Optional[int]
    sp_popularity: Optional[int]
    sp_streams: Optional[int]
    airplay_streams: Optional[int]


@dataclass
class Track:
    id: int
    name: str
    isrc: str
    image_url: str
    duration_ms: int
    composer_name: str
    moods: List[Mood]
    activities: List[str]
    artists: List[Artist]
    albums: List[Album]
    tags: str
    songwriters: List[str]
    release_date: str
    album_label: str
    explicit: bool
    score: float
    cm_statistics: Statistics
    tempo: Optional[float]


@dataclass
class TrackSpotifyState:
    value: int
    timestp: datetime
    daily_diff: int
    interpolated: bool

    def __post_init__(self):
        self.timestp = datetime.fromisoformat(self.timestp.replace("Z", "")).strftime(
            "%Y-%m-%d"
        )
