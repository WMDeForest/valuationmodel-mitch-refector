import json
import streamlit as st

from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from requests import HTTPError

from services.chartmetric_services.dto import (
    ArtistStateCampareListner,
    CountryListeners,
    Track,
    TrackSpotifyState,
)
from services.chartmetric_services.urlpatterns import (
    CHARTMETRIC_API_TOKEN_URL,
    CHARTMETRIC_API_TRACK_METADATA_URL,
    CHARTMETRIC_API_ARTIST_SP_STREAMS__URL,
    CHARTMETRIC_API_TRACK_SP_STREAMS_DAY_BY_DAY_URL,
    CHARTMETRIC_API_ARTIST_WHERE_PEOPLE_LISTEN_URL,
)
from services.chartmetric_services.http_client import IHTTPClient
from services.chartmetric_services.exceptions import ChartmetricTokenError
from services.chartmetric_services.wrappers import handle_chartmetric_errors


REFRESH_TOKEN = st.secrets["CM_TOKEN"]


class IChartmetricSDK(ABC):

    def __init__(self, refresh_token: str, client: IHTTPClient) -> None:
        self._refresh_token = refresh_token
        self._client = client
        self._headers: dict | None = {}

    @abstractmethod
    def get_artist_spotify_stats(
        self,
        artist_id: int | None = None,
        since: str | None = None,
        untill: str | None = None,
    ) -> list[ArtistStateCampareListner]: ...

    @abstractmethod
    def get_track_sp_streams_campare(
        self, track_id: int
    ) -> list[TrackSpotifyState]: ...

    @abstractmethod
    def get_track_detail(self, track_id: int) -> Track: ...

    @abstractmethod
    def get_artist_track_where_people_listen(
        self, artist_id: int = 0
    ) -> list[CountryListeners]: ...


class ChartMetricService(IChartmetricSDK):

    def get_token(
        self,
    ) -> None:
        try:
            response_text = self._client.request(
                url=CHARTMETRIC_API_TOKEN_URL,
                method="POST",
                data={"refreshtoken": REFRESH_TOKEN},
            )
            response_json = json.loads(response_text)
        except HTTPError as err:
            raise ChartmetricTokenError(
                code=err.response.status_code, message=err.response.json()["error"]
            )
        access_token = response_json["token"]
        self._headers.update({
            "Authorization": f"Bearer {access_token}",
            "Content-Type":"application/json"
        })

    @handle_chartmetric_errors
    def __get_artist_sp_stream_state_campare_request(
        self,
        artist_id: int,
        untill: str = None,
        since: str = None,
    ) -> dict:
        url = CHARTMETRIC_API_ARTIST_SP_STREAMS__URL(artist_id=artist_id)
        params = {
            "until": untill,
            "since": since,
        }
        response_text = self._client.request(
            url=url,
            method="GET",
            headers=self._headers,
            params=params
        )
        return json.loads(response_text)

    def __create_artist_spotify_state_campare_output(
        self, objects: list[dict]
    ) -> list[ArtistStateCampareListner]:
        return [ArtistStateCampareListner(**listner) for listner in objects["listeners"]]

    def get_artist_spotify_stats(
        self, artist_id: int = 0, since: str = None, untill: str = None
    ) -> list[ArtistStateCampareListner]:
        today = datetime.today().date()
        ten_years_ago = today - timedelta(days=3650)  # ~10 years

        response_json = self.__get_artist_sp_stream_state_campare_request(
            artist_id=artist_id, untill=today, since=ten_years_ago
        )
        response_obj = response_json["obj"]
        return self.__create_artist_spotify_state_campare_output(response_obj)


    @handle_chartmetric_errors
    def __get_track_sp_streams_campare_request(
        self, track_id, release_date: str
    ) -> dict:
        params = {"since": release_date, "type": "streams", "interpolated": True}
        url = CHARTMETRIC_API_TRACK_SP_STREAMS_DAY_BY_DAY_URL(track_id=track_id)
        response = self._client.request(
            url=url, method="GET", headers=self._headers, params=params
        )
        return json.loads(response)

    def __create_track_sp_streams_campare_output(
        self, objects
    ) -> list[TrackSpotifyState]:
        return [TrackSpotifyState(**obj) for obj in objects]

    def get_track_sp_streams_campare(self, track_id) -> list[TrackSpotifyState]:
        track = self.get_track_detail(track_id=track_id)
        response_json = self.__get_track_sp_streams_campare_request(
            track_id, track.release_date
        )
        response_obj = response_json["obj"][0]["data"]
        response = self.__create_track_sp_streams_campare_output(response_obj)
        return response

    @handle_chartmetric_errors
    def __get_track_details_request(self, track_id) -> dict:
        url = CHARTMETRIC_API_TRACK_METADATA_URL(track_id=track_id)
        response = self._client.request(url=url, method="GET", headers=self._headers)
        return json.loads(response)

    def __create_track_details_output(self, object) -> Track:
        return Track(
            id=object["id"],
            name=object["name"],
            isrc=object.get("isrc", ""),
            image_url=object.get("image_url", ""),
            duration_ms=object.get("duration_ms", 0),
            composer_name=object.get("composer_name", ""),
            moods=object.get("moods", []),
            activities=object.get("activities", []),
            artists=object.get("artists", []),
            albums=object.get("albums", []),
            tags=object.get("tags", ""),
            songwriters=object.get("songwriters", []),
            release_date=object.get("release_date", ""),
            album_label=object.get("album_label", ""),
            explicit=object.get("explicit", False),
            score=object.get("score", 0.0),
            cm_statistics=object.get("cm_statistics", {}),
            tempo=object.get("tempo", None),
        )

    def get_track_detail(self, track_id: int) -> Track:
        response = self.__get_track_details_request(track_id=track_id)
        response_obj = response["obj"]
        response = self.__create_track_details_output(response_obj)
        return response

    @handle_chartmetric_errors
    def __get_artist_track_where_people_listen_request(self, artist_id, params=None) -> dict:
        # Set default parameters if none provided
        if params is None:
            params = {}
        
        # Ensure limit is set to 50 if not specified
        if 'limit' not in params:
            params['limit'] = 50
            
        # If no date-related parameters are provided, default to a recent date before Aug 12, 2024
        if 'date' not in params and 'since' not in params and 'until' not in params and 'latest' not in params:
            # Use a date slightly before Aug 12, 2024 to get top 50 cities
            params['since'] = '2024-08-11'
        
        response = self._client.request(
            url=CHARTMETRIC_API_ARTIST_WHERE_PEOPLE_LISTEN_URL(artist_id),
            method="GET",
            headers=self._headers,
            params=params
        )
        response_json = json.loads(response)
        return response_json

    def __create_artist_track_where_people_listen(self, object: dict) -> dict:
        """
        Process the artist geography data from the API
        
        Parameters:
        -----------
        object : dict
            The API response object
            
        Returns:
        --------
        dict
            Complete geography data with both cities and countries
        """
        try:
            # Simply return the raw object which includes both cities and countries
            # This preserves the full data structure needed by our display functions
            return object
        except Exception as e:
            # Log the error but don't crash
            print(f"Error processing geography data: {str(e)}")
            return {"cities": {}, "countries": {}}

    def get_artist_track_where_people_listen(self, artist_id=0, params=None):
        """
        Get audience geography data for an artist
        
        Parameters:
        -----------
        artist_id : int
            ChartMetric artist ID
        params : dict, optional
            Additional parameters to pass to the API
            
        Returns:
        --------
        dict
            Complete geography data with both cities and countries
        """
        try:
            response = self.__get_artist_track_where_people_listen_request(artist_id, params)
            response_obj = response["obj"]
            return self.__create_artist_track_where_people_listen(response_obj)
        except Exception as e:
            # Return empty dict on error
            print(f"Error in get_artist_track_where_people_listen: {str(e)}")
            return {"cities": {}, "countries": {}}
