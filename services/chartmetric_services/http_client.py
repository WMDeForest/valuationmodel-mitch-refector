import requests

from abc import ABC, abstractmethod

from services.comman.types import HttpMethod


class IHTTPClient(ABC):

    @abstractmethod
    def request(
        self,
        url: str,
        method: HttpMethod = "GET",
        params: dict | None = None,
        data: dict | list | None = None,
        headers: dict | None = None,
        cookies: dict | None = None,
        json: dict | None = None,
        timeout: int | None = None,
    ) -> str: ...


class RequestsHTTPClient(IHTTPClient):

    def request(
        self,
        url: str,
        method: HttpMethod = "GET",
        params: dict | None = None,
        data: dict | list | None = None,
        headers: dict | None = None,
        cookies: dict | None = None,
        json: dict | None = None,
        timeout: int | None = None,
    ) -> str:
        response = requests.request(
            method=method,
            url=url,
            params=params,
            data=data,
            headers=headers,
            cookies=cookies,
            json=json,
            timeout=timeout,
        )
        response.raise_for_status()
        return response.text
