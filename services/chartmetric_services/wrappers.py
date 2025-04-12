import logging
import time

from functools import wraps
from requests import HTTPError

from services.chartmetric_services.exceptions import ChartmetricApiError


def handle_chartmetric_errors(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except HTTPError as err:
            status_code = err.response.status_code
            logging.info(f"{err.response.text=}")
            match status_code:
                case 429:
                    headers = dict(err.response.headers)
                    reset_time = int(headers.get("X-RateLimit-Reset"))
                    wait_time = max(0, reset_time - int(time.time()) + 1)
                    logging.warning(
                        f"Rate limit hit. Waiting for {wait_time} seconds..."
                    )
                    time.sleep(wait_time)
                    return wrapper(*args, **kwargs)
                case 401:
                    logging.info(
                        "Unauthorized error occurred. Refreshing authorization headers."
                    )
                    args[0].get_token()
                    return wrapper(*args, **kwargs)
                case 504 | 502:
                    logging.warning(f"{status_code=}; {err.response.text=}")
                    return wrapper(*args, **kwargs)
                case _:
                    raise ChartmetricApiError(
                        code=status_code, message=err.response.text
                    )

    return wrapper
