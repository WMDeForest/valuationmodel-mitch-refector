class ChartmetricApiError(Exception):
    def __init__(self, code, message) -> None:
        self.code = code
        self.message = message
        super().__init__(message)

    def __str__(self):
        return f"Status code {self.code}: {self.message}"


class ChartmetricTokenError(ChartmetricApiError):
    pass
