import requests


class RapidAPI:
    def __init__(self):
        pass

    def getData(self, symbol):
        url = "https://alpha-vantage.p.rapidapi.com/query"

        querystring = {"interval": "5min", "function": "TIME_SERIES_INTRADAY",
                       "symbol": symbol, "datatype": "json", "output_size": "compact"}

        headers = {
            "X-RapidAPI-Key": "1c1a28d3a9mshc137ad75fd3c883p1c559cjsnd8c0303ebb90",
            "X-RapidAPI-Host": "alpha-vantage.p.rapidapi.com"
        }

        response = requests.request(
            "GET", url, headers=headers, params=querystring)

        return response.text
