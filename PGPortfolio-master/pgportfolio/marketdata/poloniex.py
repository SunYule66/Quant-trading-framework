import os
import sys
import time
import requests

if sys.version_info[0] == 3:
    from urllib.parse import urlencode
else:
    from urllib import urlencode

BINANCE_DEFAULT_BASE_URLS = [
    "https://api.binance.com",
    "https://api1.binance.com",
    "https://api2.binance.com",
    "https://api3.binance.com",
]
BINANCE_INTERVAL_MAP = {
    60 * 5: "5m",
    60 * 15: "15m",
    60 * 30: "30m",
    60 * 60: "1h",
    60 * 60 * 2: "2h",
    60 * 60 * 4: "4h",
    60 * 60 * 24: "1d",
}
BINANCE_LIMIT = 1000


class Poloniex:
    """
    Legacy Poloniex adapter that now proxies to the Binance public REST API.
    We keep the original class name to avoid touching the rest of the codebase.
    """

    def __init__(self, APIKey="", Secret="", base_urls=None, timeout=15):
        self.APIKey = APIKey.encode()
        self.Secret = Secret.encode()
        self._session = requests.Session()
        self._headers = {
            "User-Agent": "pgportfolio-binance-adapter/1.0",
            "Accept": "application/json",
        }
        self._session.headers.update(self._headers)
        self._ticker_cache = None
        self._ticker_cache_time = 0
        self._ticker_cache_ttl = 30  # seconds
        env_base = os.getenv("PGP_BINANCE_BASE_URL") or os.getenv("BINANCE_BASE_URL")
        if base_urls is None:
            if env_base:
                base_urls = [env_base]
            else:
                base_urls = BINANCE_DEFAULT_BASE_URLS
        elif isinstance(base_urls, str):
            base_urls = [base_urls]
        self._base_urls = [url.rstrip("/") for url in base_urls if url]
        if not self._base_urls:
            raise ValueError("At least one Binance base URL must be provided.")
        self._timeout = timeout
        self._base_url_idx = 0

    def marketTicker(self):
        ticker = {}
        for meta in self._fetch_filtered_tickers():
            ticker[meta["pair"]] = {
                "last": meta["lastPrice"],
                "quoteVolume": meta["quoteVolume"],
                "baseVolume": meta["baseVolume"],
            }
        return ticker

    def marketVolume(self):
        volume = {}
        for meta in self._fetch_filtered_tickers():
            base, quote = meta["pair"].split("_")
            if meta["pair"].startswith("BTC_"):
                al_coin = quote
            else:
                al_coin = base
            volume[meta["pair"]] = {
                "BTC": meta["quoteVolume"],
                al_coin: meta["baseVolume"],
            }
        return volume

    def marketStatus(self):
        status = {}
        for meta in self._fetch_filtered_tickers():
            base, quote = meta["pair"].split("_")
            if meta["pair"].startswith("BTC_"):
                status[quote] = {"status": "TRADING"}
            else:
                status[base] = {"status": "TRADING"}
        return status

    def marketLoans(self, coin):
        raise NotImplementedError("Loans are not supported on the Binance adapter.")

    def marketOrders(self, pair="all", depth=10):
        raise NotImplementedError("Order book is not supported on the Binance adapter.")

    def marketChart(self, pair, period, start, end):
        symbol = self._pair_to_symbol(pair)
        interval = BINANCE_INTERVAL_MAP.get(int(period))
        if interval is None:
            raise ValueError("Unsupported period {} seconds".format(period))

        start_ms = int(start) * 1000
        end_ms = int(end) * 1000
        fetch_start = start_ms
        chart = []

        while fetch_start < end_ms:
            params = {
                "symbol": symbol,
                "interval": interval,
                "startTime": fetch_start,
                "endTime": end_ms,
                "limit": BINANCE_LIMIT,
            }
            candles = self._public_get("/api/v3/klines", params)
            if not candles:
                break
            for candle in candles:
                open_time = int(candle[0]) // 1000
                if open_time < start:
                    continue
                if open_time > end:
                    break
                volume = float(candle[5])
                quote_volume = float(candle[7])
                weighted_average = quote_volume / volume if volume > 0 else 0.0
                chart.append(
                    {
                        "date": open_time,
                        "open": float(candle[1]),
                        "high": float(candle[2]),
                        "low": float(candle[3]),
                        "close": float(candle[4]),
                        "volume": volume,
                        "quoteVolume": quote_volume,
                        "weightedAverage": weighted_average,
                    }
                )
            last_close_time = int(candles[-1][6])
            fetch_start = last_close_time + 1
            if len(candles) < BINANCE_LIMIT:
                break
        return chart

    def marketTradeHist(self, pair):
        raise NotImplementedError("Trade history is not supported on the Binance adapter.")

    # ----------------------
    # Internal helper utils
    # ----------------------
    def _public_get(self, path, params=None):
        if params is None:
            params = {}
        query = ""
        if params:
            query = "?" + urlencode(params)
        last_error = None
        for attempt in range(len(self._base_urls)):
            idx = (self._base_url_idx + attempt) % len(self._base_urls)
            base = self._base_urls[idx]
            url = base + path + query
            try:
                response = self._session.get(url, timeout=self._timeout)
                response.raise_for_status()
                self._base_url_idx = idx
                return response.json()
            except requests.RequestException as exc:
                last_error = exc
                continue
        raise last_error  # type: ignore

    def _fetch_filtered_tickers(self):
        now = time.time()
        if (
            self._ticker_cache is not None
            and now - self._ticker_cache_time < self._ticker_cache_ttl
        ):
            return self._ticker_cache

        raw_tickers = self._public_get("/api/v3/ticker/24hr")
        filtered = []
        for item in raw_tickers:
            symbol = item.get("symbol", "")
            pair = self._symbol_to_pair(symbol)
            if pair is None:
                continue
            filtered.append(
                {
                    "symbol": symbol,
                    "pair": pair,
                    "lastPrice": float(item.get("lastPrice", 0.0)),
                    "quoteVolume": float(item.get("quoteVolume", 0.0)),
                    "baseVolume": float(item.get("volume", 0.0)),
                }
            )
        self._ticker_cache = filtered
        self._ticker_cache_time = now
        return filtered

    @staticmethod
    def _symbol_to_pair(symbol):
        if symbol.endswith("BTC"):
            coin = symbol[:-3]
            if not coin:
                return None
            return "BTC_{}".format(coin)
        if symbol.startswith("BTC"):
            coin = symbol[3:]
            if not coin:
                return None
            return "{}_BTC".format(coin)
        return None

    @staticmethod
    def _pair_to_symbol(pair):
        if pair.startswith("BTC_"):
            coin = pair.split("_", 1)[1]
            return "{}BTC".format(coin)
        if pair.endswith("_BTC"):
            coin = pair.split("_", 1)[0]
            return "BTC{}".format(coin)
        raise ValueError("Unsupported pair format: {}".format(pair))
