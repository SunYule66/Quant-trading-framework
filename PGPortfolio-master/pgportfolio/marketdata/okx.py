import os
import sys
import time
import requests

if sys.version_info[0] == 3:
    from urllib.parse import urlencode
else:
    from urllib import urlencode

OKX_DEFAULT_BASE_URLS = ["https://www.okx.com"]
OKX_INTERVAL_MAP = {
    60 * 5: "5m",
    60 * 15: "15m",
    60 * 30: "30m",
    60 * 60: "1H",
    60 * 60 * 2: "2H",
    60 * 60 * 4: "4H",
    60 * 60 * 24: "1D",
}
OKX_LIMIT = 100


class Okx:
    """
    Lightweight OKX spot market adapter that mimics the legacy Poloniex interface
    used across PGPortfolio. Only the endpoints that the platform depends on are
    implemented (ticker, volume, market status and historical candles).
    """

    def __init__(self, api_key="", secret="", base_urls=None, timeout=15):
        self.APIKey = api_key.encode()
        self.Secret = secret.encode()
        self._session = requests.Session()
        # Disable inheriting system-wide proxy settings because many users run PGPortfolio
        # on machines that enforce proxies through OS-level configuration (e.g. Windows
        # Internet Options). Those proxies often block OKX, and requests enables them by
        # default via ``trust_env``. Allow overriding by setting PGP_USE_SYSTEM_PROXY=1.
        if os.getenv("PGP_USE_SYSTEM_PROXY", "").lower() not in ("1", "true", "yes"):
            self._session.trust_env = False
            self._session.proxies.clear()
        self._session.headers.update(
            {
                "User-Agent": "pgportfolio-okx-adapter/1.0",
                "Accept": "application/json",
            }
        )
        self._ticker_cache = None
        self._ticker_cache_time = 0
        self._ticker_cache_ttl = 15  # seconds
        env_base = os.getenv("PGP_OKX_BASE_URL") or os.getenv("OKX_BASE_URL")
        if base_urls is None:
            if env_base:
                base_urls = [env_base]
            else:
                base_urls = OKX_DEFAULT_BASE_URLS
        elif isinstance(base_urls, str):
            base_urls = [base_urls]
        self._base_urls = [url.rstrip("/") for url in base_urls if url]
        if not self._base_urls:
            raise ValueError("At least one OKX base URL must be provided.")
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
                alt_coin = quote
            else:
                alt_coin = base
            volume[meta["pair"]] = {
                "BTC": meta["quoteVolume"],
                alt_coin: meta["baseVolume"],
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
        raise NotImplementedError("Loans are not supported on the OKX adapter.")

    def marketOrders(self, pair="all", depth=10):
        raise NotImplementedError("Order book is not supported on the OKX adapter.")

    def marketChart(self, pair, period, start, end):
        inst_id = self._pair_to_inst_id(pair)
        interval = OKX_INTERVAL_MAP.get(int(period))
        if interval is None:
            raise ValueError("Unsupported period {} seconds".format(period))

        start_ms = int(start) * 1000
        end_ms = int(end) * 1000
        cursor = end_ms
        candles = []

        while cursor > start_ms:
            params = {
                "instId": inst_id,
                "bar": interval,
                "limit": OKX_LIMIT,
                "before": str(cursor),
            }
            batch = self._public_get("/api/v5/market/history-candles", params)
            if not batch:
                break
            for candle in batch:
                open_time_ms = int(candle[0])
                if open_time_ms < start_ms:
                    continue
                if open_time_ms > end_ms:
                    continue
                volume = float(candle[5])
                quote_volume = float(candle[6])
                weighted_average = quote_volume / volume if volume > 0 else 0.0
                candles.append(
                    {
                        "date": open_time_ms // 1000,
                        "open": float(candle[1]),
                        "high": float(candle[2]),
                        "low": float(candle[3]),
                        "close": float(candle[4]),
                        "volume": volume,
                        "quoteVolume": quote_volume,
                        "weightedAverage": weighted_average,
                    }
                )
            oldest_in_batch = int(batch[-1][0])
            if oldest_in_batch <= start_ms or len(batch) < OKX_LIMIT:
                break
            cursor = oldest_in_batch

        candles.sort(key=lambda item: item["date"])
        return candles

    def marketTradeHist(self, pair):
        raise NotImplementedError("Trade history is not supported on the OKX adapter.")

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
                break
            except requests.RequestException as exc:
                last_error = exc
                continue
        else:
            raise last_error  # type: ignore
        payload = response.json()
        if isinstance(payload, dict):
            code = payload.get("code")
            if code not in (None, 0, "0"):
                raise ValueError(
                    "OKX API error {}: {}".format(code, payload.get("msg", ""))
                )
            return payload.get("data", [])
        return payload

    def _fetch_filtered_tickers(self):
        now = time.time()
        if (
            self._ticker_cache is not None
            and now - self._ticker_cache_time < self._ticker_cache_ttl
        ):
            return self._ticker_cache

        raw_tickers = self._public_get(
            "/api/v5/market/tickers", {"instType": "SPOT"}
        )
        filtered = []
        for item in raw_tickers:
            inst_id = item.get("instId", "")
            pair = self._inst_id_to_pair(inst_id)
            if pair is None:
                continue
            if inst_id.endswith("-BTC"):
                btc_volume = float(item.get("volCcy24h", 0.0))
                alt_volume = float(item.get("vol24h", 0.0))
            elif inst_id.startswith("BTC-"):
                btc_volume = float(item.get("vol24h", 0.0))
                alt_volume = float(item.get("volCcy24h", 0.0))
            else:
                continue
            filtered.append(
                {
                    "symbol": inst_id,
                    "pair": pair,
                    "lastPrice": float(item.get("last", 0.0)),
                    "quoteVolume": btc_volume,
                    "baseVolume": alt_volume,
                }
            )
        self._ticker_cache = filtered
        self._ticker_cache_time = now
        return filtered

    @staticmethod
    def _inst_id_to_pair(inst_id):
        if inst_id.endswith("-BTC"):
            coin = inst_id[:-4]
            if not coin:
                return None
            return "BTC_{}".format(coin)
        if inst_id.startswith("BTC-"):
            coin = inst_id[4:]
            if not coin:
                return None
            return "{}_BTC".format(coin)
        return None

    @staticmethod
    def _pair_to_inst_id(pair):
        if pair.startswith("BTC_"):
            coin = pair.split("_", 1)[1]
            if not coin:
                raise ValueError("Unsupported pair format: {}".format(pair))
            return "{}-BTC".format(coin)
        if pair.endswith("_BTC"):
            coin = pair.split("_", 1)[0]
            if not coin:
                raise ValueError("Unsupported pair format: {}".format(pair))
            return "BTC-{}".format(coin)
        raise ValueError("Unsupported pair format: {}".format(pair))


