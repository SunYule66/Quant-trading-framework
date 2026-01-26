from __future__ import absolute_import, division, print_function

import csv
import logging
import os
import time

from pgportfolio.marketdata.okx import Okx
from pgportfolio.tools.trade import get_coin_name_list


class OkxRealtimeCollector(object):
    """
    Poller that keeps downloading the latest OHLCV bars from OKX so downstream
    components (training, monitoring or live trading) can reuse the exact same
    feature tensor layout as the historical backtests.

    The collector writes one CSV file per coin into ``output_dir``. Each row
    stores the close timestamp plus OHLCV information of a single bar.
    """

    def __init__(
        self,
        config,
        coins=None,
        period=None,
        window_size=None,
        output_dir=None,
        max_snapshots=None,
        quote_currency="BTC",
    ):
        input_config = config["input"]
        self._period = int(period or input_config["global_period"])
        self._window_size = int(window_size or input_config["window_size"])
        self._quote_currency = quote_currency.upper()
        self._output_dir = output_dir or os.path.join("database", "realtime")
        self._max_snapshots = max_snapshots
        os.makedirs(self._output_dir, exist_ok=True)

        self._okx = Okx()
        self._coins = self._resolve_coins(config, coins)
        self._last_snapshot_ts = None

    def run_forever(self):
        """
        Block until interrupted (Ctrl+C) or until ``max_snapshots`` has been
        reached. Each iteration waits for the end of the current period so the
        stored bar aligns with the backtest / training grid.
        """
        logging.info(
            "Starting OKX realtime collector for %s (period=%ss).",
            ", ".join(self._coins),
            self._period,
        )
        snapshots = 0
        try:
            while self._max_snapshots is None or snapshots < self._max_snapshots:
                wait = self._wait_until_period_close()
                if wait > 0:
                    time.sleep(wait)
                timestamp = self._aligned_timestamp()
                if timestamp == self._last_snapshot_ts:
                    continue
                self.collect_once(timestamp)
                self._last_snapshot_ts = timestamp
                snapshots += 1
        except KeyboardInterrupt:
            logging.info("Realtime collector interrupted by user.")

    def collect_once(self, timestamp=None):
        """
        Collect a single OHLCV snapshot for every configured coin at the given
        timestamp (default: aligned current time) and append it to the CSV
        files.
        """
        ts = int(timestamp or self._aligned_timestamp())
        start = ts - self._period * (self._window_size + 1)
        logging.debug("Collecting candles for window [%s, %s).", start, ts)
        for coin in self._coins:
            pair = self._coin_to_pair(coin)
            try:
                candles = self._okx.marketChart(pair, self._period, start, ts)
            except Exception as exc:
                logging.warning("Failed to download candles for %s: %s", pair, exc)
                continue
            if not candles:
                logging.warning("No candles returned for %s", pair)
                continue
            latest = candles[-1]
            if int(latest["date"]) != ts - self._period:
                logging.debug(
                    "Skipping %s; latest candle (%s) not aligned with %s",
                    pair,
                    latest["date"],
                    ts,
                )
                continue
            self._persist_candle(coin, latest)

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _resolve_coins(self, config, coins):
        if coins:
            if isinstance(coins, str):
                iterable = coins.split(",")
            else:
                iterable = coins
            resolved = [c.strip() for c in iterable if c.strip()]
            if not resolved:
                raise ValueError("Provided coin list is empty.")
            return resolved
        logging.info("No coin list provided; selecting coins via HistoryManager.")
        resolved = get_coin_name_list(config, online=True)
        if not resolved:
            raise ValueError("Unable to infer coin list from configuration.")
        return resolved

    def _wait_until_period_close(self):
        now = time.time()
        remaining = self._period - (now % self._period)
        return remaining

    def _aligned_timestamp(self):
        current = int(time.time())
        return current - (current % self._period)

    def _coin_to_pair(self, coin_name):
        base = coin_name.replace("reversed_", "").upper()
        if coin_name.startswith("reversed_"):
            return "{}_{}".format(base, self._quote_currency)
        return "{}_{}".format(self._quote_currency, base)

    def _persist_candle(self, coin, candle):
        file_path = os.path.join(self._output_dir, "{}.csv".format(coin))
        is_new = not os.path.exists(file_path)
        row = [
            int(candle["date"]),
            float(candle["open"]),
            float(candle["high"]),
            float(candle["low"]),
            float(candle["close"]),
            float(candle["volume"]),
            float(candle["quoteVolume"]),
            float(candle["weightedAverage"]),
        ]
        with open(file_path, "a", newline="") as csv_file:
            writer = csv.writer(csv_file)
            if is_new:
                writer.writerow(
                    [
                        "timestamp",
                        "open",
                        "high",
                        "low",
                        "close",
                        "volume",
                        "quote_volume",
                        "weighted_average",
                    ]
                )
            writer.writerow(row)
        logging.info("Saved candle for %s at %s", coin, candle["date"])

