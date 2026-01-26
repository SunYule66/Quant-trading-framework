from __future__ import absolute_import, division, print_function

import csv
import logging
import os
import re
import sqlite3

from pgportfolio.constants import get_database_path


def normalize_coin_name(raw_name):
    """
    Convert various OKX export naming styles into the canonical coin symbol used
    throughout PGPortfolio (uppercase ticker, optional ``reversed_`` prefix).
    """
    if raw_name is None:
        raise ValueError("coin name cannot be None")
    name = raw_name.strip()
    if not name:
        raise ValueError("coin name cannot be empty")
    if name.startswith("reversed_"):
        return name
    # Strip OKX candlestick suffixes, e.g. BTC-USDT-candlesticks-2024-11
    if "-candlesticks" in name:
        name = name.split("-candlesticks", 1)[0]
    # If the name looks like BASE-QUOTE (OKX historical dumps), keep BASE.
    if "-" in name:
        name = name.split("-", 1)[0]
    # Collapse any leftover non-alphanumeric characters.
    name = re.sub(r"[^A-Za-z0-9_]", "", name)
    if not name:
        raise ValueError("unable to normalize coin name '{}'".format(raw_name))
    return name.upper()


def normalize_timestamp(value):
    """
    Ensure timestamps are stored as Unix seconds (legacy training format). OKX
    CSV exports often provide milliseconds.
    """
    if value is None:
        raise ValueError("timestamp cannot be None")
    ts = int(float(value))
    # Anything above ~year 5138 in seconds is clearly millisecond precision.
    if ts > 10 ** 11:
        ts //= 1000
    return ts


class CSVDatabaseBuilder(object):
    """
    Utility that ingests the CSV snapshots produced by OkxRealtimeCollector
    (or any compatible exporter) into the legacy sqlite format expected by the
    original training pipeline (History table inside Data.db).
    """

    def __init__(self, source_dir, database_file=None, chunk_size=1000):
        if not source_dir:
            raise ValueError("source_dir must be provided")
        self._source_dir = source_dir
        self._db_path = get_database_path(database_file)
        self._chunk_size = max(1, int(chunk_size))

    def build(self):
        csv_files = self._discover_csv_files()
        if not csv_files:
            raise ValueError(
                "No CSV files found under {}. Please point --source_dir to the folder "
                "containing realtime snapshots.".format(self._source_dir)
            )

        db_dir = os.path.dirname(self._db_path)
        if db_dir and not os.path.exists(db_dir):
            os.makedirs(db_dir)

        connection = sqlite3.connect(self._db_path)
        try:
            cursor = connection.cursor()
            cursor.execute(
                "CREATE TABLE IF NOT EXISTS History ("
                "date INTEGER,"
                "coin VARCHAR(20),"
                "high FLOAT,"
                "low FLOAT,"
                "open FLOAT,"
                "close FLOAT,"
                "volume FLOAT,"
                "quoteVolume FLOAT,"
                "weightedAverage FLOAT,"
                "PRIMARY KEY (date, coin)"
                ");"
            )
            total_rows = 0
            for file_path, coin in csv_files:
                inserted = self._ingest_file(cursor, file_path, coin)
                total_rows += inserted
                logging.info(
                    "Imported %s rows for %s from %s", inserted, coin, file_path
                )
            connection.commit()
        finally:
            connection.close()

        logging.info(
            "Database build complete. %s rows written to %s",
            total_rows,
            self._db_path,
        )

    def _discover_csv_files(self):
        files = []
        if not os.path.isdir(self._source_dir):
            raise ValueError(
                "source_dir {} is not a directory or does not exist".format(
                    self._source_dir
                )
            )
        for entry in os.listdir(self._source_dir):
            if not entry.lower().endswith(".csv"):
                continue
            coin = os.path.splitext(entry)[0]
            files.append((os.path.join(self._source_dir, entry), coin))
        files.sort()
        return files

    def _ingest_file(self, cursor, file_path, coin):
        inserted = 0
        batch = []
        normalized_coin = normalize_coin_name(coin)
        with open(file_path, "r") as handle:
            reader = csv.DictReader(handle)
            for line_index, row in enumerate(reader, start=2):
                try:
                    batch.append(self._row_to_tuple(row, normalized_coin))
                except (ValueError, KeyError) as exc:
                    logging.warning(
                        "Skipping %s line %s (%s): %s", coin, line_index, file_path, exc
                    )
                    continue
                if len(batch) >= self._chunk_size:
                    cursor.executemany(self._insert_sql(), batch)
                    inserted += len(batch)
                    batch = []
        if batch:
            cursor.executemany(self._insert_sql(), batch)
            inserted += len(batch)
        logging.info(
            "Imported %s rows for %s (source tag %s) from %s",
            inserted,
            normalized_coin,
            coin,
            file_path,
        )
        return inserted

    @staticmethod
    def _insert_sql():
        return (
            "INSERT OR REPLACE INTO History "
            "(date, coin, high, low, open, close, volume, quoteVolume, weightedAverage) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)"
        )

    def _row_to_tuple(self, row, coin):
        timestamp = self._extract_number(row, "timestamp", "open_time")
        high = self._extract_number(row, "high")
        low = self._extract_number(row, "low")
        open_price = self._extract_number(row, "open")
        close = self._extract_number(row, "close")
        volume = self._extract_number(row, "volume", "vol", "vol_base", "vol_ccy")
        quote_volume = self._extract_number(
            row, "quote_volume", "quoteVolume", "vol_quote"
        )
        weighted_avg = self._extract_number(
            row, "weighted_average", "weightedAverage", default=close
        )
        return (
            normalize_timestamp(timestamp),
            coin,
            float(high),
            float(low),
            float(open_price),
            float(close),
            float(volume),
            float(quote_volume),
            float(weighted_avg),
        )

    def _extract_number(self, row, *keys, **kwargs):
        default = kwargs.get("default", None)
        for key in keys:
            if key in row and row[key] not in (None, "", "null"):
                try:
                    return float(row[key])
                except ValueError:
                    pass
        if default is not None:
            return default
        raise KeyError("Missing numeric column(s) {} in row {}".format(keys, row))


__all__ = ["CSVDatabaseBuilder", "normalize_coin_name", "normalize_timestamp"]


