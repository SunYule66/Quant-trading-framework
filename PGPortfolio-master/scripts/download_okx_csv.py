import argparse
import csv
import datetime as dt
import logging
import os
import time

import requests

OKX_HISTORY_ENDPOINT = "https://www.okx.com/api/v5/market/history-candles"
DEFAULT_BAR = "1H"
DEFAULT_LIMIT = 100


def parse_datetime(value):
    """
    Support handy CLI formats (YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS). Values are
    interpreted as UTC timestamps because OKX candlestick payloads are UTC-based.
    """
    if value is None:
        return None
    normalized = value.strip()
    if not normalized:
        return None
    for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        try:
            return int(dt.datetime.strptime(normalized, fmt).replace(tzinfo=dt.timezone.utc).timestamp() * 1000)
        except ValueError:
            continue
    raise ValueError("Unsupported datetime format: {}".format(value))


def fetch_candles(session, inst_id, bar, start_ms, end_ms, limit):
    """
    Download historical candles for a single instrument. The OKX endpoint returns
    newest-first, so we gather everything into memory and sort before writing.
    """
    cursor = end_ms
    fetched = []
    while cursor > start_ms:
        params = {"instId": inst_id, "bar": bar, "limit": str(limit), "before": str(cursor)}
        response = session.get(OKX_HISTORY_ENDPOINT, params=params, timeout=15)
        response.raise_for_status()
        payload = response.json()
        data = payload.get("data", [])
        if not isinstance(data, list) or not data:
            break
        for candle in data:
            open_time = int(candle[0])
            if open_time < start_ms:
                continue
            if open_time > end_ms:
                continue
            volume = float(candle[5])
            quote_volume = float(candle[6])
            weighted_avg = quote_volume / volume if volume else 0.0
            fetched.append(
                (
                    open_time // 1000,
                    float(candle[1]),
                    float(candle[2]),
                    float(candle[3]),
                    float(candle[4]),
                    volume,
                    quote_volume,
                    weighted_avg,
                )
            )
        oldest = int(data[-1][0])
        if oldest <= start_ms or len(data) < limit:
            break
        cursor = oldest
        time.sleep(0.25)  # be gentle with the public API
    fetched.sort(key=lambda row: row[0])
    return fetched


def write_csv(rows, output_path, overwrite):
    if not rows:
        logging.warning("No data downloaded for %s", output_path)
        return 0
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    mode = "w" if overwrite else "a"
    header = [
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "quote_volume",
        "weighted_average",
    ]
    file_exists = os.path.exists(output_path)
    with open(output_path, mode, newline="") as handle:
        writer = csv.writer(handle)
        if overwrite or (not file_exists):
            writer.writerow(header)
        writer.writerows(rows)
    logging.info("Wrote %s rows to %s", len(rows), output_path)
    return len(rows)


def download_dataset(coins, output_dir, bar, start_ms, end_ms, overwrite, limit):
    session = requests.Session()
    session.trust_env = False
    session.headers.update({"User-Agent": "pgportfolio-okx-downloader/1.0"})

    total = 0
    for inst_id in coins:
        logging.info("Downloading %s (%s)...", inst_id, bar)
        rows = fetch_candles(session, inst_id, bar, start_ms, end_ms, limit)
        filename = "{}-{}-candles.csv".format(inst_id, bar.lower())
        output_path = os.path.join(output_dir, filename)
        total += write_csv(rows, output_path, overwrite=overwrite)
    logging.info("Finished. %s total candles stored under %s", total, output_dir)


def build_parser():
    parser = argparse.ArgumentParser(description="Download OKX candlestick CSV snapshots into ./Data")
    parser.add_argument(
        "--coins",
        required=True,
        help="Comma separated OKX instruments (e.g. BTC-USDT,ETH-USDT).",
    )
    parser.add_argument(
        "--bar",
        default=DEFAULT_BAR,
        help="OKX interval (e.g. 1H, 4H, 1D). Defaults to {}.".format(DEFAULT_BAR),
    )
    parser.add_argument(
        "--start",
        required=True,
        help="UTC start time (YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS).",
    )
    parser.add_argument(
        "--end",
        default=None,
        help="UTC end time. Defaults to now.",
    )
    parser.add_argument(
        "--output-dir",
        default="Data",
        help="Directory to store CSV files (default: Data).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing files instead of appending.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_LIMIT,
        help="OKX page size (default {}).".format(DEFAULT_LIMIT),
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Logging level (DEBUG, INFO, ...).",
    )
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))

    coins = [token.strip().upper() for token in args.coins.split(",") if token.strip()]
    if not coins:
        raise ValueError("Please provide at least one instrument via --coins")
    start_ms = parse_datetime(args.start)
    end_ms = parse_datetime(args.end) if args.end else int(time.time() * 1000)
    if start_ms is None or end_ms is None:
        raise ValueError("Start/end time could not be parsed.")
    if start_ms >= end_ms:
        raise ValueError("--start must be earlier than --end")

    download_dataset(
        coins=coins,
        output_dir=args.output_dir,
        bar=args.bar,
        start_ms=start_ms,
        end_ms=end_ms,
        overwrite=args.overwrite,
        limit=max(1, args.limit),
    )


if __name__ == "__main__":
    main()

