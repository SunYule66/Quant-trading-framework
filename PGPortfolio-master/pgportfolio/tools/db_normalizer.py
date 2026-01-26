from __future__ import absolute_import, division, print_function

import argparse
import logging
import sqlite3

from pgportfolio.constants import get_database_path
from pgportfolio.tools.db_builder import normalize_coin_name, normalize_timestamp


def normalize_history_table(database_file=None, chunk_size=5000, dry_run=False):
    """
    Rewrite ``History`` rows so ``coin`` names and ``date`` columns match the
    legacy ``Data.db`` format (coin tickers without OKX suffixes and timestamps
    stored in Unix seconds).
    """
    db_path = get_database_path(database_file)
    logging.info("Normalizing History table inside %s", db_path)
    connection = sqlite3.connect(db_path)
    try:
        cursor = connection.cursor()
        cursor.execute(
            "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='History'"
        )
        if cursor.fetchone()[0] == 0:
            raise ValueError("No History table found inside {}".format(db_path))

        if dry_run:
            cursor.execute("SELECT COUNT(*) FROM History")
            total = cursor.fetchone()[0]
            logging.info("Dry run: %s rows would be inspected.", total)
            return total, 0

        cursor.execute("DROP TABLE IF EXISTS History_normalized")
        cursor.execute(
            "CREATE TABLE History_normalized ("
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

        read_cursor = connection.cursor()
        write_cursor = connection.cursor()
        read_cursor.execute(
            "SELECT date, coin, high, low, open, close, volume, quoteVolume, weightedAverage FROM History"
        )

        total_rows = 0
        changed_rows = 0
        batch = []
        for row in read_cursor:
            total_rows += 1
            old_date, old_coin = row[0], row[1]
            new_date = normalize_timestamp(old_date)
            new_coin = normalize_coin_name(old_coin)
            if new_date != old_date or new_coin != old_coin:
                changed_rows += 1
            batch.append(
                (
                    new_date,
                    new_coin,
                    row[2],
                    row[3],
                    row[4],
                    row[5],
                    row[6],
                    row[7],
                    row[8],
                )
            )
            if len(batch) >= chunk_size:
                write_cursor.executemany(
                    "INSERT OR REPLACE INTO History_normalized "
                    "(date, coin, high, low, open, close, volume, quoteVolume, weightedAverage) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    batch,
                )
                connection.commit()
                batch = []
        if batch:
            write_cursor.executemany(
                "INSERT OR REPLACE INTO History_normalized "
                "(date, coin, high, low, open, close, volume, quoteVolume, weightedAverage) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                batch,
            )
            connection.commit()

        cursor.execute("DROP TABLE History")
        cursor.execute("ALTER TABLE History_normalized RENAME TO History")
        connection.commit()
        logging.info(
            "Normalization complete. %s rows processed; %s rows modified.",
            total_rows,
            changed_rows,
        )
        return total_rows, changed_rows
    finally:
        connection.close()


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Normalize OKX sqlite history files (coin names + timestamps)."
    )
    parser.add_argument(
        "--database_file",
        default=None,
        help="sqlite filename under ./database (default: Data.db from config)",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=5000,
        help="number of rows to batch per INSERT (default: 5000)",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="only report how many rows would be touched without modifying the DB",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="enable verbose logging output"
    )
    return parser.parse_args()


def main():
    args = _parse_args()
    logging.basicConfig(level=logging.INFO if not args.verbose else logging.DEBUG)
    normalize_history_table(
        database_file=args.database_file,
        chunk_size=args.chunk_size,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()

