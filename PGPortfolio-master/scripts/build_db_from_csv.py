import argparse
import logging

from pgportfolio.tools.db_builder import CSVDatabaseBuilder


def build_parser():
    parser = argparse.ArgumentParser(
        description="Convert CSV snapshots in ./Data into the sqlite database used by PGPortfolio."
    )
    parser.add_argument(
        "--source",
        default="Data",
        help="Directory containing CSV files (default: Data).",
    )
    parser.add_argument(
        "--database-file",
        default=None,
        help="Optional sqlite filename (stored under ./database). Default is Data.db.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1000,
        help="Number of rows per insert batch (default: 1000).",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Logging verbosity.",
    )
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))

    builder = CSVDatabaseBuilder(
        source_dir=args.source,
        database_file=args.database_file,
        chunk_size=args.chunk_size,
    )
    builder.build()


if __name__ == "__main__":
    main()

