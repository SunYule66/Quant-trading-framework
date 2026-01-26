from __future__ import absolute_import
import copy
import json
import logging
import os
import time
from argparse import ArgumentParser
from datetime import datetime

from pgportfolio.constants import get_database_path
from pgportfolio.tools.configprocess import preprocess_config
from pgportfolio.tools.configprocess import load_config
from pgportfolio.tools.trade import save_test_data
from pgportfolio.tools.shortcut import execute_backtest
from pgportfolio.resultprocess import plot

MODE_CHOICES = (
    "train",
    "generate",
    "download_data",
    "backtest",
    "save_test_data",
    "plot",
    "table",
    "okx_dataset",
    "okx_trade",
)


def build_parser():
    parser = ArgumentParser()
    parser.add_argument("--mode", dest="mode",
                        help="run mode ({})".format(", ".join(MODE_CHOICES)),
                        metavar="MODE", default="plot",choices=MODE_CHOICES)
    parser.add_argument("--processes", dest="processes",
                        help="number of processes you want to start to train the network",
                        default="1")
    parser.add_argument("--repeat", dest="repeat",
                        help="repeat times of generating training subfolder",
                        default="1")
    parser.add_argument("--algo",
                        help="algo name or indexes of training_package ",
                        dest="algo", default="2")
    parser.add_argument("--algos",
                        help="algo names or indexes of training_package, seperated by \",\"",
                        dest="algos", default="2,best,ubah")
    parser.add_argument("--labels", dest="labels",
                        help="names that will shown in the figure caption or table header")
    parser.add_argument("--format", dest="format", default="raw",
                        help="format of the table printed")
    parser.add_argument("--device", dest="device", default="gpu",
                        help="device to be used to train")
    parser.add_argument("--folder", dest="folder", type=int,
                        help="folder(int) to load the config, neglect this option if loading from ./pgportfolio/net_config",
                        default=2)
    parser.add_argument("--live_steps", dest="live_steps", type=int,
                        help="limit iterations for okx_dataset realtime capture or okx_trade",
                        default=100)
    parser.add_argument("--coins", dest="coins",
                        help="comma separated coin list for live modes", default=None)
    parser.add_argument("--output_dir", dest="output_dir",
                        help="directory to store realtime snapshots", default="database")
    parser.add_argument("--source_dir", dest="source_dir",
                        help="directory of CSV snapshots to import into sqlite", default="Data")
    parser.add_argument("--database_file", dest="database_file",
                        help="override sqlite filename under ./database (default: Data.db)",
                        default=None)
    parser.add_argument(
        "--dataset_type",
        dest="dataset_type",
        choices=("historical", "realtime", "all"),
        default="realtime",
        help="choose whether okx_dataset collects historical data, realtime snapshots, or both",
    )
    return parser


def main():
    parser = build_parser()
    options = parser.parse_args()
    # use project-root-local directories (avoid creating folders in caller's CWD)
    project_root = os.path.abspath(os.path.dirname(__file__))
    train_package_dir = os.path.join(project_root, "train_package")
    database_dir = os.path.join(project_root, "database")

    os.makedirs(train_package_dir, exist_ok=True)
    os.makedirs(database_dir, exist_ok=True)

    if options.mode == "train":
        import pgportfolio.autotrain.training
        if not options.algo:
            pgportfolio.autotrain.training.train_all(int(options.processes), options.device)
        else:
            if options.folder is None:
                raise ValueError("--folder must be provided when --algo is specified")
            pgportfolio.autotrain.training.train_selected(
                options.folder, device=options.device
            )
    elif options.mode == "generate":
        import pgportfolio.autotrain.generate as generate
        logging.basicConfig(level=logging.INFO)
        generate.add_packages(load_config(), int(options.repeat))
    elif options.mode == "download_data":
        from pgportfolio.marketdata.datamatrices import DataMatrices
        with open("./pgportfolio/net_config.json") as file:
            config = json.load(file)
        config = preprocess_config(config)
        start = time.mktime(datetime.strptime(config["input"]["start_date"], "%Y/%m/%d").timetuple())
        end = time.mktime(datetime.strptime(config["input"]["end_date"], "%Y/%m/%d").timetuple())
        DataMatrices(start=start,
                     end=end,
                     feature_number=config["input"]["feature_number"],
                     window_size=config["input"]["window_size"],
                     online=True,
                     period=config["input"]["global_period"],
                     volume_average_days=config["input"]["volume_average_days"],
                     coin_filter=config["input"]["coin_number"],
                     is_permed=config["input"]["is_permed"],
                     test_portion=config["input"]["test_portion"],
                     portion_reversed=config["input"]["portion_reversed"])
    elif options.mode == "backtest":
        config = _config_by_algo(options.algo)
        _set_logging_by_algo(logging.DEBUG, logging.DEBUG, options.algo, "backtestlog")
        execute_backtest(options.algo, config)
    elif options.mode == "save_test_data":
        # This is used to export the test data
        save_test_data(load_config(options.folder))
    elif options.mode == "plot":
        logging.basicConfig(level=logging.INFO)
        algos = options.algos.split(",")
        if options.labels:
            labels = options.labels.replace("_"," ")
            labels = labels.split(",")
        else:
            labels = algos
        plot.plot_backtest(load_config(), algos, labels)
    elif options.mode == "table":
        algos = options.algos.split(",")
        if options.labels:
            labels = options.labels.replace("_"," ")
            labels = labels.split(",")
        else:
            labels = algos
        plot.table_backtest(load_config(), algos, labels, format=options.format)
    elif options.mode == "okx_dataset":
        logging.basicConfig(level=logging.INFO)
        dataset_type = (options.dataset_type or "historical").lower()
        if dataset_type not in ("historical", "realtime", "all"):
            raise ValueError("Unsupported --dataset_type {}".format(dataset_type))
        if options.algo:
            config = _config_by_algo(options.algo)
        else:
            config = load_config()
        config = copy.deepcopy(config)
        target_db = options.database_file or config["input"].get("database_file")
        if target_db:
            config["input"]["database_file"] = target_db
        phases = []
        if dataset_type in ("historical", "all"):
            from pgportfolio.marketdata.datamatrices import DataMatrices
            matrices = DataMatrices.create_from_config(config)
            db_path = get_database_path(config["input"].get("database_file"))
            logging.info(
                "Historical extraction complete. %s coins from %s to %s stored in %s",
                len(matrices.coin_list) if matrices.coin_list else 0,
                config["input"]["start_date"],
                config["input"]["end_date"],
                db_path,
            )
            phases.append("historical")
        if dataset_type in ("realtime", "all"):
            from pgportfolio.live import OkxRealtimeCollector
            from pgportfolio.tools.db_builder import CSVDatabaseBuilder

            coin_list = _coin_list_from_arg(options.coins)
            live_steps = options.live_steps or config["trading"].get("live_steps") or None
            output_dir = options.output_dir or os.path.join("database", "realtime")
            collector = OkxRealtimeCollector(
                config=config,
                coins=coin_list,
                period=config["input"]["global_period"],
                window_size=config["input"]["window_size"],
                output_dir=output_dir,
                max_snapshots=live_steps,
                quote_currency=config["trading"].get("quote_currency", "BTC"),
            )
            collector.run_forever()
            source_dir = options.source_dir or output_dir
            builder = CSVDatabaseBuilder(
                source_dir=source_dir,
                database_file=config["input"].get("database_file"),
            )
            builder.build()
            phases.append("realtime")
        if not phases:
            raise ValueError("okx_dataset needs --dataset_type historical|realtime|all")
    elif options.mode == "okx_trade":
        if not options.algo or not options.algo.isdigit():
            raise ValueError("--mode okx_trade requires --algo=<trained_folder_index>")
        config = _config_by_algo(options.algo)
        net_dir = _net_dir_for_algo(options.algo)
        live_steps = options.live_steps or config["trading"].get("live_steps") or None
        from pgportfolio.trade import OkxTrader
        logging.basicConfig(level=logging.INFO)
        trader = OkxTrader(
            config=config,
            net_dir=net_dir,
            total_steps=live_steps,
            device=options.device,
        )
        trader.start_trading()

def _set_logging_by_algo(console_level, file_level, algo, name):
    if algo.isdigit():
            # put log files under the project-local train_package folder
            base = os.path.abspath(os.path.dirname(__file__))
            algo_dir = os.path.join(base, "train_package", str(algo))
            os.makedirs(algo_dir, exist_ok=True)
            logfile = os.path.join(algo_dir, name)
            logging.basicConfig(filename=logfile, level=file_level)
            console = logging.StreamHandler()
            console.setLevel(console_level)
            logging.getLogger().addHandler(console)
    else:
        logging.basicConfig(level=console_level)


def _config_by_algo(algo):
    """
    :param algo: a string represent index or algo name
    :return : a config dictionary
    """
    if not algo:
        raise ValueError("please input a specific algo")
    elif algo.isdigit():
        config = load_config(algo)
    else:
        config = load_config()
    return config


def _coin_list_from_arg(raw):
    if not raw:
        return None
    return [token.strip() for token in raw.split(",") if token.strip()]


def _net_dir_for_algo(algo):
    if not algo or not algo.isdigit():
        return None
    base = os.path.abspath(os.path.dirname(__file__))
    net_dir = os.path.join(base, "train_package", str(algo), "netfile")
    if not os.path.exists(net_dir):
        raise ValueError("net_dir {} does not exist, please train the model first.".format(net_dir))
    return net_dir

if __name__ == "__main__":
    main()
