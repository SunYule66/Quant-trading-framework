from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import os
import time
from multiprocessing import Process
from pgportfolio.learn.tradertrainer import TraderTrainer
from pgportfolio.constants import get_train_package_path
from pgportfolio.tools.configprocess import load_config


def train_one(save_path, config, log_file_dir, index, logfile_level, console_level, device):
    """
    train an agent
    :param save_path: the path to save the PyTorch model (.pt), could be None
    :param config: the json configuration file
    :param log_file_dir: the directory to save the tensorboard logging file, could be None
    :param index: identifier of this train, which is also the sub directory in the train_package,
    if it is 0. nothing would be saved into the summary file.
    :param logfile_level: logging level of the file
    :param console_level: logging level of the console
    :param device: 0 or 1 to show which gpu to use, if 0, means use cpu instead of gpu
    :return : the Result namedtuple
    """
    if log_file_dir:
        logging.basicConfig(filename=log_file_dir.replace("tensorboard","programlog"),
                            level=logfile_level)
        console = logging.StreamHandler()
        console.setLevel(console_level)
        logging.getLogger().addHandler(console)
    print("training at %s started" % index)
    return TraderTrainer(config, save_path=save_path, device=device).train_net(log_file_dir=log_file_dir, index=index)

def _logging_levels_for_processes(processes):
    if processes == 1:
        return logging.INFO, logging.DEBUG
    else:
        return logging.WARNING, logging.INFO


def train_all(processes=1, device="cpu"):
    """
    train all the agents in the train_package folders

    :param processes: the number of the processes. If equal to 1, the logging level is debug
                      at file and info at console. If greater than 1, the logging level is
                      info at file and warming at console.
    """
    console_level, logfile_level = _logging_levels_for_processes(processes)
    # use project-local train_package folder
    train_dir = get_train_package_path()
    if not os.path.exists(train_dir):
        os.makedirs(train_dir, exist_ok=True)
    all_subdir = os.listdir(train_dir)
    all_subdir.sort()
    pool = []
    for dir in all_subdir:
        # train only if the log dir does not exist
        if not str.isdigit(dir):
            return
        # NOTE: logfile is for compatibility reason
        target_base = os.path.join(train_dir, dir)
        if not (os.path.isdir(os.path.join(target_base, "tensorboard")) or os.path.isdir(os.path.join(target_base, "logfile"))):
            p = Process(target=train_one, args=(
                os.path.join(target_base, "netfile"),
                load_config(dir),
                os.path.join(target_base, "tensorboard"),
                dir, logfile_level, console_level, device))
            p.start()
            pool.append(p)
        else:
            continue

        # suspend if the processes are too many
        wait = True
        while wait:
            time.sleep(5)
            for p in pool:
                alive = p.is_alive()
                if not alive:
                    pool.remove(p)
            if len(pool)<processes:
                wait = False
    print("All the Tasks are Over")


def train_selected(subdirs, device="cpu"):
    """
    train specific agents identified by their sub folder names inside train_package

    :param subdirs: iterable of folder names or a single folder
    :param device: cpu/gpu selection propagated to Trainer
    """
    if not subdirs:
        raise ValueError("train_selected requires at least one sub directory")

    if isinstance(subdirs, (str, int)):
        subdirs = [subdirs]

    console_level, logfile_level = _logging_levels_for_processes(1)
    train_dir = get_train_package_path()
    if not os.path.exists(train_dir):
        raise ValueError("train_package directory does not exist. Generate configs first.")

    for dir in subdirs:
        dir = str(dir)
        target_dir = os.path.join(train_dir, dir)
        if not os.path.isdir(target_dir):
            raise ValueError("Specified train_package folder {} does not exist".format(dir))

        tensorboard_dir = os.path.join(target_dir, "tensorboard")
        logfile_dir = os.path.join(target_dir, "logfile")

        if os.path.isdir(tensorboard_dir) or os.path.isdir(logfile_dir):
            logging.info("Skip %s because training artifacts already exist", dir)
            continue

        os.makedirs(tensorboard_dir, exist_ok=True)

        train_one(
            os.path.join(target_dir, "netfile"),
            load_config(dir),
            tensorboard_dir,
            dir,
            logfile_level,
            console_level,
            device,
        )
