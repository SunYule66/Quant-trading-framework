from __future__ import print_function, absolute_import, division
import json
import os
import logging
from os import path


def add_packages(config, repeat=1):
    # place train_package at the project root (two levels up from this module)
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    package_dir = os.path.join(project_root, 'train_package')
    # make sure the train_package directory exists
    if not os.path.exists(package_dir):
        os.makedirs(package_dir, exist_ok=True)

    all_subdir = [int(s) for s in os.listdir(package_dir) if os.path.isdir(os.path.join(package_dir, s))]
    if all_subdir:
        max_dir_num = max(all_subdir)
    else:
        max_dir_num = 0
    indexes = []

    for i in range(repeat):
        max_dir_num += 1
        directory = os.path.join(package_dir, str(max_dir_num))
        config["random_seed"] = i
        os.makedirs(directory, exist_ok=True)
        indexes.append(max_dir_num)
        with open(os.path.join(directory, "net_config.json"), 'w') as outfile:
            json.dump(config, outfile, indent=4, sort_keys=True)
    logging.info("create indexes %s" % indexes)
    return indexes

