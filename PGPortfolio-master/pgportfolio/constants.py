#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
from os import path

def get_database_path(database_file=None):
    """
    获取数据库路径
    :param database_file: 数据库文件名，如果为None则使用默认的Data.db
    :return: 数据库文件的完整路径
    """
    if database_file is None:
        database_file = "Data.db"
    # project root is the parent directory of the pgportfolio package
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    base_path = os.path.join(project_root, 'database')
    return path.join(base_path, database_file)


def get_project_root():
    """Return the absolute path to the project root (one level above pgportfolio)."""
    return os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))


def get_train_package_path(*parts):
    """Return a path to the train_package folder inside the project root.
    If parts are provided they are joined after train_package.
    """
    base = os.path.join(get_project_root(), 'train_package')
    if parts:
        return path.join(base, *map(str, parts))
    return base

# 默认数据库路径（保持向后兼容）
DATABASE_DIR = get_database_path()
CONFIG_FILE_DIR = 'net_config.json'
LAMBDA = 1e-4  # lambda in loss function 5 in training
   # About time
NOW = 0
FIVE_MINUTES = 60 * 5
FIFTEEN_MINUTES = FIVE_MINUTES * 3
HALF_HOUR = FIFTEEN_MINUTES * 2
HOUR = HALF_HOUR * 2
TWO_HOUR = HOUR * 2
FOUR_HOUR = HOUR * 4
DAY = HOUR * 24
YEAR = DAY * 365
   # trading table name
TABLE_NAME = 'test'

