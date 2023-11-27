#! coding: utf-8

import os

from common.build import download_and_build_index
from common.config import ROOT_PATH

if __name__ == '__main__':
    data_dir = os.path.join(ROOT_PATH, 'simple/data')
    index_dir = os.path.join(ROOT_PATH, 'simple/index')
    download_and_build_index('北京市', data_dir, index_dir)
