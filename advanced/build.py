#! coding: utf-8

import os

from common.build import download_and_build_index
from common.config import ROOT_PATH

if __name__ == '__main__':
    titles = ['北京市', '上海市', '深圳市', '杭州市', '南京市']
    data_dir = os.path.join(ROOT_PATH, 'advanced/data')
    index_dir = os.path.join(ROOT_PATH, 'advanced/index')
    for title in titles:
        download_and_build_index(title, data_dir, index_dir)
