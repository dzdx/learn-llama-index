#! coding: utf-8


import os
from typing import List

from llama_index import ServiceContext, StorageContext, VectorStoreIndex, SimpleDirectoryReader, TreeIndex
from llama_index.node_parser import SimpleNodeParser
from llama_index.schema import BaseNode
from llama_index.text_splitter import SentenceSplitter

from build.download import download
from common.config import data_dir, index_dir
from common.llm import create_llm
from common.prompt import CH_SUMMARY_PROMPT

llm = create_llm(timeout=60)
service_context = ServiceContext.from_defaults(
    llm=llm,
    node_parser=SimpleNodeParser.from_defaults(text_splitter=SentenceSplitter(
        chunk_size=1024,
        chunk_overlap=200,
    )),
)


def build_nodes(data_file: str) -> List[BaseNode]:
    # TODO
    raise NotImplementedError


def build_index(index_file: str, data_file: str):
    if os.path.exists(index_file):
        return
    # TODO
    raise NotImplementedError


def download_and_build_index(title: str, data_dir: str, index_dir: str):
    data_file = download(title, data_dir)
    build_index(index_file=os.path.join(index_dir, os.path.relpath(data_file, data_dir)), data_file=data_file)


def build_all():
    titles = ['北京市', '上海市', '深圳市']
    for title in titles:
        download_and_build_index(title, data_dir, index_dir)


if __name__ == '__main__':
    build_all()
