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
    # data_file 是一个txt文件，请使用 SimpleDirectoryReader 和 service_context.node_parser 把一个文件解析成List[BaseNode],
    # 请进行 debug 注意metadata 的处理
    # https://docs.llamaindex.ai/en/stable/understanding/loading/loading.html#parsing-documents-into-nodes
    raise NotImplementedError


def build_index(index_file: str, data_file: str):
    if os.path.exists(index_file):
        return
    nodes = build_nodes(data_file)
    storage_context = StorageContext.from_defaults()
    # TODO
    # 基于 nodes 构建 VectorStoreIndex 和 TreeIndex 索引，并同意保存到 storage_context
    # https://docs.llamaindex.ai/en/stable/understanding/indexing/indexing.html#using-vector-store-index
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
