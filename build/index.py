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
    # 把输入的文档解析为Document对象，txt的文件，一个文件一个Document
    # 如果是pdf文件，会有特殊处理，一页一个Document
    documents = SimpleDirectoryReader(input_files=[data_file]).load_data()
    for doc in documents:
        # 过滤掉自动生成的 `file_path` 这个metadata, 防止对embedding结果造成干扰
        doc.excluded_llm_metadata_keys.append("file_path")
        doc.excluded_embed_metadata_keys.append("file_path")
    # 把 document 按句子进行分割成多个 nodes
    return service_context.node_parser.get_nodes_from_documents(documents)


def build_index(index_file: str, data_file: str):
    if os.path.exists(index_file):
        return
    nodes = build_nodes(data_file)
    # 两个index共用一个存储目录，可以复用DocumentStore
    storage_context = StorageContext.from_defaults()
    vector_index = VectorStoreIndex(nodes,
                                    service_context=service_context,
                                    storage_context=storage_context,
                                    show_progress=True)
    # TreeIndex的 num_children 是自底向上逐步summary, 生成parent node的时候，每个parent包含多少个children nodes
    # summary_template 可以替换为中文的prompt，更稳定的得到中文的summary
    tree_index = TreeIndex(nodes, num_children=8,
                           service_context=service_context,
                           storage_context=storage_context,
                           summary_template=CH_SUMMARY_PROMPT,
                           show_progress=True)
    # 把两个索引的生成数据存储到index_file这个目录
    storage_context.persist(persist_dir=index_file)


def download_and_build_index(title: str, data_dir: str, index_dir: str):
    data_file = download(title, data_dir)
    build_index(index_file=os.path.join(index_dir, os.path.relpath(data_file, data_dir)), data_file=data_file)


def build_all():
    # 单独构建全部的索引，后续查询会用到多索引
    titles = ['北京市', '上海市', '深圳市']
    for title in titles:
        download_and_build_index(title, data_dir, index_dir)


if __name__ == '__main__':
    build_all()
