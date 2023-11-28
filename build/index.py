#! coding: utf-8


import os
import shutil
from typing import List

from llama_index import ServiceContext, StorageContext, VectorStoreIndex, SimpleDirectoryReader, TreeIndex
from llama_index.node_parser import SimpleNodeParser
from llama_index.schema import TextNode, BaseNode
from llama_index.text_splitter import SentenceSplitter

from build.download import download
from common.config import ROOT_PATH, data_dir, index_dir
from common.llm import create_llm
from common.prompt import CH_SUMMARY_PROMPT, CH_INSERT_PROMPT

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
    documents = SimpleDirectoryReader(input_files=[data_file]).load_data()
    for doc in documents:
        doc.excluded_llm_metadata_keys.append("file_path")
        doc.excluded_embed_metadata_keys.append("file_path")
    return service_context.node_parser.get_nodes_from_documents(documents)


def build_index(index_file: str, data_file: str):
    # TODO
    if os.path.exists(index_file):
        return
    nodes = build_nodes(data_file)
    storage_context = StorageContext.from_defaults()
    vector_index = VectorStoreIndex(nodes,
                                    service_context=service_context,
                                    storage_context=storage_context,
                                    show_progress=True)
    tree_index = TreeIndex(nodes, num_children=8,
                           service_context=service_context,
                           storage_context=storage_context,
                           summary_template=CH_SUMMARY_PROMPT,
                           show_progress=True)
    storage_context.persist(persist_dir=index_file)


def download_and_build_index(title: str, data_dir: str, index_dir: str):
    data_file = download(title, data_dir)
    build_index(index_file=os.path.join(index_dir, os.path.relpath(data_file, data_dir)), data_file=data_file)


def build_all():
    titles = ['北京市', '上海市', '深圳市']
    for title in titles:
        download_and_build_index(title, data_dir, index_dir)


if __name__ == '__main__':
    build_all()
