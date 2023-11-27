#! coding: utf-8
import os
from typing import List

from llama_index import ServiceContext, StorageContext, VectorStoreIndex, SimpleDirectoryReader
from llama_index.indices.base import BaseIndex
from llama_index.node_parser import SimpleNodeParser
from llama_index.text_splitter import SentenceSplitter

from common.debug import cb_manager
from common.download import download
from common.llm import llm

service_context = ServiceContext.from_defaults(
    llm=llm,
    node_parser=SimpleNodeParser.from_defaults(text_splitter=SentenceSplitter(
        chunk_size=1024,
        chunk_overlap=20,
        callback_manager=cb_manager,
    )),
    callback_manager=cb_manager
)


def build_index(index_file: str, data_file: str) -> List[BaseIndex]:
    documents = SimpleDirectoryReader(input_files=[data_file]).load_data()
    for doc in documents:
        doc.excluded_llm_metadata_keys.append("file_path")
        doc.excluded_embed_metadata_keys.append("file_path")
    nodes = service_context.node_parser.get_nodes_from_documents(documents)
    storage_context = StorageContext.from_defaults()
    index = VectorStoreIndex(nodes, service_context=service_context, storage_context=storage_context)
    storage_context.persist(persist_dir=index_file)
    return [index]


def download_and_build_index(title: str, data_dir: str, index_dir: str) -> List[BaseIndex]:
    data_file = download(title, data_dir)
    return build_index(index_file=os.path.join(index_dir, os.path.relpath(data_file, data_dir)), data_file=data_file)
