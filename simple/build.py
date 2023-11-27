#! coding: utf-8

import os

from llama_index import ServiceContext, StorageContext, VectorStoreIndex, SimpleDirectoryReader
from llama_index.llms import OpenAI
from llama_index.node_parser import SimpleNodeParser
from llama_index.text_splitter import SentenceSplitter

from common.config import OPENAI_API_KEY, LLM_CACHE_ENABLED
from common.config import ROOT_PATH
from common.download import download
from common.llm import CachedLLM

_llm_gpt3 = OpenAI(temperature=0, model="gpt-3.5-turbo", api_key=OPENAI_API_KEY)
llm = CachedLLM(_llm_gpt3,
                '.llm_cache',
                request_timeout=15,
                enable_cache=LLM_CACHE_ENABLED)
service_context = ServiceContext.from_defaults(
    llm=llm,
    node_parser=SimpleNodeParser.from_defaults(text_splitter=SentenceSplitter(
        chunk_size=1024,
        chunk_overlap=20,
    )),
)


def build_index(index_file: str, data_file: str):
    documents = SimpleDirectoryReader(input_files=[data_file]).load_data()
    for doc in documents:
        doc.excluded_llm_metadata_keys.append("file_path")
        doc.excluded_embed_metadata_keys.append("file_path")
    nodes = service_context.node_parser.get_nodes_from_documents(documents)
    storage_context = StorageContext.from_defaults()
    index = VectorStoreIndex(nodes, service_context=service_context, storage_context=storage_context)
    storage_context.persist(persist_dir=index_file)


def download_and_build_index(title: str, data_dir: str, index_dir: str):
    data_file = download(title, data_dir)
    build_index(index_file=os.path.join(index_dir, os.path.relpath(data_file, data_dir)), data_file=data_file)


if __name__ == '__main__':
    data_dir = os.path.join(ROOT_PATH, 'simple/data')
    index_dir = os.path.join(ROOT_PATH, 'simple/index')
    download_and_build_index('北京市', data_dir, index_dir)
