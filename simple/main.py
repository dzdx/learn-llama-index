import os
from typing import List

from llama_index import StorageContext, ServiceContext, load_indices_from_storage, SimpleDirectoryReader, \
    VectorStoreIndex, get_response_synthesizer
from llama_index.indices.base import BaseIndex
from llama_index.indices.postprocessor import LLMRerank
from llama_index.llms import OpenAI
from llama_index.node_parser import SimpleNodeParser
from llama_index.query_engine import RetrieverQueryEngine
from llama_index.response_synthesizers import ResponseMode
from llama_index.schema import BaseNode
from llama_index.text_splitter import SentenceSplitter

from debug import cb_manager

ROOT_PATH = os.path.dirname(__file__)
data_dir = os.path.join(ROOT_PATH, 'data')
index_dir = os.path.join(ROOT_PATH, 'index')

OPENAI_API_KEY = None
llm = OpenAI(temperature=0, model="gpt-3.5-turbo", api_key=OPENAI_API_KEY)
service_context = ServiceContext.from_defaults(
    llm=llm,
    node_parser=SimpleNodeParser.from_defaults(text_splitter=SentenceSplitter(
        chunk_size=1024,
        chunk_overlap=20,
        callback_manager=cb_manager,
    )),
)


def load_index(index_dir: str, service_context: ServiceContext = None) -> List[BaseIndex]:
    storage_context = StorageContext.from_defaults(persist_dir=index_dir)
    return load_indices_from_storage(
        storage_context=storage_context, service_context=service_context
    )


def build_index(index_dir: str, nodes: List[BaseNode], service_context: ServiceContext = None) -> List[BaseIndex]:
    storage_context = StorageContext.from_defaults()
    vector_index = VectorStoreIndex(nodes, service_context=service_context, storage_context=storage_context)
    storage_context.persist(persist_dir=index_dir)
    return [vector_index]


def load_or_build_city_index(service_context: ServiceContext, file) -> List[BaseIndex]:
    index_file = os.path.join(index_dir, file)
    if os.path.exists(index_file):
        indices = load_index(index_file, service_context)
    else:
        documents = SimpleDirectoryReader(input_files=[os.path.join(data_dir, file)]).load_data()
        for doc in documents:
            doc.excluded_llm_metadata_keys.append("file_path")
            doc.excluded_embed_metadata_keys.append("file_path")
        nodes = service_context.node_parser.get_nodes_from_documents(documents)
        indices = build_index(index_file, nodes)
    return indices


index = load_or_build_city_index(service_context, '北京市.txt')[0]
response_synthesizer = get_response_synthesizer(
    response_mode=ResponseMode.TREE_SUMMARIZE,
    service_context=service_context,
)

node_postprocessors = [
    LLMRerank(top_n=3, service_context=service_context)
]

query_engine = RetrieverQueryEngine.from_args(
    index.as_retriever(),
    node_postprocessors=node_postprocessors,
    response_synthesizer=response_synthesizer,
    service_context=service_context,
)

if __name__ == '__main__':
    print(query_engine.query("北京天气如何"))
