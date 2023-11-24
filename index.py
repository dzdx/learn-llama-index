import os
from typing import List, Dict

from llama_index import Document, VectorStoreIndex, StorageContext, ServiceContext, \
    load_indices_from_storage, SimpleDirectoryReader
from llama_index.indices.base import BaseIndex

from config import ROOT_PATH

data_dir = os.path.join(ROOT_PATH, 'data')
index_dir = os.path.join(ROOT_PATH, 'index')


def load_index(index_dir: str, service_context: ServiceContext = None) -> List[BaseIndex]:
    storage_context = StorageContext.from_defaults(persist_dir=index_dir)
    return load_indices_from_storage(
        storage_context=storage_context, service_context=service_context
    )


def build_index(index_dir: str, documents: List[Document], service_context: ServiceContext = None) -> List[BaseIndex]:
    storage_context = StorageContext.from_defaults()
    vector_index = VectorStoreIndex.from_documents(
        documents,
        service_context=service_context,
        storage_context=storage_context,
    )
    storage_context.persist(persist_dir=index_dir)
    return [vector_index]


def load_or_build_city_indices(service_context: ServiceContext):
    city_indices: Dict[str, List[BaseIndex]] = {}
    for file in os.listdir(data_dir):
        basename = os.path.splitext(file)[0]
        index_file = os.path.join(index_dir, file)
        if os.path.exists(index_file):
            indices = load_index(index_file, service_context)
        else:
            documents = SimpleDirectoryReader(input_files=[os.path.join(data_dir, file)]).load_data()
            for doc in documents:
                doc.excluded_llm_metadata_keys.append("file_path")
                doc.excluded_embed_metadata_keys.append("file_path")
            indices = build_index(index_file, documents)
        city_indices[basename] = indices
    return city_indices
