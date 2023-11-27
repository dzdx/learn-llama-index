import os
from typing import List, Dict

from llama_index import StorageContext, ServiceContext, \
    load_indices_from_storage
from llama_index.indices.base import BaseIndex

from common.config import ROOT_PATH

data_dir = os.path.join(ROOT_PATH, 'advanced/data')
index_dir = os.path.join(ROOT_PATH, 'advanced/index')


def load_index(index_file: str, service_context: ServiceContext = None) -> List[BaseIndex]:
    storage_context = StorageContext.from_defaults(persist_dir=index_file)
    return load_indices_from_storage(
        storage_context=storage_context, service_context=service_context
    )


def load_indices(service_context: ServiceContext) -> Dict[str, List[BaseIndex]]:
    indices: Dict[str, List[BaseIndex]] = {}
    for file in os.listdir(data_dir):
        basename = os.path.splitext(file)[0]
        indices[basename] = load_index(os.path.join(index_dir, file), service_context)
    return indices
