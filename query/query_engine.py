#! coding: utf-8
import os
from dataclasses import dataclass
from typing import List, Dict, Optional

from llama_index import ServiceContext, get_response_synthesizer, VectorStoreIndex, StorageContext, \
    load_indices_from_storage, TreeIndex
from llama_index.indices.base import BaseIndex
from llama_index.indices.postprocessor import LLMRerank
from llama_index.indices.tree.base import TreeRetrieverMode
from llama_index.query_engine import RetrieverQueryEngine
from llama_index.response_synthesizers import ResponseMode, BaseSynthesizer

from common.config import index_dir
from common.prompt import CH_CHOICE_SELECT_PROMPT, CH_TREE_SUMMARIZE_PROMPT
from query.retrievers import MultiRetriever


def load_index(title: str, service_context: ServiceContext = None) -> List[BaseIndex]:
    storage_context = StorageContext.from_defaults(persist_dir=os.path.join(index_dir, title))
    return load_indices_from_storage(
        storage_context=storage_context,
        service_context=service_context,
    )


def load_indices(service_context: ServiceContext) -> Dict[str, List[BaseIndex]]:
    indices: Dict[str, List[BaseIndex]] = {}
    for title in os.listdir(index_dir):
        indices[title] = load_index(title, service_context)
    return indices


def create_response_synthesizer(service_context: ServiceContext = None) -> BaseSynthesizer:
    # TODO
    # https://docs.llamaindex.ai/en/stable/module_guides/querying/response_synthesizers/root.html#get-started
    raise NotImplementedError


@dataclass
class DocumentQueryEngineFactory:
    indices: List[BaseIndex]
    summary: Optional[str] = ""

    def first_index(self):
        return self.indices[0]

    def create_retrievers(self):
        # TODO
        raise NotImplementedError

    def doc_store(self):
        return self.indices[0].docstore

    def create_query_engine(self, service_context: ServiceContext) -> RetrieverQueryEngine:
        # TODO
        raise NotImplementedError
