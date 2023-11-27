#! coding=utf-8
from typing import Dict, List

from llama_index import ServiceContext
from llama_index.indices.base import BaseIndex

from advanced.index import load_indices
from advanced.query import DocumentQueryEngine
from common.llm import create_llm


class Evaluator:
    def __init__(self):
        self.city_indices: Dict[str, List[BaseIndex]] = load_indices()
        self.llm = create_llm()
        self.service_context = ServiceContext.from_defaults(
            llm=self.llm
        )
    def evaluate(self):
        for city_name, indices in self.city_indices.items():
            doc_query_engine = DocumentQueryEngine(indices)
            doc_query_engine.create_query_engine(self.service_context)

