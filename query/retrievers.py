from typing import List

from llama_index import QueryBundle
from llama_index.indices.base_retriever import BaseRetriever
from llama_index.query_engine import RetrieverQueryEngine
from llama_index.schema import NodeWithScore


class MultiRetriever(BaseRetriever):
    """Custom retriever that performs both Vector search and Knowledge Graph search"""

    def __init__(
            self,
            retrievers,
    ) -> None:
        if retrievers is None:
            raise ValueError("Invalid retrievers.")
        self._retrievers = retrievers

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        if self._retrievers is None:
            return []
        # 对多个retrieve的召回结果进行合并
        combined_dict = {}
        for retriever in self._retrievers:
            cur_nodes = retriever.retrieve(query_bundle)
            combined_dict.update({n.node.node_id: n for n in cur_nodes})
        retrieve_nodes = sorted(list(combined_dict.values()), key=lambda n: n.node_id)
        return retrieve_nodes


class QueryEngineToRetriever(BaseRetriever):
    def __init__(self, query_engine: RetrieverQueryEngine):
        self._retriever = query_engine

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        return self._retriever.retrieve(query_bundle)
