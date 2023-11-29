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
            mode: str = "OR",
    ) -> None:
        """Init params."""
        if mode not in ("AND", "OR"):
            raise ValueError("Invalid mode.")
        self._mode = mode
        if retrievers is None:
            raise ValueError("Invalid retrievers.")
        self._retrievers = retrievers

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """Retrieve nodes given query."""
        if self._retrievers is None:
            return []
        # TODO
        # self.retrievers 包含了多个子 index 的 retriever， MultiRetriever 对多路retrieve的结果取并集，以node.node_id 作为key
        raise NotImplementedError


class QueryEngineToRetriever(BaseRetriever):
    def __init__(self, query_engine: RetrieverQueryEngine):
        self._retriever = query_engine

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        return self._retriever.retrieve(query_bundle)
