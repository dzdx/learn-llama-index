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
        """Retrieve nodes given query."""
        if self._retrievers is None:
            return []
        # init collection
        retriever_names = []
        node_ids = []
        combined_dict = {}
        for retriever in self._retrievers:
            # parse retriever name, and retrieve nodes
            cur_name = retriever.__class__.__name__
            cur_nodes = retriever.retrieve(query_bundle)
            # build collection(id set & node dict)
            cur_ids_set = {n.node.node_id for n in cur_nodes}
            cur_nodes_dict = {n.node.node_id: n for n in cur_nodes}
            # record
            node_ids.append(cur_ids_set)
            combined_dict.update(cur_nodes_dict)
            retriever_names.append(cur_name)
        retrieve_ids = batch_option(node_ids)
        retrieve_ids = sorted(retrieve_ids)
        retrieve_nodes = [combined_dict[rid] for rid in retrieve_ids]
        return retrieve_nodes


def batch_option(list_with_set):
    size = len(list_with_set)
    if size == 0:
        return None
    res = list_with_set[0]
    if size == 1:
        return res

    for cur_set in list_with_set:
        res = res.union(cur_set)
    return res


class QueryEngineToRetriever(BaseRetriever):
    def __init__(self, query_engine: RetrieverQueryEngine):
        self._retriever = query_engine

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        return self._retriever.retrieve(query_bundle)
