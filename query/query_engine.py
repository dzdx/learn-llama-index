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


def load_index(title: str, service_context: ServiceContext=None) -> List[BaseIndex]:
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
    # 采用TreeSummarize的方式对多个上下文进行逐步总结，防止超过llm的context limit
    # 同时用中文prompt得到更加稳定的中文summary
    return get_response_synthesizer(
        response_mode=ResponseMode.TREE_SUMMARIZE,
        summary_template=CH_TREE_SUMMARIZE_PROMPT,
        service_context=service_context,
    )


@dataclass
class DocumentQueryEngineFactory:
    indices: List[BaseIndex]
    summary: Optional[str] = ""

    def first_index(self):
        return self.indices[0]

    def create_retrievers(self):
        ret = []
        for index in self.indices:
            # 每个索引可能会用不同的生成retriever的方式
            # 取 `similarity_top_k` 和query 最相似的 node
            if isinstance(index, VectorStoreIndex):
                ret.append(index.as_retriever(similarity_top_k=8))
            if isinstance(index, TreeIndex):
                # TreeRetriever采用自顶向下逐步检索的方式得到叶子节点
                ret.append(index.as_retriever(retriever_mode=TreeRetrieverMode.SELECT_LEAF_EMBEDDING))
        return ret

    def doc_store(self):
        return self.indices[0].docstore

    def create_query_engine(self, service_context: ServiceContext) -> RetrieverQueryEngine:
        # 组合多索引召回、排序，答案合成，构建最终的query engine
        retriever = MultiRetriever(self.create_retrievers())
        # LLMRerank只选取最相关的top_n, 进一步提高命中率，防止召回阶段拿到不相关的内容
        node_postprocessors = [
            LLMRerank(top_n=4, choice_batch_size=2, choice_select_prompt=CH_CHOICE_SELECT_PROMPT,
                      service_context=service_context)
        ]
        return RetrieverQueryEngine.from_args(
            retriever,
            node_postprocessors=node_postprocessors,
            service_context=service_context,
            response_synthesizer=create_response_synthesizer(service_context)
        )
