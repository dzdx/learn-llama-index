#! coding: utf-8
from dataclasses import dataclass
from typing import List, Dict

from llama_index import ServiceContext, ComposableGraph, \
    get_response_synthesizer, Prompt, TreeIndex, VectorStoreIndex, DocumentSummaryIndex, KeywordTableIndex
from llama_index.indices.base import BaseIndex
from llama_index.indices.document_summary.base import DocumentSummaryRetrieverMode
from llama_index.indices.postprocessor import LLMRerank
from llama_index.indices.query.base import BaseQueryEngine
from llama_index.prompts import PromptType
from llama_index.query_engine import RetrieverQueryEngine
from llama_index.response_synthesizers import ResponseMode

from common.prompt import CH_TEXT_QA_PROMPT_TMPL, CH_QUERY_PROMPT, CH_CHOICE_SELECT_PROMPT, CH_TREE_SUMMARIZE_PROMPT
from retrievers import CustomRetriever


@dataclass
class DocumentQueryEngine:
    indices: List[BaseIndex]
    summary: str = ""

    def first_index(self):
        return self.indices[0]

    def _create_retrievers(self):
        ret = []
        for index in self.indices:
            if isinstance(index, VectorStoreIndex):
                ret.append(index.as_retriever(similarity_top_k=5))
            elif isinstance(index, DocumentSummaryIndex):
                ret.append(index.as_retriever(retriever_mode=DocumentSummaryRetrieverMode.EMBEDDING))
            elif isinstance(index, KeywordTableIndex):
                ret.append(index.as_retriever())
        return ret

    def doc_store(self):
        return self.indices[0].docstore

    def create_query_engine(self, service_context: ServiceContext) -> RetrieverQueryEngine:
        retriever = CustomRetriever(self._create_retrievers())
        node_postprocessors = [
            LLMRerank(top_n=4, choice_batch_size=2, choice_select_prompt=CH_CHOICE_SELECT_PROMPT,
                      service_context=service_context)
        ]
        return RetrieverQueryEngine.from_args(
            retriever, node_postprocessors=node_postprocessors,
            service_context=service_context,
            text_qa_template=Prompt(
                CH_TEXT_QA_PROMPT_TMPL, prompt_type=PromptType.QUESTION_ANSWER
            ))


def create_compose_query_engine(city_indices: Dict[str, List[BaseIndex]],
                                service_context: ServiceContext) -> BaseQueryEngine:
    query_engines = []
    for city, indices in city_indices.items():
        summary = f"""
            此内容包含关于{city}的维基百科文章。
            如果您需要查找有关{city}的具体事实，请使用此索引。"
            如果您想分析多个城市，请不要使用此索引。
            """
        query_engines.append(DocumentQueryEngine(
            indices=indices,
            summary=summary
        ))

    graph = ComposableGraph.from_indices(
        TreeIndex,
        [e.first_index() for e in query_engines],
        [e.summary for e in query_engines],
        service_context=service_context,
    )

    response_synthesizer = get_response_synthesizer(
        response_mode=ResponseMode.TREE_SUMMARIZE,
        text_qa_template=Prompt(
            CH_TEXT_QA_PROMPT_TMPL, prompt_type=PromptType.QUESTION_ANSWER
        ),
        summary_template=CH_TREE_SUMMARIZE_PROMPT,
        service_context=service_context,
    )

    return graph.as_query_engine(
        response_synthesizer=response_synthesizer,
        custom_query_engines={e.first_index().index_id: e.create_query_engine(service_context) for e in query_engines},
        service_context=service_context,
        query_template=CH_QUERY_PROMPT,
    )
