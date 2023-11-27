#! coding: utf-8
import json
from dataclasses import dataclass
from typing import List

from llama_index import ServiceContext, ComposableGraph, \
    get_response_synthesizer, Prompt, TreeIndex
from llama_index.callbacks import CBEventType
from llama_index.indices.base import BaseIndex
from llama_index.indices.postprocessor import LLMRerank
from llama_index.node_parser import SimpleNodeParser
from llama_index.prompts import PromptType
from llama_index.query_engine import RetrieverQueryEngine
from llama_index.response_synthesizers import ResponseMode
from llama_index.text_splitter import SentenceSplitter

from debug import cb_manager, debug_handler
from index import load_or_build_cities_indices
from llm import llm
from prompt import CH_TEXT_QA_PROMPT_TMPL, CH_QUERY_PROMPT, CH_CHOICE_SELECT_PROMPT, CH_TREE_SUMMARIZE_PROMPT
from retrievers import CustomRetriever
from utils import ObjectEncoder

service_context = ServiceContext.from_defaults(
    llm=llm,
    node_parser=SimpleNodeParser.from_defaults(text_splitter=SentenceSplitter(
        chunk_size=1024,
        chunk_overlap=20,
        callback_manager=cb_manager,
    )),
    callback_manager=cb_manager
)

city_indices = load_or_build_cities_indices(service_context)


@dataclass
class DocQueryEngine:
    indices: List[BaseIndex]
    summary: str

    def first_index(self):
        return self.indices[0]

    def create_query_engine(self):
        sub_retrievers = []
        for index in self.indices:
            sub_retrievers.append(index.as_retriever())
        retriever = CustomRetriever(sub_retrievers)
        node_postprocessors = [
            LLMRerank(top_n=3, choice_select_prompt=CH_CHOICE_SELECT_PROMPT, service_context=service_context)
        ]
        return RetrieverQueryEngine.from_args(
            retriever, node_postprocessors=node_postprocessors,
            service_context=service_context,
            text_qa_template=Prompt(
                CH_TEXT_QA_PROMPT_TMPL, prompt_type=PromptType.QUESTION_ANSWER
            ))


query_engines = []
for city, indices in city_indices.items():
    summary = f"""
        此内容包含关于{city}的维基百科文章。
        如果您需要查找有关{city}的具体事实，请使用此索引。"
        如果您想分析多个城市，请不要使用此索引。
        """
    query_engines.append(DocQueryEngine(
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

query_engine = graph.as_query_engine(
    response_synthesizer=response_synthesizer,
    custom_query_engines={e.first_index().index_id: e.create_query_engine() for e in query_engines},
    service_context=service_context,
    query_template=CH_QUERY_PROMPT,
)


def ask(query):
    resp = query_engine.query(query)
    if debug_handler:
        for event in debug_handler.get_events():
            if event.event_type in (CBEventType.LLM, CBEventType.RETRIEVE):
                print(
                    f"[DebugInfo] event_type={event.event_type}, content={json.dumps(event.payload, ensure_ascii=False, cls=ObjectEncoder)}")
    return resp
