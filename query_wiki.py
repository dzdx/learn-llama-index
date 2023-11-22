#! coding: utf-8
import json
import os
from dataclasses import dataclass
from typing import List

from llama_index import VectorStoreIndex, ServiceContext, StorageContext, SimpleDirectoryReader, ComposableGraph, \
    get_response_synthesizer, Prompt, Document, load_index_from_storage, TreeIndex
from llama_index.callbacks import CallbackManager, LlamaDebugHandler
from llama_index.indices.base import BaseIndex
from llama_index.indices.postprocessor import LLMRerank
from llama_index.llms import OpenAI
from llama_index.prompts import PromptType
from llama_index.query_engine import RetrieverQueryEngine
from llama_index.response_synthesizers import ResponseMode

from prompt import CH_TEXT_QA_PROMPT_TMPL
from utils import ObjectEncoder

# os.environ['OPENAI_API_KEY'] = "xxxxxx"

DEBUG = False

if DEBUG:
    debug_handler = LlamaDebugHandler()
    cb_manager = CallbackManager([debug_handler])
else:
    debug_handler = None
    cb_manager = CallbackManager()

workspace_dir = os.path.dirname(__file__)
data_dir = os.path.join(workspace_dir, 'data')
index_dir = os.path.join(workspace_dir, 'index')
llm_gpt3 = OpenAI(temperature=0, model="gpt-3.5-turbo")
service_context = ServiceContext.from_defaults(
    llm=llm_gpt3, chunk_size=1024, callback_manager=cb_manager,
)


def load_index(file: str) -> BaseIndex:
    storage_context = StorageContext.from_defaults(persist_dir=os.path.join(index_dir, file))
    index = load_index_from_storage(
        storage_context=storage_context, service_context=service_context
    )
    return index


def build_index(file: str, documents: List[Document]) -> BaseIndex:
    storage_context = StorageContext.from_defaults()
    index = VectorStoreIndex.from_documents(
        documents,
        service_context=service_context,
        storage_context=storage_context,
    )
    index.index_struct.index_id = file
    storage_context.persist(persist_dir=os.path.join(index_dir, file))
    return index


city_indices = {}
for file in os.listdir(data_dir):
    basename = os.path.basename(file)
    if os.path.exists(os.path.join(index_dir, file)):
        index = load_index(file)
    else:
        documents = SimpleDirectoryReader(input_files=[os.path.join(data_dir, file)]).load_data()
        index = build_index(file, documents)
    city_indices[basename] = index


@dataclass
class DocQueryEngine:
    index: BaseIndex
    query_engine: RetrieverQueryEngine
    summary: str


query_engines = []
for city, vector_index in city_indices.items():
    retriever = vector_index.as_retriever()
    node_postprocessors = [
        LLMRerank(top_n=3)
    ]
    query_engine = RetrieverQueryEngine.from_args(
        retriever, node_postprocessors=node_postprocessors,
        service_context=service_context,
        text_qa_template=Prompt(
            CH_TEXT_QA_PROMPT_TMPL, prompt_type=PromptType.QUESTION_ANSWER
        ))
    summary = f"""
        此内容包含关于{city}的维基百科文章。
        如果您需要查找有关{city}的具体事实，请使用此索引。"
        如果您想分析多个城市，请不要使用此索引。
        """
    query_engines.append(DocQueryEngine(
        index=vector_index,
        query_engine=query_engine,
        summary=summary
    ))

graph = ComposableGraph.from_indices(
    TreeIndex,
    [e.index for e in query_engines],
    [e.summary for e in query_engines],
    service_context=service_context,
)

response_synthesizer = get_response_synthesizer(
    response_mode=ResponseMode.TREE_SUMMARIZE,
    text_qa_template=Prompt(
        CH_TEXT_QA_PROMPT_TMPL, prompt_type=PromptType.QUESTION_ANSWER
    ),
    service_context=service_context,
)

query_engine = graph.as_query_engine(
    response_synthesizer=response_synthesizer,
    custom_query_engines={e.index.index_id: e.query_engine for e in query_engines},
    service_context=service_context,
)


def chat(query):
    resp = query_engine.query(query)
    if debug_handler:
        for event in debug_handler.get_events():
            print(
                f"event_type={event.event_type}, content={json.dumps(event.payload, ensure_ascii=False, cls=ObjectEncoder)}")
    print(f"Query: {query}")
    print(f"Response: {resp}")


chat("北京气候如何")
chat("杭州位于中国哪里")
