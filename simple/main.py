import json
import os
from typing import List

from llama_index import StorageContext, ServiceContext, load_indices_from_storage, get_response_synthesizer
from llama_index.callbacks import CBEventType
from llama_index.indices.base import BaseIndex
from llama_index.indices.postprocessor import LLMRerank
from llama_index.query_engine import RetrieverQueryEngine
from llama_index.response_synthesizers import ResponseMode

from common.config import ROOT_PATH
from common.debug import cb_manager, debug_handler
from common.llm import llm
from common.utils import ObjectEncoder

index_dir = os.path.join(ROOT_PATH, 'simple/index')

OPENAI_API_KEY = None
service_context = ServiceContext.from_defaults(
    llm=llm,
    callback_manager=cb_manager
)


def load_index(index_file: str) -> List[BaseIndex]:
    storage_context = StorageContext.from_defaults(persist_dir=index_file)
    return load_indices_from_storage(
        storage_context=storage_context, service_context=service_context
    )


index = load_index(os.path.join(index_dir, '北京市.txt'))[0]
response_synthesizer = get_response_synthesizer(
    response_mode=ResponseMode.TREE_SUMMARIZE,
    service_context=service_context,
)

node_postprocessors = [
    LLMRerank(top_n=3, service_context=service_context)
]

query_engine = RetrieverQueryEngine.from_args(
    index.as_retriever(),
    node_postprocessors=node_postprocessors,
    response_synthesizer=response_synthesizer,
    service_context=service_context,
)


def ask_question(query):
    response = query_engine.query(query)
    if debug_handler:
        for event in debug_handler.get_events():
            if event.event_type in (CBEventType.LLM, CBEventType.RETRIEVE):
                print(
                    f"[DebugInfo] event_type={event.event_type}, content={json.dumps(event.payload, ensure_ascii=False, cls=ObjectEncoder)}")
            debug_handler.flush_event_logs()
    return response


if __name__ == '__main__':
    print(ask_question("北京天气如何"))
