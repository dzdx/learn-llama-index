import json

import llama_index.query_engine
from llama_index.callbacks import CBEventType
from llama_index.query_engine import CustomQueryEngine
from llama_index.query_engine.custom import STR_OR_RESPONSE_TYPE
from llama_index.selectors import LLMSingleSelector
from llama_index.tools import QueryEngineTool

from common.llm import llm_predict
from prompt import CH_SINGLE_SELECT_PROMPT_TMPL
from query import query_engine, city_indices, service_context
from common.utils import ObjectEncoder
from common.debug import debug_handler, cb_manager

query_tool_city = QueryEngineTool.from_defaults(
    query_engine=query_engine,
    description=(
        f"提供 {', '.join(city_indices.keys())} 这几个城市的相关信息"
    ),
)


class SimpleQueryEngine(CustomQueryEngine):
    def custom_query(self, query_str: str) -> STR_OR_RESPONSE_TYPE:
        return llm_predict(query_str)


query_tool_simple = QueryEngineTool.from_defaults(
    query_engine=SimpleQueryEngine(callback_manager=cb_manager),
    description=(
        f"提供其他所有信息"
    ),
)

tools = [query_tool_city, query_tool_simple]

query_engine = llama_index.query_engine.RouterQueryEngine(
    selector=LLMSingleSelector.from_defaults(service_context=service_context,
                                             prompt_template_str=CH_SINGLE_SELECT_PROMPT_TMPL),
    service_context=service_context,
    query_engine_tools=[
        query_tool_city,
        query_tool_simple,
    ]
)


def route_query(query):
    response = query_engine.query(query)
    if debug_handler:
        for event in debug_handler.get_events():
            if event.event_type in (CBEventType.LLM, CBEventType.RETRIEVE):
                print(
                    f"[DebugInfo] event_type={event.event_type}, content={json.dumps(event.payload, ensure_ascii=False, cls=ObjectEncoder)}")
            debug_handler.flush_event_logs()
    return response
