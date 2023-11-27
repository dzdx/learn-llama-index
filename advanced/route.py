import json

import llama_index.query_engine
from llama_index import ServiceContext, QueryBundle
from llama_index.callbacks import CBEventType, LlamaDebugHandler, CallbackManager
from llama_index.indices.query.base import BaseQueryEngine
from llama_index.llms import OpenAI
from llama_index.llms.base import LLM
from llama_index.prompts.mixin import PromptMixinType
from llama_index.response.schema import RESPONSE_TYPE, Response
from llama_index.selectors import LLMSingleSelector
from llama_index.tools import QueryEngineTool

from advanced.index import load_indices
from common.config import DEBUG, LLM_CACHE_ENABLED, OPENAI_API_KEY
from common.llm import llm_predict, CachedLLM
from common.utils import ObjectEncoder
from prompt import CH_SINGLE_SELECT_PROMPT_TMPL
from query import create_compose_query_engine


class SimpleQueryEngine(BaseQueryEngine):

    def __init__(self, llm: LLM, callback_manager: CallbackManager):
        self.llm = llm
        super().__init__(callback_manager=callback_manager)

    def _get_prompt_modules(self) -> PromptMixinType:
        return {}

    def _query(self, query_bundle: QueryBundle) -> RESPONSE_TYPE:
        return Response(llm_predict(self.llm, query_bundle.query_str))

    async def _aquery(self, query_bundle: QueryBundle) -> RESPONSE_TYPE:
        pass


class QueryEngine:
    def __init__(self, query_engine: BaseQueryEngine, debug_handler: LlamaDebugHandler):
        self.query_engine = query_engine
        self.debug_handler = debug_handler

    def query(self, input_text: str) -> str:
        return self.query_engine.query(input_text).response

    def print_and_flush_debug_info(self):
        if self.debug_handler:
            for event in self.debug_handler.get_events():
                if event.event_type in (CBEventType.LLM, CBEventType.RETRIEVE):
                    print(
                        f"[DebugInfo] event_type={event.event_type}, content={json.dumps(event.payload, ensure_ascii=False, cls=ObjectEncoder)}")
            self.debug_handler.flush_event_logs()


class Chatter:

    def _create_llm(self, cb_manager: CallbackManager):
        _llm_gpt3 = OpenAI(temperature=0, model="gpt-3.5-turbo", callback_manager=cb_manager, api_key=OPENAI_API_KEY)
        llm = CachedLLM(_llm_gpt3,
                        '.llm_cache',
                        request_timeout=15,
                        enable_cache=LLM_CACHE_ENABLED)
        return llm

    def create_query_engine(self) -> QueryEngine:
        if DEBUG:
            debug_handler = LlamaDebugHandler()
            cb_manager = CallbackManager([debug_handler])
        else:
            debug_handler = None
            cb_manager = CallbackManager()
        llm = self._create_llm(cb_manager)
        service_context = ServiceContext.from_defaults(
            llm=llm,
            callback_manager=cb_manager
        )
        city_indices = load_indices(service_context)
        index_query_engine = create_compose_query_engine(city_indices, service_context)

        query_tool_city = QueryEngineTool.from_defaults(
            query_engine=index_query_engine,
            description=(
                f"提供 {', '.join(city_indices.keys())} 这几个城市的相关信息"
            ),
        )

        query_tool_simple = QueryEngineTool.from_defaults(
            query_engine=SimpleQueryEngine(llm=llm, callback_manager=cb_manager),
            description=(
                f"提供其他所有信息"
            ),
        )
        route_query_engine = llama_index.query_engine.RouterQueryEngine(
            selector=LLMSingleSelector.from_defaults(service_context=service_context,
                                                     prompt_template_str=CH_SINGLE_SELECT_PROMPT_TMPL),
            service_context=service_context,
            query_engine_tools=[
                query_tool_city,
                query_tool_simple,
            ]
        )
        return QueryEngine(route_query_engine, debug_handler)


def route_query(query):
    query_engine = Chatter().create_query_engine()
    response = query_engine.query(query)
    query_engine.print_and_flush_debug_info()
    return response
