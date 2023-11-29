import json
from typing import Dict, List

import llama_index.query_engine
from llama_index import ServiceContext, QueryBundle
from llama_index.callbacks import CBEventType, LlamaDebugHandler, CallbackManager
from llama_index.indices.base import BaseIndex
from llama_index.indices.query.base import BaseQueryEngine
from llama_index.llms.base import LLM
from llama_index.prompts.mixin import PromptMixinType
from llama_index.response.schema import RESPONSE_TYPE, Response
from llama_index.selectors import LLMSingleSelector
from llama_index.tools import QueryEngineTool

from common.config import DEBUG, LLM_CACHE_ENABLED
from common.llm import llm_predict, create_llm
from common.prompt import CH_SINGLE_SELECT_PROMPT_TMPL
from common.utils import ObjectEncoder
from query_todo.query_engine import load_indices
from query_todo.compose import create_compose_query_engine


class EchoNameEngine(BaseQueryEngine):
    def __init__(self, name: str, callback_manager: CallbackManager = None):
        self.name = name
        super().__init__(callback_manager)

    async def _aquery(self, query_bundle: QueryBundle) -> RESPONSE_TYPE:
        pass

    def _get_prompt_modules(self) -> PromptMixinType:
        return {}

    def _query(self, query_bundle: QueryBundle) -> RESPONSE_TYPE:
        return Response(f"我是{self.name}")


class LlmQueryEngine(BaseQueryEngine):

    def __init__(self, llm: LLM, callback_manager: CallbackManager):
        self.llm = llm
        super().__init__(callback_manager=callback_manager)

    def _get_prompt_modules(self) -> PromptMixinType:
        return {}

    def _query(self, query_bundle: QueryBundle) -> RESPONSE_TYPE:
        return Response(llm_predict(self.llm, query_bundle.query_str))

    async def _aquery(self, query_bundle: QueryBundle) -> RESPONSE_TYPE:
        pass


def create_route_query_engine(query_engines: List[BaseQueryEngine], descriptions: List[str],
                              service_context: ServiceContext = None):
    assert len(query_engines) == len(descriptions)
    # TODO
    # 根据传入的多个query_engines和descriptions创建 RouteQueryEngine，实现query engine 的路由
    # https://docs.llamaindex.ai/en/stable/module_guides/querying/router/root.html#using-as-a-query-engine
    raise NotImplementedError


class Chatter:

    def __init__(self):
        if DEBUG:
            debug_handler = LlamaDebugHandler()
            cb_manager = CallbackManager([debug_handler])
        else:
            debug_handler = None
            cb_manager = CallbackManager()
        llm = create_llm(cb_manager, LLM_CACHE_ENABLED)
        service_context = ServiceContext.from_defaults(
            llm=llm,
            callback_manager=cb_manager
        )
        self.cb_manager = cb_manager
        self.city_indices: Dict[str, List[BaseIndex]] = load_indices(service_context)
        self.service_context = service_context
        self.llm = llm
        self.debug_handler = debug_handler
        self.query_engine = self.create_query_engine()

    def create_query_engine(self):
        index_query_engine = create_compose_query_engine(self.city_indices, self.service_context)
        index_summary = f"提供 {', '.join(self.city_indices.keys())} 这几个城市的相关信息"
        llm_query_engine = LlmQueryEngine(llm=self.llm, callback_manager=self.cb_manager)
        llm_summary = f"提供其他所有信息"
        # 实现意图识别，把不同的query路由到不同的query_engine上，实现聊天和城市信息查询两个功能的分流
        # https://docs.llamaindex.ai/en/stable/module_guides/querying/router/root.html#using-as-a-query-engine
        raise NotImplementedError

    def _print_and_flush_debug_info(self):
        if self.debug_handler:
            for event in self.debug_handler.get_events():
                if event.event_type in (CBEventType.LLM, CBEventType.RETRIEVE):
                    print(
                        f"[DebugInfo] event_type={event.event_type}, content={json.dumps(event.payload, ensure_ascii=False, cls=ObjectEncoder)}")
            self.debug_handler.flush_event_logs()

    def chat(self, query):
        response = self.query_engine.query(query)
        self._print_and_flush_debug_info()
        return response
