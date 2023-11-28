import hashlib
import json
import os.path
import pickle
from dataclasses import dataclass
from functools import wraps
from typing import Any, Dict, Optional, Sequence, Tuple

from llama_index.bridge.pydantic import Field
from llama_index.callbacks import CallbackManager
from llama_index.llms import (
    ChatMessage,
    ChatResponse,
    ChatResponseAsyncGen,
    ChatResponseGen,
    CompletionResponse,
    CompletionResponseAsyncGen,
    CompletionResponseGen,
    LLMMetadata, OpenAI,
)
from llama_index.llms.base import LLM

from common.config import OPENAI_API_KEY, ROOT_PATH
from common.utils import ObjectEncoder


@dataclass
class CacheRequest:
    func: str
    args: Optional[Tuple]
    kwargs: Optional[Dict]

    def __init__(self, func, *args, **kwargs):
        self.func = func
        self.args = args
        self.kwargs = kwargs

    def dump(self) -> bytes:
        return json.dumps(self, cls=ObjectEncoder, ensure_ascii=False).encode('utf-8')


@dataclass
class CacheItem:
    request: CacheRequest
    response: Optional[object]


def cached_call(method):
    @wraps(method)
    def wrapper(self, *args, **kwargs):
        cache_req = CacheRequest(method.__name__, *args, **kwargs)
        if self.enable_cache:
            cache_item = self._get_cache(cache_req)
            if cache_item.response:
                return cache_item.response
        response = method(self, *args, **kwargs)
        self._save_cache(cache_req, response)
        return response

    return wrapper


class CachedLLM(LLM):
    llm: LLM = Field()
    root_dir: str = Field()
    request_timeout: int = Field()
    enable_cache: bool = Field()

    def __init__(self, llm: LLM, root_dir: str, request_timeout=60, enable_cache: bool = False, **data: Any):
        if not os.path.exists(root_dir):
            os.makedirs(root_dir, exist_ok=True)
        super().__init__(llm=llm, root_dir=root_dir, request_timeout=request_timeout, enable_cache=enable_cache, **data)

    @classmethod
    def class_name(cls) -> str:
        return cls.__class__.__name__

    @property
    def metadata(self) -> LLMMetadata:
        return self.llm.metadata

    def _get_cache(
            self,
            req: CacheRequest,
    ) -> CacheItem:
        req_dump = req.dump()
        md5 = hashlib.md5(req_dump).hexdigest()
        cache_path = os.path.join(self.root_dir, md5)
        if os.path.exists(cache_path):
            with open(cache_path, "rb") as f:
                cache_item: CacheItem = pickle.load(f)
                if cache_item.request.dump() == req_dump:
                    return cache_item
                else:
                    print("llm cache request hash conflict: %s != %s", req, cache_item.request)
        return CacheItem(req, None)

    def _save_cache(self, cache_request: CacheRequest, response: object):
        md5 = hashlib.md5(cache_request.dump()).hexdigest()
        cache_path = os.path.join(self.root_dir, md5)
        with open(cache_path, "wb") as f:
            pickle.dump(CacheItem(cache_request, response), f)

    @cached_call
    def chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        return self.llm.chat(messages, request_timeout=self.request_timeout, timeout=self.request_timeout, **kwargs)

    @cached_call
    def complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        return self.llm.complete(prompt, request_timeout=self.request_timeout, timeout=self.request_timeout, **kwargs)

    def stream_chat(
            self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponseGen:
        return self.llm.stream_chat(messages)

    def stream_complete(self, prompt: str, **kwargs: Any) -> CompletionResponseGen:
        return self.llm.stream_complete(prompt, **kwargs)

    async def achat(
            self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponse:
        return await self.achat(messages, **kwargs)

    async def acomplete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        return await self.acomplete(prompt, **kwargs)

    async def astream_chat(
            self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponseAsyncGen:
        return await self.astream_chat(messages, **kwargs)

    async def astream_complete(
            self, prompt: str, **kwargs: Any
    ) -> CompletionResponseAsyncGen:
        return await self.astream_complete(prompt, **kwargs)


def create_llm(callback_manager: CallbackManager = None, enable_cache: bool = True, timeout=15):
    _llm_gpt3 = OpenAI(temperature=0, model="gpt-3.5-turbo", callback_manager=callback_manager, api_key=OPENAI_API_KEY)
    return CachedLLM(_llm_gpt3,
                     os.path.join(ROOT_PATH, '.llm_cache'),
                     request_timeout=timeout,
                     enable_cache=enable_cache)


def llm_predict(llm: LLM, content: str):
    response = llm.chat([ChatMessage(
        content=content
    )])
    return response.message.content
