import os

from llama_index import VectorStoreIndex, ServiceContext, StorageContext, \
    load_indices_from_storage, TreeIndex, QueryBundle
from llama_index.indices.postprocessor import LLMRerank
from llama_index.indices.tree.base import TreeRetrieverMode
from llama_index.query_engine import RetrieverQueryEngine

from common.config import index_dir
from common.llm import create_llm
from common.prompt import CH_CHOICE_SELECT_PROMPT
from query.query_engine import create_response_synthesizer
from query.retrievers import MultiRetriever

title = "北京市"
storage_context = StorageContext.from_defaults(persist_dir=os.path.join(index_dir, title))
service_context = ServiceContext.from_defaults(llm=create_llm())
indices = load_indices_from_storage(
    storage_context=storage_context,
    service_context=service_context
)
retrievers = []
for index in indices:
    if isinstance(index, VectorStoreIndex):
        retrievers.append(index.as_retriever(similarity_top_k=8))
    if isinstance(index, TreeIndex):
        retrievers.append(index.as_retriever(retriever_mode=TreeRetrieverMode.SELECT_LEAF_EMBEDDING))

multi_retriever = MultiRetriever(retrievers)

node_postprocessors = [
    LLMRerank(top_n=4, choice_batch_size=2, choice_select_prompt=CH_CHOICE_SELECT_PROMPT,
              service_context=service_context)
]
query_engine = RetrieverQueryEngine.from_args(
    multi_retriever,
    node_postprocessors=node_postprocessors,
    service_context=service_context,
    response_synthesizer=create_response_synthesizer(service_context)
)
print(query_engine.retrieve(QueryBundle("北京气候如何")))
