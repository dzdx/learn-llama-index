import os

from llama_index import TreeIndex
from llama_index import VectorStoreIndex, StorageContext, \
    load_indices_from_storage, ServiceContext
from llama_index.indices.tree.base import TreeRetrieverMode

from common.config import index_dir
from common.llm import create_llm
from import_route import MultiRetriever

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
print(multi_retriever.retrieve("北京气候如何"))
