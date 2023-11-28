import os

from llama_index import VectorStoreIndex, StorageContext, \
    load_indices_from_storage, ServiceContext

from common.config import index_dir
from common.llm import create_llm
from common.utils import find_typed

title = "北京市"
storage_context = StorageContext.from_defaults(persist_dir=os.path.join(index_dir, title))
service_context = ServiceContext.from_defaults(llm=create_llm())
indices = load_indices_from_storage(
    storage_context=storage_context,
    service_context=service_context
)

vector_index = find_typed(indices, VectorStoreIndex)
retriever = vector_index.as_retriever()
print(retriever.retrieve("北京气候如何"))
