import os

from llama_index import VectorStoreIndex, StorageContext, \
    load_indices_from_storage, ServiceContext
from llama_index.callbacks import CallbackManager, LlamaDebugHandler

from common.config import index_dir
from common.llm import create_llm

title = "北京市"
debug_handler = LlamaDebugHandler()
cb_manager = CallbackManager([debug_handler])
storage_context = StorageContext.from_defaults(persist_dir=os.path.join(index_dir, title))
service_context = ServiceContext.from_defaults(llm=create_llm(), callback_manager=cb_manager)
indices = load_indices_from_storage(
    storage_context=storage_context,
    service_context=service_context
)

vector_index = [idx for idx in indices if isinstance(idx, VectorStoreIndex)][0]
retriever = vector_index.as_retriever()
print(retriever.retrieve("北京气候如何"))
for event in debug_handler.get_events():
    print(event)
