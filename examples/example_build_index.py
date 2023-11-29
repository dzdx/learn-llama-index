import os

from llama_index import SimpleDirectoryReader, VectorStoreIndex, TreeIndex, StorageContext, ServiceContext
from llama_index.node_parser import SimpleNodeParser
from llama_index.text_splitter import SentenceSplitter

from import_route import download
from common.config import data_dir, index_dir
from common.llm import create_llm
from common.prompt import CH_SUMMARY_PROMPT

titles = ["北京市", "上海市"]
for title in titles:
    data_file = download(title, data_dir)
    index_file = os.path.join(index_dir, title)
    documents = SimpleDirectoryReader(input_files=[data_file]).load_data()
    node_parser = SimpleNodeParser.from_defaults(text_splitter=SentenceSplitter(
        chunk_size=1024,
        chunk_overlap=200,
    ))
    service_context = ServiceContext.from_defaults(llm=create_llm(timeout=60))
    nodes = node_parser.get_nodes_from_documents(documents)
    storage_context = StorageContext.from_defaults()
    vector_index = VectorStoreIndex(nodes,
                                    service_context=service_context,
                                    storage_context=storage_context,
                                    show_progress=True)
    tree_index = TreeIndex(nodes, num_children=8,
                           service_context=service_context,
                           storage_context=storage_context,
                           summary_template=CH_SUMMARY_PROMPT,
                           show_progress=True)
    storage_context.persist(persist_dir=index_file)
