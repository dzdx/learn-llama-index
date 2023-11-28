#! coding=utf-8
import os
from typing import cast

from llama_index import VectorStoreIndex, ServiceContext
from llama_index.indices.vector_store import VectorIndexRetriever
from llama_index.response_synthesizers import TreeSummarize

from build.download import download
from build.index import download_and_build_index, data_dir, index_dir, build_all, build_nodes
from common.prompt import CH_TREE_SUMMARIZE_PROMPT
from query import load_index, DocumentQueryEngineFactory, create_response_synthesizer


def test_build_nodes():
    title = '北京市'
    data_file = download(title, data_dir)
    assert len(build_nodes(data_file)) > 10


def test_build_index():
    title = '北京市'
    download_and_build_index(title, data_dir, index_dir)
    index_file = os.path.join(index_dir, title)
    assert os.path.exists(os.path.join(data_dir, title))
    assert os.path.exists(index_file) and os.path.isdir(index_file)
    assert any(idx for idx in load_index(title) if isinstance(idx, VectorStoreIndex))


def test_build_all_index():
    build_all()
    titles = ['北京市', '上海市', '深圳市', '杭州市', '南京市']
    all_index = os.listdir(index_dir)
    for title in titles:
        assert title in all_index


def test_create_retrievers():
    title = "北京市"
    indices = load_index(title)
    factory = DocumentQueryEngineFactory(indices)
    retrievers = factory.create_retrievers()
    assert len(retrievers) == len(indices)
    vector_retrievers = [r for r in retrievers if isinstance(r, VectorIndexRetriever)]
    assert len(vector_retrievers) > 0
    assert vector_retrievers[0].similarity_top_k == 8


def test_response_synthesizer():
    service_context = ServiceContext.from_defaults()
    synthesizer = create_response_synthesizer(service_context)
    assert id(synthesizer.service_context) == id(service_context)
    assert isinstance(synthesizer, TreeSummarize)
    summarize = cast(TreeSummarize, synthesizer)
    assert summarize._summary_template == CH_TREE_SUMMARIZE_PROMPT


def test_create_query_engine():
    pass


def test_compose_query_engine():
    pass


def test_route_query_engine():
    pass
