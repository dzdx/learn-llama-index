#! coding=utf-8
import os
from typing import cast, List

from llama_index import VectorStoreIndex, ServiceContext, QueryBundle
from llama_index.indices.postprocessor import LLMRerank
from llama_index.indices.vector_store import VectorIndexRetriever
from llama_index.query_engine import ComposableGraphQueryEngine
from llama_index.response_synthesizers import TreeSummarize
from llama_index.schema import TextNode, NodeWithScore, BaseNode

from build.download import download
from build.index import download_and_build_index, data_dir, index_dir, build_all, build_nodes
from common.llm import create_llm
from common.prompt import CH_TREE_SUMMARIZE_PROMPT
from query import load_index, DocumentQueryEngineFactory, create_response_synthesizer, load_indices, \
    create_compose_query_engine
from retrievers import MultiRetriever
from route import EchoNameEngine, create_route_query_engine, Chatter

test_llm = create_llm()


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


def test_response_synthesizer():
    service_context = ServiceContext.from_defaults(llm=test_llm)
    synthesizer = create_response_synthesizer(service_context)
    assert id(synthesizer.service_context) == id(service_context)
    assert isinstance(synthesizer, TreeSummarize)
    summarize = cast(TreeSummarize, synthesizer)
    assert summarize._summary_template == CH_TREE_SUMMARIZE_PROMPT
    response = synthesizer.synthesize("北京气候如何", [
        NodeWithScore(node=TextNode(text="2011年11月-12月，北京市遭遇了持续近一个月的烟霾天气")),
        NodeWithScore(
            node=TextNode(
                text="北京市地处暖温带半濕潤地区，气候属于暖温带半湿润大陆性季风气候，平原地区平均年降水量约600毫米")),
    ])
    print(response.response)
    assert '600' in response.response


def test_retriever():
    title = "北京市"
    indices = load_index(title)
    factory = DocumentQueryEngineFactory(indices)
    retrievers = factory.create_retrievers()
    assert len(retrievers) == len(indices)
    vector_retrievers = [r for r in retrievers if isinstance(r, VectorIndexRetriever)]
    assert len(vector_retrievers) > 0
    retriever = vector_retrievers[0]
    nodes = retriever.retrieve("北京气候如何")
    assert len(nodes) == retriever.similarity_top_k
    assert any(n for n in nodes if '平原地区平均年降水量约600毫米' in n.text)


def test_query_engine():
    title = "北京市"
    indices = load_index(title)
    factory = DocumentQueryEngineFactory(indices)
    service_context = ServiceContext.from_defaults(llm=create_llm())
    query_engine = factory.create_query_engine(service_context)
    assert isinstance(query_engine._retriever, MultiRetriever)
    multi_retriever = cast(MultiRetriever, query_engine._retriever)
    assert len(multi_retriever._retrievers) > 0
    assert any(p for p in query_engine._node_postprocessors if isinstance(p, LLMRerank))
    query = "北京气候如何"
    nodes = query_engine.retrieve(QueryBundle(query_str=query))
    assert any(n for n in nodes if '平原地区平均年降水量约600毫米' in n.text)
    print(query_engine.query(query))


def test_compose_query_engine():
    city_indices = load_indices()
    service_context = ServiceContext.from_defaults(llm=create_llm())
    query_engine = create_compose_query_engine(city_indices, service_context)
    compose_engine = cast(ComposableGraphQueryEngine, query_engine)
    assert len(compose_engine._custom_query_engines) == len(city_indices)
    beijing_response = compose_engine.query(QueryBundle(query_str="北京气候如何"))
    hangzhou_response = compose_engine.query(QueryBundle(query_str="杭州气候如何"))
    wuxi_response = compose_engine.query(QueryBundle(query_str="无锡气候如何"))
    print(beijing_response)
    print(hangzhou_response)
    print(wuxi_response)


def test_simple_route():
    hong_engine = EchoNameEngine("小红")
    hei_engine = EchoNameEngine("小黑")
    route_query_engine = create_route_query_engine([hong_engine, hei_engine],
                                                   ["可以和小红说话",
                                                    "可以和小黑说话"])
    assert route_query_engine.query("小黑你好").response == '我是小黑'
    assert route_query_engine.query("早上好小红").response == '我是小红'


def test_route_query_engine():
    chatter = Chatter()
    chatter.chat("你好呀")
    chatter.chat("北京气候如何")
    chatter.chat("杭州在中国什么位置")
