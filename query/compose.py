from typing import Dict, List

from llama_index import ServiceContext, ComposableGraph, TreeIndex
from llama_index.indices.base import BaseIndex
from llama_index.indices.query.base import BaseQueryEngine

from common.prompt import CH_QUERY_PROMPT
from query.query_engine import DocumentQueryEngineFactory, create_response_synthesizer


def create_compose_query_engine(city_indices: Dict[str, List[BaseIndex]],
                                service_context: ServiceContext) -> BaseQueryEngine:
    query_engines = []
    for city, indices in city_indices.items():
        summary = f"""
            此内容包含关于{city}的维基百科文章。
            如果您需要查找有关{city}的具体事实，请使用此索引。"
            如果您想分析多个城市，请不要使用此索引。
            """
        query_engines.append(DocumentQueryEngineFactory(
            indices=indices,
            summary=summary
        ))

    # 为每个城市的query engine指定一个summary，ComposableGraph会使用大模型来判断问题需要使用哪一个QueryEngine
    graph = ComposableGraph.from_indices(
        TreeIndex,
        [e.first_index() for e in query_engines],
        [e.summary for e in query_engines],
        service_context=service_context,
    )
    return graph.as_query_engine(
        response_synthesizer=create_response_synthesizer(service_context=service_context),
        custom_query_engines={e.first_index().index_id: e.create_query_engine(service_context) for e in query_engines},
        service_context=service_context,
        query_template=CH_QUERY_PROMPT,
    )
