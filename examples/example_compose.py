from llama_index import ServiceContext, ComposableGraph, TreeIndex, VectorStoreIndex

from common.llm import create_llm
from common.prompt import CH_QUERY_PROMPT
from common.utils import find_typed
from query.query_engine import create_response_synthesizer, load_index

service_context = ServiceContext.from_defaults(llm=create_llm())
titles = ["北京市", "上海市"]
summaries = []
indices = []
for city in titles:
    city_indices = load_index(city, service_context)
    summary = f"""
        此内容包含关于{city}的维基百科文章。
        如果您需要查找有关{city}的具体事实，请使用此索引。"
        如果您想分析多个城市，请不要使用此索引。
        """
    index = find_typed(city_indices, VectorStoreIndex)
    indices.append(index)
    summaries.append(summary)
graph = ComposableGraph.from_indices(
    TreeIndex,
    indices,
    summaries)
query_engine = graph.as_query_engine(
    response_synthesizer=create_response_synthesizer(service_context=service_context),
    service_context=service_context,
    query_template=CH_QUERY_PROMPT,
)
print(query_engine.query("北京气候如何"))
print(query_engine.query("深圳在中国什么位置"))
