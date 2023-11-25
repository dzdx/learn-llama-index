# Demo

1. 首先要准备一个OPENAI_API_KEY 来访问大模型

```bash
export OPENAI_API_KEY=XXXXX
```

2. 安装依赖

```
virtualenv venv
source venv/bin/activate
pip install requirements.txt
```

3. 准备一部分数据集，作为我们可供查询的数据，这边从维基百科下载了中国几个城市的介绍数据，执行下载操作 `python download.py`

```python
from pathlib import Path
import requests

wiki_titles = ["北京市", "上海市", "杭州市", "广州市", "南京市"]

for title in wiki_titles:
    response = requests.get(
        'https://zh.wikipedia.org/w/api.php',
        params={
            'action': 'query',
            'format': 'json',
            'titles': title,
            'prop': 'extracts',
            'explaintext': True,
        }
    ).json()
    page = next(iter(response['query']['pages'].values()))
    wiki_text = page['extract']

    data_path = Path('data')
    if not data_path.exists():
        Path.mkdir(data_path)

    with open(data_path / f"{title}.txt", 'w') as fp:
        fp.write(wiki_text)
```

4. 首先我们对索引进行构建，按照每个城市一套所有的方式构建向量索引，这边简单期间直接使用本地文件作为索引

```python
data_dir = os.path.join(ROOT_PATH, 'data')
index_dir = os.path.join(ROOT_PATH, 'index')


def load_index(index_dir: str, service_context: ServiceContext = None) -> List[BaseIndex]:
    storage_context = StorageContext.from_defaults(persist_dir=index_dir)
    return load_indices_from_storage(
        storage_context=storage_context, service_context=service_context
    )


def build_index(index_dir: str, documents: List[Document], service_context: ServiceContext = None) -> List[BaseIndex]:
    storage_context = StorageContext.from_defaults()
    vector_index = VectorStoreIndex.from_documents(
        documents,
        service_context=service_context,
        storage_context=storage_context,
    )
    storage_context.persist(persist_dir=index_dir)
    return [vector_index]


def load_or_build_city_index(service_context: ServiceContext, file) -> List[BaseIndex]:
    index_file = os.path.join(index_dir, file)
    if os.path.exists(index_file):
        indices = load_index(index_file, service_context)
    else:
        documents = SimpleDirectoryReader(input_files=[os.path.join(data_dir, file)]).load_data()
        for doc in documents:
            doc.excluded_llm_metadata_keys.append("file_path")
            doc.excluded_embed_metadata_keys.append("file_path")
        indices = build_index(index_file, documents)
    return indices


def load_or_build_cities_indices(service_context: ServiceContext) -> Dict[str, List[BaseIndex]]:
    city_indices: Dict[str, List[BaseIndex]] = {}
    for file in os.listdir(data_dir):
        basename = os.path.splitext(file)[0]
        city_indices[basename] = load_or_build_city_index(service_context, file)
    return city_indices

```

5. 基于index构建一个多城市组合的查询引擎，这边会提供summary，把问题路由到对应城市的index进行查询

```python


city_indices = load_or_build_cities_indices(service_context)


@dataclass
class DocQueryEngine:
    indices: List[BaseIndex]
    summary: str

    def first_index(self):
        return self.indices[0]

    def create_query_engine(self):
        sub_retrievers = []
        for index in self.indices:
            sub_retrievers.append(index.as_retriever())
        retriever = CustomRetriever(sub_retrievers)
        node_postprocessors = [
            LLMRerank(top_n=3, choice_select_prompt=CH_CHOICE_SELECT_PROMPT, service_context=service_context)
        ]
        return RetrieverQueryEngine.from_args(
            retriever, node_postprocessors=node_postprocessors,
            service_context=service_context,
            text_qa_template=Prompt(
                CH_TEXT_QA_PROMPT_TMPL, prompt_type=PromptType.QUESTION_ANSWER
            ))


query_engines = []
for city, indices in city_indices.items():
    summary = f"""
        此内容包含关于{city}的维基百科文章。
        如果您需要查找有关{city}的具体事实，请使用此索引。"
        如果您想分析多个城市，请不要使用此索引。
        """
    query_engines.append(DocQueryEngine(
        indices=indices,
        summary=summary
    ))

graph = ComposableGraph.from_indices(
    TreeIndex,
    [e.first_index() for e in query_engines],
    [e.summary for e in query_engines],
    service_context=service_context,

```

6. 对于返回了超过LLM context limit的多个节点，我们采用 TREE_SUMMARIZE 的方式逐步总结，得到最终答案

```python


response_synthesizer = get_response_synthesizer(
    response_mode=ResponseMode.TREE_SUMMARIZE,
    text_qa_template=Prompt(
        CH_TEXT_QA_PROMPT_TMPL, prompt_type=PromptType.QUESTION_ANSWER
    ),
    summary_template=CH_TREE_SUMMARIZE_PROMPT,
    service_context=service_context,
)

query_engine = graph.as_query_engine(
    response_synthesizer=response_synthesizer,
    custom_query_engines={e.first_index().index_id: e.create_query_engine() for e in query_engines},
    service_context=service_context,
    query_template=CH_QUERY_PROMPT,
)

```


7. 如何我们还希望提供回答除了5个城市相关信息以外的问题的功能，可以使用RouteQueryEngine进行分流

```python

query_tool_city = QueryEngineTool.from_defaults(
    query_engine=query_engine,
    description=(
        f"提供 {', '.join(city_indices.keys())} 这几个城市的相关信息"
    ),
)


class SimpleQueryEngine(CustomQueryEngine):
    def custom_query(self, query_str: str) -> STR_OR_RESPONSE_TYPE:
        return llm_predict(query_str)


query_tool_simple = QueryEngineTool.from_defaults(
    query_engine=SimpleQueryEngine(callback_manager=cb_manager),
    description=(
        f"提供其他所有信息"
    ),
)

tools = [query_tool_city, query_tool_simple]

query_engine = llama_index.query_engine.RouterQueryEngine(
    selector=LLMSingleSelector.from_defaults(service_context=service_context,
                                             prompt_template_str=CH_SINGLE_SELECT_PROMPT_TMPL),
    service_context=service_context,
    query_engine_tools=[
        query_tool_city,
        query_tool_simple,
    ]
)


def route_query(query):
    response = query_engine.query(query)
    if debug_handler:
        for event in debug_handler.get_events():
            if event.event_type in (CBEventType.LLM, CBEventType.RETRIEVE):
                print(
                    f"[DebugInfo] event_type={event.event_type}, content={json.dumps(event.payload, ensure_ascii=False, cls=ObjectEncoder)}")
            debug_handler.flush_event_logs()
    return response

```

8. 让我们测试一下我们的问答系统

```python
if __name__ == '__main__':
    for q in ["你好呀",
              "北京气候如何",
              "杭州在中国的位置"]:
        print(f"Question: {q}")
        print(f"Answer: {route_query(q)}")
```

得到结果

```
Question: 你好呀
Answer: 你好！有什么我可以帮助你的吗？
Question: 北京气候如何
Answer: 北京气候属于暖温带半湿润大陆性季风气候，四季分明。春季多风和沙尘，夏季炎热多雨，秋季晴朗干燥，冬季寒冷且大风猛烈。其中春季和秋季很短，大概一个月出头左右；而夏季和冬季则很长，各接近五个月。年平均气温约为12.9°C，最冷月（1月）平均气温为-3.1°C，最热月（7月）平均气温为26.7°C。
Question: 杭州在中国的位置
Answer: 杭州位于中国的浙江省北部。
```
9. 如果希望观测整个查询链路的过程，可以提供tracing功能

``` python
  
DEBUG = True
  
if DEBUG:  
    debug_handler = LlamaDebugHandler()  
    cb_manager = CallbackManager([debug_handler])  
else:  
    debug_handler = None  
    cb_manager = CallbackManager()
```

让我们跑一看看完整的Debug log
```python
route_query("北京气候如何")
```

从下面的日志可以清晰的看到llama index在回答我们的问题经历的步骤，每个召回的chunk，LLM的prompt，每一步的耗时

```angular2html
Trace: query
    |_query ->  15.793248 seconds
      |_templating ->  1.5e-05 seconds
      |_llm ->  2.585527 seconds
      |_query ->  13.206116 seconds
        |_retrieve ->  2.10315 seconds
          |_templating ->  1.9e-05 seconds
          |_llm ->  2.099882 seconds
        |_query ->  6.699856 seconds
          |_retrieve ->  1.78654 seconds
            |_embedding ->  0.626603 seconds
            |_templating ->  4.5e-05 seconds
            |_llm ->  1.150914 seconds
          |_synthesize ->  4.913122 seconds
            |_templating ->  1.6e-05 seconds
            |_llm ->  4.909513 seconds
        |_synthesize ->  4.402676 seconds
          |_templating ->  1.3e-05 seconds
          |_llm ->  4.400681 seconds
**********
[DebugInfo] event_type=llm, content={"messages": [{"role": "user", "content": "以下给出了一些选项。它们被编号为从1到2的列表，列表中的每一项对应一个摘要。\n---------------------\n(1) 提供 广州市, 杭州市, 上海市, 北京市, 南京市 这几个城市的相关信息\n\n(2) 提供其他所有信息\n---------------------\n只使用上述选项，而不使用先前的知识，返回与问题：'北京气候如何'最相关的选项\n\n\nThe output should be ONLY JSON formatted as a JSON instance.\n\nHere is an example:\n[\n    {{\n        choice: 1,\n        reason: \"<insert reason for choice>\"\n    }},\n    ...\n]\n", "additional_kwargs": {}}], "additional_kwargs": {"request_timeout": 15, "timeout": 15}, "serialized": {"model": "gpt-3.5-turbo", "temperature": 0.0, "max_tokens": null, "additional_kwargs": {}, "max_retries": 10, "api_type": "open_ai", "api_base": "https://api.openai.com/v1", "api_version": "", "class_name": "openai_llm"}}
[DebugInfo] event_type=llm, content={"messages": [{"role": "user", "content": "以下给出了一些选项。它们被编号为从1到2的列表，列表中的每一项对应一个摘要。\n---------------------\n(1) 提供 广州市, 杭州市, 上海市, 北京市, 南京市 这几个城市的相关信息\n\n(2) 提供其他所有信息\n---------------------\n只使用上述选项，而不使用先前的知识，返回与问题：'北京气候如何'最相关的选项\n\n\nThe output should be ONLY JSON formatted as a JSON instance.\n\nHere is an example:\n[\n    {{\n        choice: 1,\n        reason: \"<insert reason for choice>\"\n    }},\n    ...\n]\n", "additional_kwargs": {}}], "response": {"message": {"role": "assistant", "content": "[\n    {\n        \"choice\": 1,\n        \"reason\": \"北京是所提供城市中的一个，因此与问题最相关\"\n    }\n]", "additional_kwargs": {}}, "raw": {"id": "chatcmpl-8Oh17cxJcd6rPNCtrw227zpyCOr3w", "object": "chat.completion", "created": 1700895921, "model": "gpt-3.5-turbo-0613", "choices": [{"index": 0, "message": {"role": "assistant", "content": "[\n    {\n        \"choice\": 1,\n        \"reason\": \"北京是所提供城市中的一个，因此与问题最相关\"\n    }\n]"}, "finish_reason": "stop"}], "usage": {"prompt_tokens": 171, "completion_tokens": 36, "total_tokens": 207}}, "delta": null, "additional_kwargs": {"prompt_tokens": 171, "completion_tokens": 36, "total_tokens": 207}}}
[DebugInfo] event_type=retrieve, content={"query_str": "北京气候如何"}
[DebugInfo] event_type=llm, content={"messages": [{"role": "user", "content": "下面给出了一些选择，以编号列表的形式提供。(1 到 5), 其中列表中的每个项目对应一个摘要。\n---------------------\n(1) 此内容包含关于广州市的维基百科文章。         如果您需要查找有关广州市的具体事实，请使用此索引。\"         如果您想分析多个城市，请不要使用此索引。\n\n(2) 此内容包含关于杭州市的维基百科文章。         如果您需要查找有关杭州市的具体事实，请使用此索引。\"         如果您想分析多个城市，请不要使用此索引。\n\n(3) 此内容包含关于上海市的维基百科文章。         如果您需要查找有关上海市的具体事实，请使用此索引。\"         如果您想分析多个城市，请不要使用此索引。\n\n(4) 此内容包含关于北京市的维基百科文章。         如果您需要查找有关北京市的具体事实，请使用此索引。\"         如果您想分析多个城市，请不要使用此索引。\n\n(5) 此内容包含关于南京市的维基百科文章。         如果您需要查找有关南京市的具体事实，请使用此索引。\"         如果您想分析多个城市，请不要使用此索引。\n---------------------\n仅使用上述选项而不依赖先前知识，返回与问题 '北京气候如何' 最相关的选择。以以下格式提供答案：'ANSWER: <number>' 并解释为什么选择该摘要与问题相关。\n", "additional_kwargs": {}}], "additional_kwargs": {"request_timeout": 15, "timeout": 15}, "serialized": {"model": "gpt-3.5-turbo", "temperature": 0.0, "max_tokens": null, "additional_kwargs": {}, "max_retries": 10, "api_type": "open_ai", "api_base": "https://api.openai.com/v1", "api_version": "", "class_name": "openai_llm"}}
[DebugInfo] event_type=llm, content={"messages": [{"role": "user", "content": "下面给出了一些选择，以编号列表的形式提供。(1 到 5), 其中列表中的每个项目对应一个摘要。\n---------------------\n(1) 此内容包含关于广州市的维基百科文章。         如果您需要查找有关广州市的具体事实，请使用此索引。\"         如果您想分析多个城市，请不要使用此索引。\n\n(2) 此内容包含关于杭州市的维基百科文章。         如果您需要查找有关杭州市的具体事实，请使用此索引。\"         如果您想分析多个城市，请不要使用此索引。\n\n(3) 此内容包含关于上海市的维基百科文章。         如果您需要查找有关上海市的具体事实，请使用此索引。\"         如果您想分析多个城市，请不要使用此索引。\n\n(4) 此内容包含关于北京市的维基百科文章。         如果您需要查找有关北京市的具体事实，请使用此索引。\"         如果您想分析多个城市，请不要使用此索引。\n\n(5) 此内容包含关于南京市的维基百科文章。         如果您需要查找有关南京市的具体事实，请使用此索引。\"         如果您想分析多个城市，请不要使用此索引。\n---------------------\n仅使用上述选项而不依赖先前知识，返回与问题 '北京气候如何' 最相关的选择。以以下格式提供答案：'ANSWER: <number>' 并解释为什么选择该摘要与问题相关。\n", "additional_kwargs": {}}], "response": {"message": {"role": "assistant", "content": "ANSWER: 4\n\n我选择摘要4，因为它包含关于北京市的维基百科文章。根据问题'北京气候如何'，我可以通过查找有关北京市的具体事实来回答这个问题。", "additional_kwargs": {}}, "raw": {"id": "chatcmpl-8Oh19i4aVaSxqlEKKihsVT3YGJGc0", "object": "chat.completion", "created": 1700895923, "model": "gpt-3.5-turbo-0613", "choices": [{"index": 0, "message": {"role": "assistant", "content": "ANSWER: 4\n\n我选择摘要4，因为它包含关于北京市的维基百科文章。根据问题'北京气候如何'，我可以通过查找有关北京市的具体事实来回答这个问题。"}, "finish_reason": "stop"}], "usage": {"prompt_tokens": 440, "completion_tokens": 68, "total_tokens": 508}}, "delta": null, "additional_kwargs": {"prompt_tokens": 440, "completion_tokens": 68, "total_tokens": 508}}}
[DebugInfo] event_type=retrieve, content={"nodes": [{"node": {"id_": "db00d8d7-4fb1-46d8-8aec-945ae6c680e0", "embedding": null, "metadata": {}, "excluded_embed_metadata_keys": [], "excluded_llm_metadata_keys": [], "relationships": {"1": {"node_id": "0aeb2a04-d99b-4969-9e37-653e0fd736a9", "node_type": "3", "metadata": {}, "hash": null}}, "hash": "24ad677ecd2dab1fbc757f2280a03a72175e5021ccc16c4e4b3554a44cd64853", "text": "\n        此内容包含关于北京市的维基百科文章。\n        如果您需要查找有关北京市的具体事实，请使用此索引。\"\n        如果您想分析多个城市，请不要使用此索引。\n        ", "start_char_idx": null, "end_char_idx": null, "text_template": "{metadata_str}\n\n{content}", "metadata_template": "{key}: {value}", "metadata_seperator": "\n", "index_id": "0aeb2a04-d99b-4969-9e37-653e0fd736a9"}, "score": null}]}
[DebugInfo] event_type=retrieve, content={"query_str": "北京气候如何"}
[DebugInfo] event_type=llm, content={"messages": [{"role": "user", "content": "以下显示了一系列文档。每个文档旁边都有一个数字，以及文档的摘要。也提供了一个问题。\n请回答你认为应该参考哪些文档来回答这个问题，按照相关性的顺序，\n并给出相关性的评分。相关性评分是一个1-10的数字，根据你认为文档对问题的相关性进行评定。\n不要包含任何与问题无关的文档。\n示例格式：\nDocument 1:\n<文档1的摘要>\n\nDocument 2:\n<文档2的摘要>\n\n...\n\nDocument 10:\n<文档10的摘要>\n\n问题：<问题>\n答案：\nDoc: 9, Relevance: 7\nDoc: 3, Relevance: 4\nDoc: 7, Relevance: 3\n\n现在我们试试：\n\nDocument 1:\n=== 氣候 ===\n北京市地处暖温带半濕潤地区，气候属于暖温带半湿润大陆性季风气候，平原地区平均年降水量约600毫米，市区代表站最大年降水量为1404.6毫米，最大24小时降水量为404.2毫米，平均年最大积雪深度为7.5厘米，历史最大积雪深度33.5厘米。北京四季分明，春季多風和沙尘，夏季炎熱多雨，秋季晴朗干燥，冬季寒冷且大风猛烈。其中春季和秋季很短，大概一个月出头左右；而夏季和冬季则很长，各接近五个月。北京季风性特征明显，全年60%的降水集中在夏季的7、8月份，而其他季节空气较为干燥。年平均气温約為12.9 °C。最冷月（1月）平均气温为−3.1 °C，最热月（7月）平均气温为26.7 °C。极端最低气温为−33.2 °C（1980年1月30日佛爷顶），极端最高气温43.5 °C（1961年6月10日房山区）。市区极端最低气温为−27.4 °C（1966年2月22日），市区极端最高气温41.9 °C（1999年7月24日）。\n\n\n=== 環境問題 ===\n\n\n==== 空气污染 ====\n北京的空气污染长期为人诟病。空气的主要污染源为工业废气及汽车尾气排放。\n\nDocument 2:\n2008年北京奥运会之后，北京的空气质量并未因此而好转，2011年2月21日，美国驻华大使馆录得的北京空气污染指数已超过了正常的可测量范围，北京市气象部门亦在2月24日通报称，截止当日北京空气质量已连续三天达到重度污染。亚洲开发银行和清华大学发布《中华人民共和国国家环境分析》报告成北京市为世界十大空气污染城市之一。飞絮则是北京每年4、5月間另一特有现象，其来源以杨絮為主，也有柳絮。由於北京城市道路绿化曾以毛白楊作為主要树种，而其白絮极易随风飘扬，因此形成了北京春天漫天飞杨絮的景观，也造成了一定程度的环境污染。随着近年来北京市通過樹幹注射抑止劑、嫁接以及用其他樹種逐漸取代毛白楊等方法治理楊柳飛絮，飞絮现象已逐漸得到控制。\n\n2011年11月-12月，北京市遭遇了持续近一个月的烟霾天气，美国驻华大使馆测量的空气污染指数连续“爆表”，然而北京市环保局发布的空气污染级别却为“轻度污染”，引发公众疑虑。后有媒体爆料，当今世界上发达国家普遍开始采用的空气污染测量标准为PM2.5（直径2.5微米的空气颗粒，乃影响大气透明度的首要污染物，同时由于其粒径较小容易进入人体）。\n问题：北京气候如何\n答案：\n", "additional_kwargs": {}}], "additional_kwargs": {"request_timeout": 15, "timeout": 15}, "serialized": {"model": "gpt-3.5-turbo", "temperature": 0.0, "max_tokens": null, "additional_kwargs": {}, "max_retries": 10, "api_type": "open_ai", "api_base": "https://api.openai.com/v1", "api_version": "", "class_name": "openai_llm"}}
[DebugInfo] event_type=llm, content={"messages": [{"role": "user", "content": "以下显示了一系列文档。每个文档旁边都有一个数字，以及文档的摘要。也提供了一个问题。\n请回答你认为应该参考哪些文档来回答这个问题，按照相关性的顺序，\n并给出相关性的评分。相关性评分是一个1-10的数字，根据你认为文档对问题的相关性进行评定。\n不要包含任何与问题无关的文档。\n示例格式：\nDocument 1:\n<文档1的摘要>\n\nDocument 2:\n<文档2的摘要>\n\n...\n\nDocument 10:\n<文档10的摘要>\n\n问题：<问题>\n答案：\nDoc: 9, Relevance: 7\nDoc: 3, Relevance: 4\nDoc: 7, Relevance: 3\n\n现在我们试试：\n\nDocument 1:\n=== 氣候 ===\n北京市地处暖温带半濕潤地区，气候属于暖温带半湿润大陆性季风气候，平原地区平均年降水量约600毫米，市区代表站最大年降水量为1404.6毫米，最大24小时降水量为404.2毫米，平均年最大积雪深度为7.5厘米，历史最大积雪深度33.5厘米。北京四季分明，春季多風和沙尘，夏季炎熱多雨，秋季晴朗干燥，冬季寒冷且大风猛烈。其中春季和秋季很短，大概一个月出头左右；而夏季和冬季则很长，各接近五个月。北京季风性特征明显，全年60%的降水集中在夏季的7、8月份，而其他季节空气较为干燥。年平均气温約為12.9 °C。最冷月（1月）平均气温为−3.1 °C，最热月（7月）平均气温为26.7 °C。极端最低气温为−33.2 °C（1980年1月30日佛爷顶），极端最高气温43.5 °C（1961年6月10日房山区）。市区极端最低气温为−27.4 °C（1966年2月22日），市区极端最高气温41.9 °C（1999年7月24日）。\n\n\n=== 環境問題 ===\n\n\n==== 空气污染 ====\n北京的空气污染长期为人诟病。空气的主要污染源为工业废气及汽车尾气排放。\n\nDocument 2:\n2008年北京奥运会之后，北京的空气质量并未因此而好转，2011年2月21日，美国驻华大使馆录得的北京空气污染指数已超过了正常的可测量范围，北京市气象部门亦在2月24日通报称，截止当日北京空气质量已连续三天达到重度污染。亚洲开发银行和清华大学发布《中华人民共和国国家环境分析》报告成北京市为世界十大空气污染城市之一。飞絮则是北京每年4、5月間另一特有现象，其来源以杨絮為主，也有柳絮。由於北京城市道路绿化曾以毛白楊作為主要树种，而其白絮极易随风飘扬，因此形成了北京春天漫天飞杨絮的景观，也造成了一定程度的环境污染。随着近年来北京市通過樹幹注射抑止劑、嫁接以及用其他樹種逐漸取代毛白楊等方法治理楊柳飛絮，飞絮现象已逐漸得到控制。\n\n2011年11月-12月，北京市遭遇了持续近一个月的烟霾天气，美国驻华大使馆测量的空气污染指数连续“爆表”，然而北京市环保局发布的空气污染级别却为“轻度污染”，引发公众疑虑。后有媒体爆料，当今世界上发达国家普遍开始采用的空气污染测量标准为PM2.5（直径2.5微米的空气颗粒，乃影响大气透明度的首要污染物，同时由于其粒径较小容易进入人体）。\n问题：北京气候如何\n答案：\n", "additional_kwargs": {}}], "response": {"message": {"role": "assistant", "content": "Doc: 1, Relevance: 9\nDoc: 2, Relevance: 8", "additional_kwargs": {}}, "raw": {"id": "chatcmpl-8Oh1Bd8EVx0UiUIRx5vsTm2J8pgbb", "object": "chat.completion", "created": 1700895925, "model": "gpt-3.5-turbo-0613", "choices": [{"index": 0, "message": {"role": "assistant", "content": "Doc: 1, Relevance: 9\nDoc: 2, Relevance: 8"}, "finish_reason": "stop"}], "usage": {"prompt_tokens": 1413, "completion_tokens": 21, "total_tokens": 1434}}, "delta": null, "additional_kwargs": {"prompt_tokens": 1413, "completion_tokens": 21, "total_tokens": 1434}}}
[DebugInfo] event_type=retrieve, content={"nodes": [{"node": {"id_": "fea8a8ef-b26e-4f71-be43-376c52f6d24d", "embedding": null, "metadata": {"file_path": "/Users/dzdx/Documents/learn-llama-index/data/北京市.txt", "creation_date": "2023-11-24", "last_modified_date": "2023-11-24", "last_accessed_date": "2023-11-24"}, "excluded_embed_metadata_keys": ["creation_date", "last_modified_date", "last_accessed_date", "file_path"], "excluded_llm_metadata_keys": ["creation_date", "last_modified_date", "last_accessed_date", "file_path"], "relationships": {"1": {"node_id": "f1a4410d-0926-41d6-8844-c126071c8dbc", "node_type": "4", "metadata": {"file_path": "/Users/dzdx/Documents/learn-llama-index/data/北京市.txt", "creation_date": "2023-11-24", "last_modified_date": "2023-11-24", "last_accessed_date": "2023-11-24"}, "hash": "27b2f2b0146b996e14fae6d6f4670c7b6030f237b7daab9dff11740c5b898724"}, "2": {"node_id": "7d8b09ca-7109-4e43-b8f1-3ccab783b078", "node_type": "1", "metadata": {"file_path": "/Users/dzdx/Documents/learn-llama-index/data/北京市.txt", "creation_date": "2023-11-24", "last_modified_date": "2023-11-24", "last_accessed_date": "2023-11-24"}, "hash": "8436c8065e75a5114ed1718842e8fdf70dfbca1f6447813c4f7d1d92d3640707"}, "3": {"node_id": "ddd0587f-2c47-41d1-a6a3-bccba0844a16", "node_type": "1", "metadata": {"file_path": "/Users/dzdx/Documents/learn-llama-index/data/北京市.txt", "creation_date": "2023-11-24", "last_modified_date": "2023-11-24", "last_accessed_date": "2023-11-24"}, "hash": "fc1544a36ec674717680120e8ba773a6223dbf7b4dfade9f4d4e7cff3f69ef5c"}}, "hash": "7edd377fa4ebd8962330e229a00870286a98bdf211696a3b75138df1599f28dc", "text": "=== 氣候 ===\n北京市地处暖温带半濕潤地区，气候属于暖温带半湿润大陆性季风气候，平原地区平均年降水量约600毫米，市区代表站最大年降水量为1404.6毫米，最大24小时降水量为404.2毫米，平均年最大积雪深度为7.5厘米，历史最大积雪深度33.5厘米。北京四季分明，春季多風和沙尘，夏季炎熱多雨，秋季晴朗干燥，冬季寒冷且大风猛烈。其中春季和秋季很短，大概一个月出头左右；而夏季和冬季则很长，各接近五个月。北京季风性特征明显，全年60%的降水集中在夏季的7、8月份，而其他季节空气较为干燥。年平均气温約為12.9 °C。最冷月（1月）平均气温为−3.1 °C，最热月（7月）平均气温为26.7 °C。极端最低气温为−33.2 °C（1980年1月30日佛爷顶），极端最高气温43.5 °C（1961年6月10日房山区）。市区极端最低气温为−27.4 °C（1966年2月22日），市区极端最高气温41.9 °C（1999年7月24日）。\n\n\n=== 環境問題 ===\n\n\n==== 空气污染 ====\n北京的空气污染长期为人诟病。空气的主要污染源为工业废气及汽车尾气排放。", "start_char_idx": null, "end_char_idx": null, "text_template": "{metadata_str}\n\n{content}", "metadata_template": "{key}: {value}", "metadata_seperator": "\n"}, "score": 9.0}, {"node": {"id_": "ddd0587f-2c47-41d1-a6a3-bccba0844a16", "embedding": null, "metadata": {"file_path": "/Users/dzdx/Documents/learn-llama-index/data/北京市.txt", "creation_date": "2023-11-24", "last_modified_date": "2023-11-24", "last_accessed_date": "2023-11-24"}, "excluded_embed_metadata_keys": ["creation_date", "last_modified_date", "last_accessed_date", "file_path"], "excluded_llm_metadata_keys": ["creation_date", "last_modified_date", "last_accessed_date", "file_path"], "relationships": {"1": {"node_id": "f1a4410d-0926-41d6-8844-c126071c8dbc", "node_type": "4", "metadata": {"file_path": "/Users/dzdx/Documents/learn-llama-index/data/北京市.txt", "creation_date": "2023-11-24", "last_modified_date": "2023-11-24", "last_accessed_date": "2023-11-24"}, "hash": "27b2f2b0146b996e14fae6d6f4670c7b6030f237b7daab9dff11740c5b898724"}, "2": {"node_id": "fea8a8ef-b26e-4f71-be43-376c52f6d24d", "node_type": "1", "metadata": {"file_path": "/Users/dzdx/Documents/learn-llama-index/data/北京市.txt", "creation_date": "2023-11-24", "last_modified_date": "2023-11-24", "last_accessed_date": "2023-11-24"}, "hash": "7edd377fa4ebd8962330e229a00870286a98bdf211696a3b75138df1599f28dc"}, "3": {"node_id": "a6850820-43be-4688-938d-a551f078d89d", "node_type": "1", "metadata": {"file_path": "/Users/dzdx/Documents/learn-llama-index/data/北京市.txt", "creation_date": "2023-11-24", "last_modified_date": "2023-11-24", "last_accessed_date": "2023-11-24"}, "hash": "30b8205d77ba56d5f4ffea199011d0b203f5b98785dc49d5fc9f86e16f3ea26d"}}, "hash": "fc1544a36ec674717680120e8ba773a6223dbf7b4dfade9f4d4e7cff3f69ef5c", "text": "2008年北京奥运会之后，北京的空气质量并未因此而好转，2011年2月21日，美国驻华大使馆录得的北京空气污染指数已超过了正常的可测量范围，北京市气象部门亦在2月24日通报称，截止当日北京空气质量已连续三天达到重度污染。亚洲开发银行和清华大学发布《中华人民共和国国家环境分析》报告成北京市为世界十大空气污染城市之一。飞絮则是北京每年4、5月間另一特有现象，其来源以杨絮為主，也有柳絮。由於北京城市道路绿化曾以毛白楊作為主要树种，而其白絮极易随风飘扬，因此形成了北京春天漫天飞杨絮的景观，也造成了一定程度的环境污染。随着近年来北京市通過樹幹注射抑止劑、嫁接以及用其他樹種逐漸取代毛白楊等方法治理楊柳飛絮，飞絮现象已逐漸得到控制。\n\n2011年11月-12月，北京市遭遇了持续近一个月的烟霾天气，美国驻华大使馆测量的空气污染指数连续“爆表”，然而北京市环保局发布的空气污染级别却为“轻度污染”，引发公众疑虑。后有媒体爆料，当今世界上发达国家普遍开始采用的空气污染测量标准为PM2.5（直径2.5微米的空气颗粒，乃影响大气透明度的首要污染物，同时由于其粒径较小容易进入人体）。", "start_char_idx": null, "end_char_idx": null, "text_template": "{metadata_str}\n\n{content}", "metadata_template": "{key}: {value}", "metadata_seperator": "\n"}, "score": 8.0}]}
[DebugInfo] event_type=llm, content={"messages": [{"role": "user", "content": "上下文信息如下.\n---------------------\n=== 氣候 ===\n北京市地处暖温带半濕潤地区，气候属于暖温带半湿润大陆性季风气候，平原地区平均年降水量约600毫米，市区代表站最大年降水量为1404.6毫米，最大24小时降水量为404.2毫米，平均年最大积雪深度为7.5厘米，历史最大积雪深度33.5厘米。北京四季分明，春季多風和沙尘，夏季炎熱多雨，秋季晴朗干燥，冬季寒冷且大风猛烈。其中春季和秋季很短，大概一个月出头左右；而夏季和冬季则很长，各接近五个月。北京季风性特征明显，全年60%的降水集中在夏季的7、8月份，而其他季节空气较为干燥。年平均气温約為12.9 °C。最冷月（1月）平均气温为−3.1 °C，最热月（7月）平均气温为26.7 °C。极端最低气温为−33.2 °C（1980年1月30日佛爷顶），极端最高气温43.5 °C（1961年6月10日房山区）。市区极端最低气温为−27.4 °C（1966年2月22日），市区极端最高气温41.9 °C（1999年7月24日）。\n\n\n=== 環境問題 ===\n\n\n==== 空气污染 ====\n北京的空气污染长期为人诟病。空气的主要污染源为工业废气及汽车尾气排放。\n\n2008年北京奥运会之后，北京的空气质量并未因此而好转，2011年2月21日，美国驻华大使馆录得的北京空气污染指数已超过了正常的可测量范围，北京市气象部门亦在2月24日通报称，截止当日北京空气质量已连续三天达到重度污染。亚洲开发银行和清华大学发布《中华人民共和国国家环境分析》报告成北京市为世界十大空气污染城市之一。飞絮则是北京每年4、5月間另一特有现象，其来源以杨絮為主，也有柳絮。由於北京城市道路绿化曾以毛白楊作為主要树种，而其白絮极易随风飘扬，因此形成了北京春天漫天飞杨絮的景观，也造成了一定程度的环境污染。随着近年来北京市通過樹幹注射抑止劑、嫁接以及用其他樹種逐漸取代毛白楊等方法治理楊柳飛絮，飞絮现象已逐漸得到控制。\n\n2011年11月-12月，北京市遭遇了持续近一个月的烟霾天气，美国驻华大使馆测量的空气污染指数连续“爆表”，然而北京市环保局发布的空气污染级别却为“轻度污染”，引发公众疑虑。后有媒体爆料，当今世界上发达国家普遍开始采用的空气污染测量标准为PM2.5（直径2.5微米的空气颗粒，乃影响大气透明度的首要污染物，同时由于其粒径较小容易进入人体）。\n---------------------\n请注意：在回答时仅依据给出的上下文即可，不要依据你已有的知识来回答。\n请注意：在回答时仅依据给出的上下文即可，不要依据你已有的知识来回答。\n请注意：在回答时仅依据给出的上下文即可，不要依据你已有的知识来回答。\n如果你对答案不自信的话，请回答我不知道。\n请用中文回答以下问题: 北京气候如何\n", "additional_kwargs": {}}], "additional_kwargs": {"request_timeout": 15, "timeout": 15}, "serialized": {"model": "gpt-3.5-turbo", "temperature": 0.0, "max_tokens": null, "additional_kwargs": {}, "max_retries": 10, "api_type": "open_ai", "api_base": "https://api.openai.com/v1", "api_version": "", "class_name": "openai_llm"}}
[DebugInfo] event_type=llm, content={"messages": [{"role": "user", "content": "上下文信息如下.\n---------------------\n=== 氣候 ===\n北京市地处暖温带半濕潤地区，气候属于暖温带半湿润大陆性季风气候，平原地区平均年降水量约600毫米，市区代表站最大年降水量为1404.6毫米，最大24小时降水量为404.2毫米，平均年最大积雪深度为7.5厘米，历史最大积雪深度33.5厘米。北京四季分明，春季多風和沙尘，夏季炎熱多雨，秋季晴朗干燥，冬季寒冷且大风猛烈。其中春季和秋季很短，大概一个月出头左右；而夏季和冬季则很长，各接近五个月。北京季风性特征明显，全年60%的降水集中在夏季的7、8月份，而其他季节空气较为干燥。年平均气温約為12.9 °C。最冷月（1月）平均气温为−3.1 °C，最热月（7月）平均气温为26.7 °C。极端最低气温为−33.2 °C（1980年1月30日佛爷顶），极端最高气温43.5 °C（1961年6月10日房山区）。市区极端最低气温为−27.4 °C（1966年2月22日），市区极端最高气温41.9 °C（1999年7月24日）。\n\n\n=== 環境問題 ===\n\n\n==== 空气污染 ====\n北京的空气污染长期为人诟病。空气的主要污染源为工业废气及汽车尾气排放。\n\n2008年北京奥运会之后，北京的空气质量并未因此而好转，2011年2月21日，美国驻华大使馆录得的北京空气污染指数已超过了正常的可测量范围，北京市气象部门亦在2月24日通报称，截止当日北京空气质量已连续三天达到重度污染。亚洲开发银行和清华大学发布《中华人民共和国国家环境分析》报告成北京市为世界十大空气污染城市之一。飞絮则是北京每年4、5月間另一特有现象，其来源以杨絮為主，也有柳絮。由於北京城市道路绿化曾以毛白楊作為主要树种，而其白絮极易随风飘扬，因此形成了北京春天漫天飞杨絮的景观，也造成了一定程度的环境污染。随着近年来北京市通過樹幹注射抑止劑、嫁接以及用其他樹種逐漸取代毛白楊等方法治理楊柳飛絮，飞絮现象已逐漸得到控制。\n\n2011年11月-12月，北京市遭遇了持续近一个月的烟霾天气，美国驻华大使馆测量的空气污染指数连续“爆表”，然而北京市环保局发布的空气污染级别却为“轻度污染”，引发公众疑虑。后有媒体爆料，当今世界上发达国家普遍开始采用的空气污染测量标准为PM2.5（直径2.5微米的空气颗粒，乃影响大气透明度的首要污染物，同时由于其粒径较小容易进入人体）。\n---------------------\n请注意：在回答时仅依据给出的上下文即可，不要依据你已有的知识来回答。\n请注意：在回答时仅依据给出的上下文即可，不要依据你已有的知识来回答。\n请注意：在回答时仅依据给出的上下文即可，不要依据你已有的知识来回答。\n如果你对答案不自信的话，请回答我不知道。\n请用中文回答以下问题: 北京气候如何\n", "additional_kwargs": {}}], "response": {"message": {"role": "assistant", "content": "北京气候属于暖温带半湿润大陆性季风气候，四季分明。春季多风和沙尘，夏季炎热多雨，秋季晴朗干燥，冬季寒冷且大风猛烈。其中春季和秋季很短，大概一个月出头左右；而夏季和冬季则很长，各接近五个月。年平均气温约为12.9°C，最冷月（1月）平均气温为-3.1°C，最热月（7月）平均气温为26.7°C。", "additional_kwargs": {}}, "raw": {"id": "chatcmpl-8Oh1CHvIMiSQChxURR1DuDLxDcIJ7", "object": "chat.completion", "created": 1700895926, "model": "gpt-3.5-turbo-0613", "choices": [{"index": 0, "message": {"role": "assistant", "content": "北京气候属于暖温带半湿润大陆性季风气候，四季分明。春季多风和沙尘，夏季炎热多雨，秋季晴朗干燥，冬季寒冷且大风猛烈。其中春季和秋季很短，大概一个月出头左右；而夏季和冬季则很长，各接近五个月。年平均气温约为12.9°C，最冷月（1月）平均气温为-3.1°C，最热月（7月）平均气温为26.7°C。"}, "finish_reason": "stop"}], "usage": {"prompt_tokens": 1335, "completion_tokens": 207, "total_tokens": 1542}}, "delta": null, "additional_kwargs": {"prompt_tokens": 1335, "completion_tokens": 207, "total_tokens": 1542}}}
[DebugInfo] event_type=llm, content={"messages": [{"role": "user", "content": "以下是来自多个来源的上下文信息。\n---------------------\n北京气候属于暖温带半湿润大陆性季风气候，四季分明。春季多风和沙尘，夏季炎热多雨，秋季晴朗干燥，冬季寒冷且大风猛烈。其中春季和秋季很短，大概一个月出头左右；而夏季和冬季则很长，各接近五个月。年平均气温约为12.9°C，最冷月（1月）平均气温为-3.1°C，最热月（7月）平均气温为26.7°C。\n---------------------\n根据来自多个来源的信息，而非先前的知识，回答下列查询。\n查询：北京气候如何\n答案：", "additional_kwargs": {}}], "additional_kwargs": {"request_timeout": 15, "timeout": 15}, "serialized": {"model": "gpt-3.5-turbo", "temperature": 0.0, "max_tokens": null, "additional_kwargs": {}, "max_retries": 10, "api_type": "open_ai", "api_base": "https://api.openai.com/v1", "api_version": "", "class_name": "openai_llm"}}
[DebugInfo] event_type=llm, content={"messages": [{"role": "user", "content": "以下是来自多个来源的上下文信息。\n---------------------\n北京气候属于暖温带半湿润大陆性季风气候，四季分明。春季多风和沙尘，夏季炎热多雨，秋季晴朗干燥，冬季寒冷且大风猛烈。其中春季和秋季很短，大概一个月出头左右；而夏季和冬季则很长，各接近五个月。年平均气温约为12.9°C，最冷月（1月）平均气温为-3.1°C，最热月（7月）平均气温为26.7°C。\n---------------------\n根据来自多个来源的信息，而非先前的知识，回答下列查询。\n查询：北京气候如何\n答案：", "additional_kwargs": {}}], "response": {"message": {"role": "assistant", "content": "北京气候属于暖温带半湿润大陆性季风气候，四季分明。春季多风和沙尘，夏季炎热多雨，秋季晴朗干燥，冬季寒冷且大风猛烈。其中春季和秋季很短，大概一个月出头左右；而夏季和冬季则很长，各接近五个月。年平均气温约为12.9°C，最冷月（1月）平均气温为-3.1°C，最热月（7月）平均气温为26.7°C。", "additional_kwargs": {}}, "raw": {"id": "chatcmpl-8Oh1HTx7BtLao7WjQ80P5775txNTR", "object": "chat.completion", "created": 1700895931, "model": "gpt-3.5-turbo-0613", "choices": [{"index": 0, "message": {"role": "assistant", "content": "北京气候属于暖温带半湿润大陆性季风气候，四季分明。春季多风和沙尘，夏季炎热多雨，秋季晴朗干燥，冬季寒冷且大风猛烈。其中春季和秋季很短，大概一个月出头左右；而夏季和冬季则很长，各接近五个月。年平均气温约为12.9°C，最冷月（1月）平均气温为-3.1°C，最热月（7月）平均气温为26.7°C。"}, "finish_reason": "stop"}], "usage": {"prompt_tokens": 272, "completion_tokens": 207, "total_tokens": 479}}, "delta": null, "additional_kwargs": {"prompt_tokens": 272, "completion_tokens": 207, "total_tokens": 479}}}

```

11. 最后可以写一个简单的loop 作为交互式的接口
```python

def chat_loop():
    while True:
        query = input("\nEnter a query:")
        if query == "exit":
            break
        if query.strip() == "":
            continue
        ans = route_query(query)
        print(ans)
```

```bash

Enter a query:北京气候如何
北京气候属于暖温带半湿润大陆性季风气候，四季分明。春季多风和沙尘，夏季炎热多雨，秋季晴朗干燥，冬季寒冷且大风猛烈。其中春季和秋季很短，大概一个月出头左右；而夏季和冬季则很长，各接近五个月。年平均气温约为12.9°C，最冷月（1月）平均气温为-3.1°C，最热月（7月）平均气温为26.7°C。

Enter a query:
```