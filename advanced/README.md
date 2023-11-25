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

7. 如果希望观测整个查询链路的过程，可以提供tracing功能

``` python
  
DEBUG = False  
  
if DEBUG:  
    debug_handler = LlamaDebugHandler()  
    cb_manager = CallbackManager([debug_handler])  
else:  
    debug_handler = None  
    cb_manager = CallbackManager()
```

8. 如何我们还希望提供回答除了5个城市相关信息以外的问题的功能，可以使用RouteQueryEngine进行分流

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

9. 让我们测试一下我们的问答系统

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
10. 最后可以写一个简单的loop 作为交互式的接口
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