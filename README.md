# 介绍

**LlamaIndex** 它在 LLM 和外部数据源（如 API、PDF、SQL
等）之间提供一个简单的接口进行交互。它提了供结构化和非结构化数据的索引，有助于抽象出数据源之间的差异。它可以存储提示工程所需的上下文，处理当上下文窗口过大时的限制，并有助于在查询期间在成本和性能之间进行权衡。

# 主要阶段

基于LlamaIndex构建应用的5个关键阶段

## 数据加载

这指的是从数据源获取数据，无论是文本文件、PDF、另一个网站、数据库还是API，都可以将其导入到流程中

## 索引

创建一个允许查询数据的数据结构。通常会创建 vector embeddings，即对数据含义进行数值表示，并采用其他多种元数据策略，以便轻松准确地找到上下文相关的数据

## 存储

一旦完成索引操作，通常希望将索引及其他元数据存储起来，以避免重新构建索引

## 查询

根据给定的索引策略，可以使用LLM和LlamaIndex等多种方式进行查询操作，比如向量匹配召回、Summary相关度召回等等。

## 评估

在任何流程中都有一个关键步骤就是检查其相对于其他策略或更改时效果如何。评估提供了客观度量标准来衡量您对查询响应精确性和速度方面表现

# 模块划分

## Loader

定义如何从各种数据源中提取结构化的数据
例如：`Node Parser`、`Database Reader`、`Twitter Reader` 等等
利用上述工具，可以从数据源中得到 `Document` 和 `Node` 这两个数据结构
Document: 围绕任何数据源的通用容器 - 例如，PDF、API输出或从数据库检索到的数据
Node: 表示Document 的一个chunk

## Index

Index是由`Document` 组成的快速查询的数据结构，通过LLM辅助进行查询，可以筛选出topN的相关 `Node`
比如 `VectorStoreIndex` 、`DocumentSummaryIndex`、`KeywordTableIndex`、`TreeIndex`...
这些索引简单的可以直接存在本地文件内，也可以存放在独立的向量和文档数据库，llama index集成了很多的成熟数据库系统作为index存储介质

## LLM

llama index对多种LLM做了统一的封装，并内置了许多方便调用LLM的方法。
支持`chatgpt` 、`claude` 等在线的LLM，也支持 `llama-cpp` 这种local llm。
使用到LLM的模块都可以方便的自定义prompt来代替原生的prompt进行调优

## Query

### Retrieval

从索引中找到并返回与查询最相关的top-k的文档

- VectorIndexRetriever: 利用embedding 的向量相似度查找topK的nodes
- DocumentSummaryIndexRetriever: 利用大模型计算和query匹配度最高的node的summary
- TreeSelectLeafRetriever: TreeIndex是对文档的多个chunk利用LLM自底向上逐级summary，查询的时候用LLM自顶向下查找相当度最高的summary节点，一直找到叶子的chunk

### Postprocessing

当 Retrieval 到的 Node 可选择重新排序、转换或过滤时，例如通过过滤具有特定的metadata的Node
- LLMRerank: 通过LLM，对Retrieval到的nodes进行相关度重排
- KeywordNodePostprocessor: 过滤nodes，需要node的content包含或者没有指定的keyword

### Response synthesis

把Query和召回的相关的Nodes结合prompt一起发送给LLM，得到对应问题的回答
- TreeSummarize: 处理nodes过多的场景，自底向上逐步总结chunk，直到生成最后的答案

## Observability

tracing 应用的整个查询链路，结合每一步的输入输出，分析查询链路的正确率和性能的瓶颈

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

3.
准备一部分数据集，作为我们可供查询的数据，这边从维基百科下载了中国几个城市的介绍数据，执行下载操作 `python download_wiki.py`

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
def load_index(file: str) -> BaseIndex:
    storage_context = StorageContext.from_defaults(persist_dir=os.path.join(index_dir, file))
    index = load_index_from_storage(
        storage_context=storage_context, service_context=service_context
    )
    return index


def build_index(file: str, documents: List[Document]) -> BaseIndex:
    storage_context = StorageContext.from_defaults()
    index = VectorStoreIndex.from_documents(
        documents,
        service_context=service_context,
        storage_context=storage_context,
    )
    index.index_struct.index_id = file
    storage_context.persist(persist_dir=os.path.join(index_dir, file))
    return index


city_indices = {}
for file in os.listdir(data_dir):
    basename = os.path.basename(file)
    if os.path.exists(os.path.join(index_dir, file)):
        index = load_index(file)
    else:
        documents = SimpleDirectoryReader(input_files=[os.path.join(data_dir, file)]).load_data()
        index = build_index(file, documents)
    city_indices[basename] = index
```

5. 基于index构建一个多城市组合的查询引擎，这边会提供summary，把问题路由到对应城市的index进行查询

```python

@dataclass
class DocQueryEngine:
    index: BaseIndex
    query_engine: RetrieverQueryEngine
    summary: str


query_engines = []
for city, vector_index in city_indices.items():
    retriever = vector_index.as_retriever()
    node_postprocessors = [
        LLMRerank(top_n=3)
    ]
    query_engine = RetrieverQueryEngine.from_args(
        retriever, node_postprocessors=node_postprocessors,
        service_context=service_context,
        text_qa_template=Prompt(
            CH_TEXT_QA_PROMPT_TMPL, prompt_type=PromptType.QUESTION_ANSWER
        ))
    summary = f"""  
        此内容包含关于{city}的维基百科文章。  
        如果您需要查找有关{city}的具体事实，请使用此索引。"  
        如果您想分析多个城市，请不要使用此索引。  
        """
    query_engines.append(DocQueryEngine(
        index=vector_index,
        query_engine=query_engine,
        summary=summary
    ))

graph = ComposableGraph.from_indices(
    TreeIndex,
    [e.index for e in query_engines],
    [e.summary for e in query_engines],
    service_context=service_context,
)
```

6. 对于返回了超过LLM context limit的多个节点，我们采用 TREE_SUMMARIZE 的方式逐步总结，得到最终答案

```python

response_synthesizer = get_response_synthesizer(
    response_mode=ResponseMode.TREE_SUMMARIZE,
    text_qa_template=Prompt(
        CH_TEXT_QA_PROMPT_TMPL, prompt_type=PromptType.QUESTION_ANSWER
    ),
    service_context=service_context,
)

query_engine = graph.as_query_engine(
    response_synthesizer=response_synthesizer,
    custom_query_engines={e.index.index_id: e.query_engine for e in query_engines},
    service_context=service_context,
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

  
def chat(query):  
    resp = query_engine.query(query)  
    if debug_handler:  
        for event in debug_handler.get_events():  
            print(  
                f"event_type={event.event_type}, content={json.dumps(event.payload, ensure_ascii=False, cls=ObjectEncoder)}")  
    print(f"Query: {query}")  
    print(f"Response: {resp}")
```

8. 最后我们对查询引擎发起提问

```python
chat("北京气候如何")
chat("杭州位于中国哪里")
```

得到结果

```
Query: 北京气候如何
Response: 北京气候属于暖温带半湿润大陆性季风气候，四季分明。春季多风和沙尘，夏季炎热多雨，秋季晴朗干燥，冬季寒冷且大风猛烈。其中春季和秋季很短，大概一个月出头左右；而夏季和冬季则很长，各接近五个月。年平均气温约为12.9 °C。最冷月（1月）平均气温为−3.1 °C，最热月（7月）平均气温为26.7 °C。极端最低气温为−33.2 °C（1980年1月30日佛爷顶），极端最高气温43.5 °C（1961年6月10日房山区）。市区极端最低气温为−27.4 °C（1966年2月22日），市区极端最高气温41.9 °C（1999年7月24日）。

Query: 杭州位于中国哪里
Response: 杭州位于中国的浙江省北部。
```