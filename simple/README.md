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

wiki_titles = ["北京市"]

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

4. 首先我们对索引进行构建，这边简单起见直接使用本地文件作为索引

```python

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


index = load_or_build_city_index(service_context, '北京市.txt')[0]
```

6. 对于返回了超过LLM context limit的多个节点，我们采用 TREE_SUMMARIZE 的方式逐步总结，得到最终答案

```python
response_synthesizer = get_response_synthesizer(
    response_mode=ResponseMode.TREE_SUMMARIZE,
    service_context=service_context,
)
```

6. 基于index构建一个查询引擎

```python
node_postprocessors = [
    LLMRerank(top_n=3, service_context=service_context)
]

query_engine = RetrieverQueryEngine.from_args(
    index.as_retriever(),
    node_postprocessors=node_postprocessors,
    response_synthesizer=response_synthesizer,
    service_context=service_context,
)

```

7. 运行一下试试
```python
print(query_engine.query("北京天气如何"))
```

```bash
python main.py
$ 北京的气候属于暖温带半湿润大陆性季风气候，四季分明。春季多风和沙尘，夏季炎热多雨，秋季晴朗干燥，冬季寒冷且大风猛烈。其中春季和秋季较短，大约一个月左右，而夏季和冬季则较长，各接近五个月。北京的季风性特征明显，全年60%的降水集中在夏季的7、8月份，其他季节空气较为干燥。年平均气温约为12.9°C，最冷的月份是1月，平均气温为-3.1°C，最热的月份是7月，平均气温为26.7°C。极端最低气温为-33.2°C，极端最高气温为43.5°C。北京的空气污染长期为人诟病，主要污染源为工业废气和汽车尾气排放。
```