# 手把手入门 llama index

## 介绍

本项目可以搭建一个简单的问答系统，可以查询 北京、上海市、深圳市 在wiki上的信息，并进行简单的意图识别，进行简单闲聊.
主要用来演示如何用 llama index做一个简单的 RAG 应用

`query_todo` 和 `build_todo` 中的 TODO 为需要编写代码的部分，需要跑通 test.py 中所有的测试用例，也可以参考 `query`
和 `build` 两个模块代码的实现

## 运行准备

系统依赖

- \>=Python3.8

安装依赖

```bash
virtualenv venv
source venv/bin/activate
pip install -r requirements.txt
```

配置OpenAI ApiKey

```bash
export OPENAI_API_KEY=xxxx
```

## 补全 TODO

### 1. 下载 wiki 并进行解析

`pytest test.py::test_build_nodes`

```python
def build_nodes(data_file: str) -> List[BaseNode]:
    # 把输入的文档解析为Document对象，txt的文件，一个文件一个Document
    # 如果是pdf文件，会有特殊处理，一页一个Document
    documents = SimpleDirectoryReader(input_files=[data_file]).load_data()
    for doc in documents:
        # 过滤掉自动生成的 `file_path` 这个metadata, 防止对embedding结果造成干扰
        doc.excluded_llm_metadata_keys.append("file_path")
        doc.excluded_embed_metadata_keys.append("file_path")
    # 把 document 按句子进行分割成多个 nodes
    return service_context.node_parser.get_nodes_from_documents(documents)
```

### 2. 构建单个城市的召回索引

`pytest test.py::test_build_index`

```python

def build_index(index_file: str, data_file: str):
    if os.path.exists(index_file):
        return
    nodes = build_nodes(data_file)
    # 两个index共用一个存储目录，可以复用DocumentStore
    storage_context = StorageContext.from_defaults()
    vector_index = VectorStoreIndex(nodes,
                                    service_context=service_context,
                                    storage_context=storage_context,
                                    show_progress=True)
    # TreeIndex的 num_children 是自底向上逐步summary, 生成parent node的时候，每个parent包含多少个children nodes
    # summary_template 可以替换为中文的prompt，更稳定的得到中文的summary
    tree_index = TreeIndex(nodes, num_children=8,
                           service_context=service_context,
                           storage_context=storage_context,
                           summary_template=CH_SUMMARY_PROMPT,
                           show_progress=True)
    # 把两个索引的生成数据存储到index_file这个目录
    storage_context.persist(persist_dir=index_file)
```

### 3. 构建三个城市的召回索引

`pytest test.py::test_build_all_index`

```python

def build_all():
    # 单独构建全部的索引，后续查询会用到多索引
    titles = ['北京市', '上海市', '深圳市']
    for title in titles:
        download_and_build_index(title, data_dir, index_dir)


```

### 4. 总结生成答案

`pytest test.py::test_response_synthesizer`

```python
def create_response_synthesizer(service_context: ServiceContext = None) -> BaseSynthesizer:
    # 采用TreeSummarize的方式对多个上下文进行逐步总结，防止超过llm的context limit
    # 同时用中文prompt得到更加稳定的中文summary
    return get_response_synthesizer(
        response_mode=ResponseMode.TREE_SUMMARIZE,
        summary_template=CH_TREE_SUMMARIZE_PROMPT,
        service_context=service_context,
    )

```

### 5. 创建召回器

`pytest test.py::test_retriever`

```python

class DocumentQueryEngineFactory:
    ...

    def create_retrievers(self):
        ret = []
        for index in self.indices:
            # 每个索引可能会用不同的生成retriever的方式
            # 取 `similarity_top_k` 和query 最相似的 node
            if isinstance(index, VectorStoreIndex):
                ret.append(index.as_retriever(similarity_top_k=8))
            if isinstance(index, TreeIndex):
                # TreeRetriever采用自顶向下逐步检索的方式得到叶子节点
                ret.append(index.as_retriever(retriever_mode=TreeRetrieverMode.SELECT_LEAF_EMBEDDING))
        return ret


```

### 6. 创建单个文档的查询引擎

`pytest test.py::test_query_engine`

```python

    def create_query_engine(self, service_context: ServiceContext) -> RetrieverQueryEngine:
        # 组合多索引召回、排序，答案合成，构建最终的query engine
        retriever = MultiRetriever(self.create_retrievers())
        # LLMRerank只选取最相关的top_n, 进一步提高命中率，防止召回阶段拿到不相关的内容
        node_postprocessors = [
            LLMRerank(top_n=4, choice_batch_size=2, choice_select_prompt=CH_CHOICE_SELECT_PROMPT,
                      service_context=service_context)
        ]
        return RetrieverQueryEngine.from_args(
            retriever,
            node_postprocessors=node_postprocessors,
            service_context=service_context,
            response_synthesizer=create_response_synthesizer(service_context)
        )
```

### 7. 创建组合查询引擎

`pytest test.py::test_compose_query_engine`

```python

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

```

### 9. 简单的路由分发

`pytest test.py::test_simple_route`

```python

def create_route_query_engine(query_engines: List[BaseQueryEngine], descriptions: List[str],
                              service_context: ServiceContext = None):
    assert len(query_engines) == len(descriptions)
    tools = []
    for i, query_engine in enumerate(query_engines):
        query_tool = QueryEngineTool.from_defaults(
            query_engine=query_engine,
            description=descriptions[i]
        )
        tools.append(query_tool)
    return llama_index.query_engine.RouterQueryEngine(
        selector=LLMSingleSelector.from_defaults(service_context=service_context,
                                                 prompt_template_str=CH_SINGLE_SELECT_PROMPT_TMPL),
        service_context=service_context,
        query_engine_tools=tools
    )

```

### 10. 意图识别

`pytest test.py::test_route_query_engine`

```python


class Chatter:
    ...
    def create_query_engine(self):
        index_query_engine = create_compose_query_engine(self.city_indices, self.service_context)
        index_summary = f"提供 {', '.join(self.city_indices.keys())} 这几个城市的相关信息"
        llm_query_engine = LlmQueryEngine(llm=self.llm, callback_manager=self.cb_manager)
        llm_summary = "提供其他所有信息"

        route_query_engine = create_route_query_engine(
            [index_query_engine, llm_query_engine],
            [index_summary, llm_summary],
            service_context=self.service_context)
        return route_query_engine


```


# 完整运行项目
```bash
python main.py
Enter a query:北京气候如何
```