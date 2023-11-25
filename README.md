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
