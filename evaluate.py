#! coding=utf-8
from typing import Dict, List

import pandas as pd
from llama_index import ServiceContext
from llama_index.evaluation import RetrieverEvaluator, generate_question_context_pairs, RetrievalEvalResult
from llama_index.indices.base import BaseIndex
from llama_index.indices.base_retriever import BaseRetriever
from llama_index.query_engine import RetrieverQueryEngine
from tqdm import tqdm

from index import load_indices
from query import DocumentQueryEngine
from retrievers import QueryEngineToRetriever
from common.llm import create_llm
from common.prompt import CH_QA_GENERATE_PROMPT_TMPL


class Evaluator:
    def __init__(self):
        self.city_indices: Dict[str, List[BaseIndex]] = load_indices()
        self.llm = create_llm()
        self.service_context = ServiceContext.from_defaults(
            llm=self.llm
        )

    def _find_retrievers(self, retriever_query_engine: RetrieverQueryEngine) -> Dict[str, BaseRetriever]:
        custom_retriever = retriever_query_engine._retriever
        name_to_retrievers = {
            "query_engine": QueryEngineToRetriever(retriever_query_engine),
            custom_retriever.__class__.__name__: custom_retriever
        }
        for retriever in custom_retriever._retrievers:
            name_to_retrievers[retriever.__class__.__name__] = retriever
        return name_to_retrievers

    def evaluate(self):
        for city_name, indices in tqdm(self.city_indices.items(), desc="document index"):
            doc_query_engine = DocumentQueryEngine(indices)
            qa_dataset = generate_question_context_pairs(
                list(doc_query_engine.doc_store().docs.values())[:5],
                llm=self.llm,
                num_questions_per_chunk=2,
                qa_generate_prompt_tmpl=CH_QA_GENERATE_PROMPT_TMPL,
            )
            doc_query_engine.create_query_engine(self.service_context)
            retriever_query_engine = doc_query_engine.create_query_engine(self.service_context)
            name_to_retrievers = self._find_retrievers(retriever_query_engine)
            results = {}
            for name, retriever in tqdm(name_to_retrievers.items(), desc="retrievers"):
                eval_results = []
                for query, doc_ids in qa_dataset.query_docid_pairs:
                    retriever_evaluator = RetrieverEvaluator.from_metric_names(
                        ["mrr", "hit_rate"], retriever=retriever
                    )
                    eval_result = retriever_evaluator.evaluate(query, doc_ids)
                    eval_results.append(eval_result)
                    print(display_eval_result(city_name, name, eval_result))
                results[name] = eval_results
            print('')
            print(display_results(results))


def display_eval_result(city, name, eval_result: RetrievalEvalResult):
    return f"Document: {city}\nRetriever: {name}\n{eval_result}"


def display_results(results: Dict[str, List[RetrievalEvalResult]]):
    """Display results from evaluate."""
    name_list = []
    hit_rate_list = []
    mrr_list = []

    for name, eval_results in results.items():
        metric_dicts = []
        for eval_result in eval_results:
            metric_dict = eval_result.metric_vals_dict
            metric_dicts.append(metric_dict)

        full_df = pd.DataFrame(metric_dicts)

        hit_rate = full_df["hit_rate"].mean()
        mrr = full_df["mrr"].mean()
        name_list.append(name)
        hit_rate_list.append(hit_rate)
        mrr_list.append(mrr)

    metric_df = pd.DataFrame(
        {"retrievers": name_list, "hit_rate": hit_rate_list, "mrr": mrr_list}
    )
    return metric_df


if __name__ == '__main__':
    evaluator = Evaluator()
    evaluator.evaluate()
