#! coding=utf-8
import os
import random
from typing import Dict, List, Tuple

import pandas as pd
from llama_index import ServiceContext
from llama_index.evaluation import RetrieverEvaluator, generate_question_context_pairs, RetrievalEvalResult
from llama_index.finetuning import EmbeddingQAFinetuneDataset
from llama_index.indices.base import BaseIndex
from llama_index.indices.base_retriever import BaseRetriever
from llama_index.query_engine import RetrieverQueryEngine
from tqdm import tqdm

from common.config import ROOT_PATH
from query import load_indices
from query import DocumentQueryEngineFactory
from retrievers import QueryEngineToRetriever
from common.llm import create_llm
from common.prompt import CH_QA_GENERATE_PROMPT_TMPL

QA_DATASET_DIR = os.path.join(ROOT_PATH, "qa_dataset")

FORCE_REBUILD_DATASET = True


class Evaluator:
    def __init__(self, force_rebuild_dataset: bool = False):
        self.force_rebuild_dataset = force_rebuild_dataset
        self.city_indices: Dict[str, List[BaseIndex]] = load_indices()
        self.llm = create_llm()
        self.service_context = ServiceContext.from_defaults(
            llm=self.llm
        )

    def _find_retrievers(self, retriever_query_engine: RetrieverQueryEngine) -> List[Tuple[str, BaseRetriever]]:
        name_to_retrievers = []
        multi_retriever = retriever_query_engine._retriever
        for retriever in multi_retriever._retrievers:
            name_to_retrievers.append((retriever.__class__.__name__, retriever))

        name_to_retrievers.append((multi_retriever.__class__.__name__, multi_retriever))
        name_to_retrievers.append(("query_engine", QueryEngineToRetriever(retriever_query_engine)))
        return name_to_retrievers

    def generate_qa_dataset(self, city_name):
        indices = self.city_indices[city_name]
        doc_query_engine = DocumentQueryEngineFactory(indices)
        qa_dataset = generate_question_context_pairs(
            random.sample(list(doc_query_engine.doc_store().docs.values()), 5),
            llm=self.llm,
            num_questions_per_chunk=2,
            qa_generate_prompt_tmpl=CH_QA_GENERATE_PROMPT_TMPL,
        )
        if not os.path.exists(QA_DATASET_DIR):
            os.makedirs(QA_DATASET_DIR, exist_ok=True)
        qa_dataset.save_json(os.path.join(QA_DATASET_DIR, f"{city_name}.json"))

    def evaluate(self):
        for city_name, indices in tqdm(list(self.city_indices.items())[:1], desc="document index"):
            doc_query_engine = DocumentQueryEngineFactory(indices)
            dataset_file = os.path.join(QA_DATASET_DIR, f"{city_name}.json")
            if self.force_rebuild_dataset or not os.path.exists(dataset_file):
                self.generate_qa_dataset(city_name)
            qa_dataset = EmbeddingQAFinetuneDataset.from_json(dataset_file)
            doc_query_engine.create_query_engine(self.service_context)
            retriever_query_engine = doc_query_engine.create_query_engine(self.service_context)
            name_to_retrievers = self._find_retrievers(retriever_query_engine)
            results = {}
            for name, retriever in tqdm(name_to_retrievers, desc="retrievers"):
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
    evaluator = Evaluator(force_rebuild_dataset=FORCE_REBUILD_DATASET)
    evaluator.evaluate()
