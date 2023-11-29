from common.config import USE_ANSWER

if USE_ANSWER:
    from query_answer.query_engine import load_indices
    from query_answer.query_engine import DocumentQueryEngineFactory
    from query_answer.retrievers import QueryEngineToRetriever
    from query_answer.route import Chatter, EchoNameEngine, create_route_query_engine
    from query_answer.compose import create_compose_query_engine
    from query_answer.query_engine import load_index, DocumentQueryEngineFactory, create_response_synthesizer, \
        load_indices
    from query_answer.retrievers import MultiRetriever
    from build_answer.download import download
    from build_answer.index import download_and_build_index, data_dir, index_dir, build_all, build_nodes
else:
    from query.query_engine import load_indices
    from query.query_engine import DocumentQueryEngineFactory
    from query.retrievers import QueryEngineToRetriever
    from query.route import Chatter, EchoNameEngine, create_route_query_engine
    from query.compose import create_compose_query_engine
    from query.query_engine import load_index, DocumentQueryEngineFactory, create_response_synthesizer, load_indices
    from query.retrievers import MultiRetriever
    from build.download import download
    from build.index import download_and_build_index, data_dir, index_dir, build_all, build_nodes
