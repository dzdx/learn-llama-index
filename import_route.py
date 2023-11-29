from common.config import ROUTE_TODO

if ROUTE_TODO:
    from query_todo.query_engine import load_indices
    from query_todo.query_engine import DocumentQueryEngineFactory
    from query_todo.retrievers import QueryEngineToRetriever
    from query_todo.route import Chatter, EchoNameEngine, create_route_query_engine
    from query_todo.compose import create_compose_query_engine
    from query_todo.query_engine import load_index, DocumentQueryEngineFactory, create_response_synthesizer, \
        load_indices
    from query_todo.retrievers import MultiRetriever
    from build_todo.download import download
    from build_todo.index import download_and_build_index, data_dir, index_dir, build_all, build_nodes
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
