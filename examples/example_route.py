
from import_route import EchoNameEngine, create_route_query_engine

hong_engine = EchoNameEngine("小红")
hei_engine = EchoNameEngine("小黑")

route_query_engine = create_route_query_engine([hong_engine, hei_engine],
                                               ["可以和小红说话",
                                                "可以和小黑说话"])
assert route_query_engine.query("小黑你好").response == '我是小黑'
assert route_query_engine.query("早上好小红").response == '我是小红'
