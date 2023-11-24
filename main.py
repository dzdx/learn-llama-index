from route import route_query


def chat_loop():
    while True:
        query = input("\nEnter a query:")
        if query == "exit":
            break
        if query.strip() == "":
            continue
        ans = route_query(query)
        print(ans)


if __name__ == '__main__':
    # chat_loop()
    print(route_query("你好呀"))
    print(route_query("北京气候如何"))
    print(route_query("杭州在中国的位置"))
