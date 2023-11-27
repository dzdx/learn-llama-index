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
    print(route_query("北京气候如何"))
    print(route_query("杭州市在中国哪里"))
    # for q in ["你好呀",
    #           "北京气候如何",
    #           "杭州在中国的位置"]:
    #     print(f"Question: {q}")
    #     print(f"Answer: {route_query(q)}")
    # chat_loop()
