from route import chat


def chat_loop():
    while True:
        query = input("\nEnter a query:")
        if query == "exit":
            break
        if query.strip() == "":
            continue
        ans = chat(query)
        print(ans)


if __name__ == '__main__':
    print(chat("你好呀"))
    print(chat("北京气候如何"))
    print(chat("杭州市在中国哪里"))
    # chat_loop()
