from route import chatter


def chat_loop():
    while True:
        query = input("\nEnter a query:")
        if query == "exit":
            break
        if query.strip() == "":
            continue
        ans = chatter.chat(query)
        print(ans)


if __name__ == '__main__':
    print(chatter.chat("你好呀"))
    print(chatter.chat("北京气候如何"))
    print(chatter.chat("杭州市在中国哪里"))
    # chat_loop()
