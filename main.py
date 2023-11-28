from query.route import Chatter


def chat_loop():
    chatter = Chatter()
    while True:
        query = input("\nEnter a query:")
        if query == "exit":
            break
        if query.strip() == "":
            continue
        ans = chatter.chat(query)
        print(ans)


if __name__ == '__main__':
    chat_loop()
