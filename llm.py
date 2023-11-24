from llama_index.llms import OpenAI, ChatMessage

llm_gpt3 = OpenAI(temperature=0, model="gpt-3.5-turbo")


def llm_predict(content: str):
    response = llm_gpt3.chat([ChatMessage(
        content=content
    )])
    return response.message.content
