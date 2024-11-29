import ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM


def ask_ollama(query):
    answer = ollama.chat(
        model="gemma2:2b",
        messages=[
            {'role': 'system', 'content': "You are helpfull assistance"},
            {"role": "user", "content": query}],
        stream=False
    )
    return answer['message']['content']


class Query:
    def __init__(self, question, chunk):
        self.question = question
        self.chunk = chunk


def ask_ollama_lg(query: Query):
    template = """
    Question: {question}
    Chunk: {chunk}
    Answer: Summarize text from chunk
    """

    prompt = ChatPromptTemplate.from_template(template)
    model = OllamaLLM(model="gemma2:2b")
    chain = prompt | model

    result = chain.invoke({"question": query.question, "chunk": query.chunk})
    return result


if __name__ == '__main__':
    query = "What is your name ? "
    print(query)
    print(ask_ollama(query))
