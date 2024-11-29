import os

import ollama
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM
from openai import OpenAI

load_dotenv()

def ask_gpt(query, chunk):
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    template = """
    Question: {question}
    Chunk: {chunk}
    Answer: Summarize text from chunk
    """.format(question=query, chunk=chunk)


    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are helpfull assistance"},
            {"role": "user", "content": template}
        ],
        temperature=0,
        max_tokens=1000
    )
    return response.choices[0].message.content


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
    model="gemma2:2b"
    chain = prompt | model

    result = chain.invoke({"question": query.question, "chunk": query.chunk})
    return result


if __name__ == '__main__':
    query = "What is your name ? "
    print(query)
    print(ask_gpt(query))
