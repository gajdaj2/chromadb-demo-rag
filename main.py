import numpy as np
from sentence_transformers import CrossEncoder

from service import chroma_collection, ask_ollama_lg, Query


def scoring(query):
    cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    results = chroma_collection.query(query_texts=query, n_results=5, include=['documents', 'embeddings'])
    retrieved_documents = results['documents'][0]

    pairs = [(query, doc) for doc in retrieved_documents]

    scores = cross_encoder.predict(pairs)
    print("Scores")
    for score in scores:
        print(score)

    the_best_answer_from_documents = retrieved_documents[np.argmax(scores)]

    return the_best_answer_from_documents


query = "Who is CEO in Microsoft ?"

the_best_answer_from_documents = scoring(query)

print(ask_ollama_lg(Query(query, the_best_answer_from_documents)))
