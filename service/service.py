import chromadb
import numpy as np
from PyPDF2 import PdfReader
from chromadb.utils.embedding_functions.sentence_transformer_embedding_function import \
    SentenceTransformerEmbeddingFunction
from langchain_text_splitters import SentenceTransformersTokenTextSplitter
from semantic_text_splitter import TextSplitter
from sentence_transformers import CrossEncoder
import chromadb.utils.embedding_functions as embedding_functions

reader = PdfReader("./documents/2023_Annual_Report.pdf")

pdf_to_text = [p.extract_text().strip() for p in reader.pages]

text_from_pdf_clean = [text for text in pdf_to_text if text]

splitter = TextSplitter(1500, overlap=25)

text = ""

for x in text_from_pdf_clean:
    text += x

chunks = splitter.chunks(text)

token_splitter = SentenceTransformersTokenTextSplitter( chunk_overlap=0,
                                                       tokens_per_chunk=256)

token_split_text = []

for text in chunks:
    token_split_text += token_splitter.split_text(text)

embedding_function = SentenceTransformerEmbeddingFunction(model_name="msmarco-distilbert-base-v2")

client = chromadb.PersistentClient(path="./chroma_db")

chroma_collection = client.get_or_create_collection("documents", embedding_function=embedding_function)

ids = [str(i) for i in range(len(token_split_text))]

chroma_collection.add(ids=ids, documents=token_split_text)

query = "Who is CIO of Microsoft ?"
print(query)

results = chroma_collection.query(query_texts=query, n_results=5)

print(results['documents'][0][0])


