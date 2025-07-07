from langchain.vectorstores import Vectorize as VectorizeRetriever
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFaceHub
from langchain.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv
import os

load_dotenv()

embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

retriever = VectorizeRetriever(
    index_name="store-rag-index",
    vectorize_api_key=os.getenv("VECTORIZE_API_KEY"),
    embedding=embedder,
)

llm = HuggingFaceHub(
    repo_id="google/flan-t5-large",
    model_kwargs={"temperature": 0.5, "max_length": 300},
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
)

qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever.as_retriever())

while True:
    query = input("\nAsk a question about the store (or type 'exit'): ")
    if query.lower() == "exit":
        break
    answer = qa.run(query)
    print("Answer:", answer)
