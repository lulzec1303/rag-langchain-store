from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from vectorize import Vectorize
import os
from dotenv import load_dotenv

load_dotenv()

embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

docs = []
for file in ["store_faq.txt", "store_catalog.txt"]:
    loader = TextLoader(file)
    docs.extend(loader.load())

splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
chunks = splitter.split_documents(docs)

vectors = embedder.embed_documents([chunk.page_content for chunk in chunks])
payloads = [{"text": chunk.page_content} for chunk in chunks]

client = Vectorize(api_key=os.getenv("VECTORIZE_API_KEY"))
index = client.Index("store-rag-index")  # Change this to your index name

index.upsert(vectors=vectors, payloads=payloads)
print(f"✅ Uploaded {len(vectors)} chunks to Vectorize.")
