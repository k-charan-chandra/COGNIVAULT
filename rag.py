from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
load_dotenv()
# 1. Embedding model
embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001"
)


# 2. Chroma
vector_store = Chroma(
    embedding_function=embeddings,
    persist_directory="chroma.db",
)


# 3. Load PDF
loader = PyPDFLoader(
    "ABC_College_Fee_Structure_50_Pages.pdf"
)

documents = loader.load()


# 4. Split PDF
splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=100
)

chunks = splitter.split_documents(documents)


# 5. Add chunks to Chroma
vector_store.add_documents(chunks)


# 6. Chroma retriever
vector_retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5}
)


# 7. BM25 retriever
bm25_retriever = BM25Retriever.from_documents(chunks)
bm25_retriever.k = 5


# 8. Hybrid retriever
retriever = EnsembleRetriever(
    retrievers=[
        bm25_retriever,
        vector_retriever
    ],
    weights=[
        0.4,
        0.6
    ]
)


# 9. User query
query = "What is the fee structure for B.Tech?"


# 10. Retrieve
results = retriever.invoke(query)


# 11. Print RAG output
print("\n========== RAG OUTPUT ==========\n")

for i, doc in enumerate(results, 1):

    print(f"========== CHUNK {i} ==========")
    print("Page:", doc.metadata.get("page", "Unknown"))
    print("Content:")
    print(doc.page_content)
    print()