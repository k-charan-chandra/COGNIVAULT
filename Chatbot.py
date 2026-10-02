import os
import math
import sqlite3
import hashlib
from typing import TypedDict, Annotated, Optional
from langchain_community.retrievers import BM25Retriever

from langchain_classic.retrievers import EnsembleRetriever
import requests
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition, InjectedState
from langchain_openrouter import ChatOpenRouter
from langchain.tools import tool
from langchain_tavily import TavilySearch
from langchain_core.messages import BaseMessage, SystemMessage
from langgraph.checkpoint.sqlite import SqliteSaver

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    UnstructuredExcelLoader,
    CSVLoader,
    UnstructuredMarkdownLoader,
    TextLoader,
    UnstructuredPowerPointLoader,
)


# ============================================================
# CONFIG
# ============================================================

load_dotenv()

stock_api_key = os.getenv("ALPHA_VANTAGE_API_KEY")
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")


# ============================================================
# SQLITE CHECKPOINT
# ============================================================

connection_obj = sqlite3.connect(
    database="chatbot.db",
    check_same_thread=False,
)

checkpoint = SqliteSaver(connection_obj)


# ============================================================
# LLM
# ============================================================

# Lower temperature is better for grounded document answers.
llm = ChatOpenRouter(
    model="openrouter/free",
    temperature=0,
)


# ============================================================
# EMBEDDINGS
# ============================================================

embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001"
)


# ============================================================
# OTHER TOOLS
# ============================================================

latest_news = TavilySearch(max_results=5)


@tool
def calculater(expression: str) -> str:
    """
    Calculate mathematical expressions.
    Example: 2000*50/30+10-4
    """
    try:
        allowed = {
            "math": math,
            "abs": abs,
            "min": min,
            "max": max,
            "sum": sum,
        }

        result = eval(
            expression,
            {"__builtins__": {}},
            allowed,
        )

        return str(result)

    except Exception as e:
        return f"Calculation error: {e}"


@tool
def stock_search_price(compeny: str) -> dict:
    """
    Fetch the latest stock price of a company.
    Example: AAPL, TSLA, MSFT
    """

    url = (
        "https://www.alphavantage.co/query"
        f"?function=GLOBAL_QUOTE&symbol={compeny}&apikey={stock_api_key}"
    )

    try:
        res = requests.get(url, timeout=10)
        res.raise_for_status()

        data = res.json()

        if "Error Message" in data:
            return f"Invalid company symbol: {compeny}"

        if "Note" in data:
            return "Alpha Vantage API limit reached. Please try again later."

        return data

    except requests.exceptions.Timeout:
        return "The stock API took too long to respond. Please try again."

    except requests.exceptions.ConnectionError:
        return (
            "Could not connect to the stock API. "
            "Please check your internet connection or try again later."
        )

    except requests.exceptions.HTTPError as e:
        return f"Stock API returned an HTTP error: {e}"

    except requests.exceptions.RequestException as e:
        return f"An error occurred while requesting stock data: {e}"

    except ValueError:
        return "The stock API returned an invalid JSON response."

    except Exception as e:
        return f"Unexpected error: {e}"


@tool
def weather(city: str) -> dict:
    """
    Fetch the current weather information for a given city.
    """

    url = "https://api.openweathermap.org/data/2.5/weather"

    params = {
        "q": city,
        "appid": OPENWEATHER_API_KEY,
        "units": "metric",
    }

    try:
        response = requests.get(
            url,
            params=params,
            timeout=10,
        )

        if response.status_code != 200:
            return f"Unable to fetch weather information for {city}."

        data = response.json()

        return {
            "city": data["name"],
            "country": data["sys"]["country"],
            "temperature": data["main"]["temp"],
            "feels_like": data["main"]["feels_like"],
            "humidity": data["main"]["humidity"],
            "description": data["weather"][0]["description"],
            "wind_speed": data["wind"]["speed"],
        }

    except requests.exceptions.RequestException as e:
        return f"Weather request failed: {e}"


# ============================================================
# VECTOR STORE
# ============================================================

vector_store = Chroma(
    embedding_function=embeddings,
    persist_directory="chroma.db",
)
document_chunks={}

# ============================================================
# DOCUMENT LOADER
# ============================================================

def get_loader(path):
    extension = os.path.splitext(path)[1].lower()

    if extension == ".pdf":
        return PyPDFLoader(path)

    elif extension == ".docx":
        return Docx2txtLoader(path)

    elif extension == ".xlsx":
        return UnstructuredExcelLoader(
            path,
            mode="elements",
        )

    elif extension == ".csv":
        return CSVLoader(path)

    elif extension == ".txt":
        return TextLoader(
            path,
            encoding="utf-8",
        )

    elif extension == ".md":
        return UnstructuredMarkdownLoader(path)

    elif extension == ".pptx":
        return UnstructuredPowerPointLoader(path)

    else:
        raise ValueError(
            f"Unsupported file type: {extension}"
        )


# ============================================================
# DOCUMENT ID
# ============================================================

def make_document_id(path: str) -> str:
    """
    Create a stable ID for a document path.

    Re-ingesting the same path therefore uses the same document ID.
    """
    absolute_path = os.path.abspath(path)

    return hashlib.sha256(
        absolute_path.encode("utf-8")
    ).hexdigest()[:16]


# ============================================================
# DOCUMENT INGESTION
# ============================================================

def ingest_rag_document(path: str):
    """
    Load and index one document.
    """

    loader = get_loader(path)
    document = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100
    )

    chunks = splitter.split_documents(document)

    document_id = make_document_id(path)
    source_name = os.path.basename(path)

    # Delete old chunks for this document
    try:
        vector_store.delete(
            where={"document_id": document_id}
        )
    except Exception:
        pass

    chunk_ids = []

    for index, chunk in enumerate(chunks):

        chunk.metadata["document_id"] = document_id
        chunk.metadata["source"] = source_name
        chunk.metadata["chunk_index"] = index

        if "page" not in chunk.metadata:
            chunk.metadata["page"] = "Unknown"

        chunk_ids.append(
            f"{document_id}_chunk_{index}"
        )

    # Store chunks for BM25
    document_chunks[document_id] = chunks

    # Store embeddings in Chroma
    if chunks:
        vector_store.add_documents(
            documents=chunks,
            ids=chunk_ids
        )

    return {
        "document_id": document_id,
        "source": source_name,
        "chunks": len(chunks)
    }


# ============================================================
# RETRIEVER
# ============================================================

def rag_retriver(
    query: str,
    document_id: str,
    k: int = 5,
):
    """
    Retrieve chunks ONLY from the selected document.
    """

    if not document_id:
        raise ValueError(
            "No active document selected."
        )
    
    chunks=document_chunks.get(document_id)

    if chunks is None:
        raise ValueError(
            f"No chunks found for document ID: {document_id}. "
            "Please ingest the document first."
        )

    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={
            "k": k,
            "filter": {
                "document_id": document_id
            },
        },
    )

    bm25_retriever = BM25Retriever.from_documents(chunks)

    bm25_retriever.k = 5

    hybrid_retriever = EnsembleRetriever(
    retrievers=[
        bm25_retriever,
        retriever
    ],
    weights=[
        0.4,   # BM25
        0.6    # Vector
    ]
            )

    return hybrid_retriever


# ============================================================
# RAG TOOL
# ============================================================

@tool
def rag_tool(
    query: str,
    document_id: Optional[str] = None,
    state: Annotated[dict, InjectedState] = None,
) -> str:
    """
    Retrieve relevant information from the currently selected document.

    The active document ID is automatically taken from graph state.
    """

    active_document_id = document_id

    if not active_document_id and state:
        active_document_id = state.get(
            "active_document_id"
        )

    if not active_document_id:
        return (
            "No active document is selected. "
            "Please select/upload a document first."
        )

    try:
        retriever = rag_retriver(
            query=query,
            document_id=active_document_id,
            k=5,
        )

        documents = retriever.invoke(query)

    except Exception as e:
        return f"RAG retrieval error: {e}"

    if not documents:
        return (
            "No relevant information was found "
            "in the selected document."
        )

    formatted_documents = []

    for index, document in enumerate(
        documents,
        start=1,
    ):

        source = document.metadata.get(
            "source",
            "Unknown source",
        )

        page = document.metadata.get(
            "page",
            "Unknown page",
        )

        chunk_index = document.metadata.get(
            "chunk_index",
            "Unknown",
        )

        formatted_documents.append(
            f"Document chunk {index}\n"
            f"Source: {source}\n"
            f"Page: {page}\n"
            f"Chunk: {chunk_index}\n"
            f"Content:\n{document.page_content}"
        )

    return "\n\n".join(formatted_documents)


# ============================================================
# LANGGRAPH STATE
# ============================================================

class State(TypedDict, total=False):

    messages: Annotated[
        list[BaseMessage],
        add_messages,
    ]

    # Currently selected PDF/document.
    active_document_id: Optional[str]


# ============================================================
# TOOLS
# ============================================================

tools = [
    calculater,
    stock_search_price,
    weather,
    latest_news,
    rag_tool,
]

llm_with_tools = llm.bind_tools(tools)


# ============================================================
# SYSTEM PROMPT
# ============================================================

SYSTEM_PROMPT = """
You are a reliable AI assistant.

TOOL RULES:
- Use the appropriate tool when the question requires it.
- If an active document exists, treat questions related to its content as document questions
  even if the user does not mention "PDF", "document", or "file".
- For such questions, ALWAYS use rag_tool first.
- Use calculator for calculations.
- Use weather for weather queries.
- Use stock_search_price for stock prices.
- Use latest_news for current news.

RAG RULES  use rag_tool if:
if the prompt contains like from pdf of uploaded pdf or document or file or any other similar words then it is a document question.
- For document questions, answer ONLY from rag_tool results.
- Do not use general knowledge to fill gaps.
- Do not add information that is not supported by the retrieved content.
- If the answer is not found, say so clearly.

Answer clearly and concisely.

"""


# ============================================================
# CHATBOT NODE
# ============================================================

def chatbot(state: State):

    system_prompt = SystemMessage(
        content=SYSTEM_PROMPT
    )

    messages = [
        system_prompt,
        *state["messages"],
    ]

    result = llm_with_tools.invoke(
        messages
    )

    return {
        "messages": [result]
    }


# ============================================================
# LANGGRAPH
# ============================================================

graph = StateGraph(State)

graph.add_node(
    "chatbot",
    chatbot,
)

graph.add_node(
    "tools",
    ToolNode(tools),
)

graph.add_edge(
    START,
    "chatbot",
)

graph.add_conditional_edges(
    "chatbot",
    tools_condition,
)

graph.add_edge(
    "tools",
    "chatbot",
)

app = graph.compile(
    checkpointer=checkpoint,
)


# ============================================================
# HELPER FUNCTION FOR YOUR UI
# ============================================================

def create_initial_state(
    user_message: str,
    document_id: Optional[str] = None,
) -> State:
    """
    Build the state for a chat request.

    Example:

        info = ingest_rag_document("my_file.pdf")

        state = create_initial_state(
            "What is the B.Tech fee?",
            info["document_id"],
        )

        result = app.invoke(
            state,
            config={
                "configurable": {
                    "thread_id": "user-1"
                }
            },
        )
    """

    return {
        "messages": [
            {
                "role": "user",
                "content": user_message,
            }
        ],
        "active_document_id": document_id,
    }


# ============================================================
# THREAD IDS
# ============================================================

def get_threads():

    threads = set()

    for cp in checkpoint.list(None):

        threads.add(
            cp.config["configurable"]["thread_id"]
        )

    return list(threads)
