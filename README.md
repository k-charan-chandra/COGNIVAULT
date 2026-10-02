# 🧠 CogniVault

> An intelligent document-based AI assistant that combines Retrieval-Augmented Generation, hybrid search, and agentic workflows to provide context-aware answers from user-provided documents.

## 📌 Overview

CogniVault is an AI-powered knowledge assistant designed to interact with user-provided documents and generate context-aware responses.

The system combines document ingestion, intelligent chunking, vector-based retrieval, keyword-based retrieval, and LLM reasoning to ground responses in relevant information.

Instead of relying solely on an LLM's pre-trained knowledge, CogniVault retrieves relevant information from the user's documents before generating an answer.

---

## ✨ Key Features

* 📄 **Document Ingestion** — Upload and process PDF documents.
* ✂️ **Intelligent Chunking** — Splits documents into manageable overlapping chunks.
* 🔎 **Hybrid Retrieval** — Combines semantic vector search with BM25 keyword search.
* 🧠 **RAG Pipeline** — Grounds LLM responses using retrieved document context.
* 🤖 **Agentic Workflow** — Uses LangGraph to coordinate the chatbot and tool execution.
* 🌐 **Web Search** — Retrieves current information when external knowledge is required.
* 🧮 **Tool Calling** — Supports specialized tools such as calculations and external data retrieval.
* 💾 **Persistent State** — Maintains conversational state using SQLite-based checkpointing.
* ⚡ **Streamlit Interface** — Provides an interactive chat-based frontend.
* 🔐 **Environment-based Configuration** — API credentials are managed through environment variables.

---

## 🏗️ Architecture

```text
                         ┌─────────────────────┐
                         │     Streamlit UI    │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │    LangGraph Agent  │
                         └──────────┬──────────┘
                                    │
                         ┌──────────▼──────────┐
                         │         LLM         │
                         │  Tool Decision /    │
                         │  Response Generation│
                         └──────────┬──────────┘
                                    │
                   ┌────────────────┼────────────────┐
                   │                │                │
                   ▼                ▼                ▼
              RAG Tool          Web Search       Other Tools
                   │
                   ▼
          ┌─────────────────┐
          │ Hybrid Retriever│
          └────────┬────────┘
                   │
             ┌─────┴─────┐
             ▼           ▼
       Vector Search    BM25
             │           │
             └─────┬─────┘
                   ▼
             Relevant Chunks
                   │
                   ▼
                 LLM
                   │
                   ▼
             Final Response
```

---

## 🔄 RAG Workflow

### 1. Document Upload

The user uploads a PDF through the Streamlit interface.

### 2. Document Loading

The PDF is loaded and converted into LangChain document objects.

### 3. Chunking

The document is divided into smaller overlapping chunks using a recursive text splitter.

```text
PDF
 │
 ├── Chunk 1
 ├── Chunk 2
 ├── Chunk 3
 ├── ...
 └── Chunk N
```

The overlap helps preserve contextual continuity between neighboring chunks.

### 4. Embedding Generation

Each chunk is converted into a numerical vector using the configured embedding model.

### 5. Vector Storage

The generated embeddings are stored in Chroma for semantic similarity search.

### 6. Hybrid Retrieval

When the user asks a document-related question, CogniVault performs two retrieval strategies:

**Semantic Retrieval**

Finds chunks that are conceptually similar to the question.

**BM25 Retrieval**

Finds chunks based on keyword and term relevance.

The results are combined using an ensemble retriever.

```text
User Query
    │
    ├───────────────┐
    ▼               ▼
Vector Search     BM25
    │               │
    └───────┬───────┘
            ▼
     Ensemble Retrieval
            │
            ▼
     Relevant Documents
```

### 7. Context Injection

The retrieved chunks are returned to the agent as context.

The LLM uses this retrieved information to generate the final answer.

---

## 🤖 Agent Workflow

CogniVault uses LangGraph to manage the agent workflow.

```text
START
  │
  ▼
Chatbot
  │
  ▼
Does the LLM need a tool?
  │
 ┌┴───────────────┐
 │                │
No               Yes
 │                │
 ▼                ▼
Final Answer    Tool Node
                  │
                  ▼
               Chatbot
                  │
                  ▼
             Final Answer
```

The agent can decide when to use specialized tools rather than treating every question as a simple LLM-generation task.

---

## 🧩 Technology Stack

| Component             | Technology                   |
| --------------------- | ---------------------------- |
| Frontend              | Streamlit                    |
| Agent Framework       | LangGraph                    |
| LLM Framework         | LangChain                    |
| LLM                   | OpenRouter-compatible model  |
| Embeddings            | Configurable embedding model |
| Vector Database       | Chroma                       |
| Keyword Retrieval     | BM25                         |
| Document Processing   | LangChain document loaders   |
| State / Checkpointing | SQLite                       |
| Programming Language  | Python                       |

---

## 📁 Project Structure

```text
COGNIVAULT/
│
├── frontend/
│   └── streamlit application
│
├── backend/
│   ├── chatbot/
│   ├── RAG/
│   ├── tools/
│   └── ...
│
├── chroma.db/
│
├── chatbot.db
│
├── .env
├── .gitignore
├── requirements.txt
└── README.md
```

> Update this structure to exactly match the repository before publishing the README.

---

## ⚙️ Installation

### Clone the repository

```bash
git clone https://github.com/k-charan-chandra/COGNIVAULT.git
cd COGNIVAULT
```

### Create a virtual environment

```bash
python -m venv .venv
```

### Activate the environment

**Windows**

```bash
.venv\Scripts\activate
```

### Install dependencies

```bash
pip install -r requirements.txt
```

---

## 🔑 Environment Variables

Create a `.env` file in the project root.

```env
OPENROUTER_API_KEY=your_key
TAVILY_API_KEY=your_key
GOOGLE_API_KEY=your_key
```

Only include the variables that are actually required by the implementation.

**Never commit `.env` or API keys to GitHub.**

---

## ▶️ Running the Application

Start the Streamlit application with:

```bash
streamlit run <your_streamlit_file>.py
```

Then open the URL displayed by Streamlit in your browser.

---

## 💡 Example

```text
User:
"What is the B.Tech fee structure?"

        ↓

Agent identifies this as a document question

        ↓

RAG Tool

        ↓

Hybrid Retrieval
 ┌──────────────┐
 │ Vector Search│
 │     +        │
 │     BM25     │
 └──────────────┘

        ↓

Relevant PDF chunks

        ↓

LLM

        ↓

Grounded Answer
```

---

## 🎯 Why CogniVault?

Traditional LLM applications can generate answers without considering the user's private documents.

CogniVault addresses this by introducing a retrieval layer between the user's question and the language model.

This allows the application to:

* work with private document knowledge,
* retrieve relevant context,
* combine semantic and lexical search,
* use external tools when required,
* and maintain conversational state.

---

## 🚀 Future Improvements

Potential improvements include:

* Reranking retrieved documents using a cross-encoder.
* Support for additional document formats.
* Streaming responses.
* User authentication and document-level access control.
* Persistent document metadata.
* Improved retrieval evaluation.
* Conversation analytics.
* Production deployment with Docker.
* Scalable vector database deployment.
* Background document ingestion.
* generation of Document id will be improved  
