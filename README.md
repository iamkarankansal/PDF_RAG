# PDF RAG – Retrieval-Augmented Generation from PDFs

A FastAPI-based application that lets users upload PDF files and ask context-based questions. It uses LangChain, OpenAI Embeddings, Qdrant, and Valkey to enable efficient retrieval-augmented generation (RAG).

## Features

- Upload and process PDF documents
- Extract and split text using LangChain
- Generate semantic embeddings with OpenAI
- Store and search document chunks using Qdrant
- Use Valkey (Redis-compatible) for background task queuing
- Ask questions and get relevant answers from uploaded PDFs

## Tech Stack

- Python
- FastAPI
- LangChain
- OpenAI Embeddings
- Qdrant (Vector Database)
- Valkey (Task Queue / In-memory Store)
- Redis Queue (`rq`)

## Getting Started

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/pdf-rag.git
   cd pdf-rag
