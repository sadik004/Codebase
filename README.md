# Agentic RAG System

## Description
This repository contains an Agentic RAG (Retrieval-Augmented Generation) system built with CrewAI and Google Gemini API. The system is designed for smart document analysis, debugging, and autonomous code generation.

It leverages LlamaParse for accurate document parsing, LangChain for chunking and orchestration, and ChromaDB for local vector storage of embeddings.

## Key Technologies
* **CrewAI**: Orchestrating agentic workflows.
* **Google Gemini API**: Providing LLM reasoning and embedding generation.
* **LlamaParse**: Parsing complex documents accurately.
* **ChromaDB**: Local vector database for semantic search.
* **LangChain**: Tools and wrappers for data processing and vector DB integration.

## Setup Instructions

1. Clone the repository.
2. Initialize the Python environment.
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Set up your environment variables by creating a `.env` file:
   ```
   GOOGLE_API_KEY=your_google_api_key
   LLAMA_CLOUD_API_KEY=your_llama_cloud_api_key
   ```
5. Add your documents to the `docs/` directory.
6. Run the ingestion script:
   ```bash
   python data_ingestion.py
   ```

## Architecture
1. **Data Ingestion**: Documents in the `docs/` folder are processed by `LlamaParse`.
2. **Chunking & Embedding**: Parsed text is chunked into logical segments. The Google Gemini API is used to generate embeddings.
3. **Vector Database**: Embeddings and chunk metadata are stored in a local ChromaDB instance.
4. **Agentic Interaction**: CrewAI agents query the ChromaDB instance to retrieve context and generate responses for document analysis, debugging, or code generation tasks.
