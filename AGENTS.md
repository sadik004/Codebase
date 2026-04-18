# Instructions for AI Agents

Welcome to the Agentic RAG System repository. As an AI Agent working on this codebase, please adhere to the following guidelines and architectural conventions.

## Project Scope
This project is an Agentic RAG system built with CrewAI and the Google Gemini API. It is aimed at smart document analysis, debugging, and autonomous code generation.

## Directory Structure
- `docs/`: Place source documents for parsing here.
- `data_ingestion.py`: The entry point for parsing, chunking, and embedding documents.
- `chroma_db/`: Default directory for the local ChromaDB storage (this may be auto-generated).

## Core Libraries & APIs
- **CrewAI**: Use for defining agents, tasks, and crews.
- **Google Gemini API**: Use for LLM models (`gemini-pro`, etc.) and embeddings via `langchain-google-genai`.
- **LlamaParse**: Primary tool for document ingestion.
- **ChromaDB**: Use for local vector storage.

## Coding Conventions
1. **Environment Variables**: Never hardcode API keys. Always use `.env` files and `python-dotenv` or `os.environ` to load them (e.g., `GOOGLE_API_KEY`, `LLAMA_CLOUD_API_KEY`).
2. **Type Hinting**: Use Python type hints where applicable.
3. **Docstrings**: Document your classes and functions.
4. **Modularity**: Keep agent definitions, tool definitions, and orchestration logic clearly separated.

## Important Notes
- When working on data ingestion, ensure robust error handling, especially for API rate limits and empty document directories.
- Ensure that ChromaDB is configured to store data persistently to the local disk.
