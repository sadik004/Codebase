import os
import glob
from pathlib import Path
import nest_asyncio

# Load environment variables
from dotenv import load_dotenv

# Document parsers and splitters
from llama_parse import LlamaParse
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

# Embeddings and Vector Store
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma

# Support nested async loops for LlamaParse (often needed in environments with existing loops)
nest_asyncio.apply()

# Initialize environment
load_dotenv()

# Constants
DOCS_DIR = "docs"
CHROMA_DB_DIR = "./chroma_db"
COLLECTION_NAME = "agentic_rag_knowledge"

def get_llama_parser():
    """Initializes the LlamaParse object."""
    return LlamaParse(result_type="markdown")

def process_markdown_content(content, metadata):
    """Splits markdown content based on headers."""
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    chunks = markdown_splitter.split_text(content)

    # Add base metadata to each chunk
    for chunk in chunks:
        chunk.metadata.update(metadata)
    return chunks

def process_raw_code_content(content, metadata):
    """Splits raw code content using RecursiveCharacterTextSplitter."""
    code_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = code_splitter.create_documents([content])

    # Add base metadata to each chunk
    for chunk in chunks:
        chunk.metadata.update(metadata)
    return chunks

def main(test_mode=False):
    print("Starting data ingestion process...")

    # Verify environment keys
    llama_key = os.getenv("LLAMA_CLOUD_API_KEY")
    google_key = os.getenv("GOOGLE_API_KEY")

    if not test_mode and (not llama_key or not google_key or llama_key == "test_llama_key"):
        print("WARNING: Using dummy/missing API keys in production mode. Set valid keys in .env")

    # If in test mode, we might just want to print what would be done to avoid real API calls with dummy keys

    # Gather all supported files
    supported_extensions = ['*.pdf', '*.md', '*.txt', '*.py', '*.js']
    files_to_process = []

    for ext in supported_extensions:
        files_to_process.extend(glob.glob(os.path.join(DOCS_DIR, ext)))

    print(f"Found {len(files_to_process)} files to process.")

    all_chunks = []
    parser = get_llama_parser()

    for file_path in files_to_process:
        print(f"Processing file: {file_path}")
        ext = Path(file_path).suffix.lower()
        metadata = {"source": file_path}

        try:
            if ext in ['.pdf', '.md', '.txt']:
                # Use LlamaParse for documents
                if test_mode:
                    print(f"[TEST] Mocking LlamaParse for {file_path}")
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                else:
                    parsed_docs = parser.load_data(file_path)
                    # LlamaParse returns a list of Document objects. Combine the text for simplicity
                    content = "\n\n".join([doc.text for doc in parsed_docs])

                chunks = process_markdown_content(content, metadata)
                all_chunks.extend(chunks)

            elif ext in ['.py', '.js']:
                # Read raw text for code
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()

                chunks = process_raw_code_content(content, metadata)
                all_chunks.extend(chunks)

        except Exception as e:
            print(f"Error processing {file_path}: {e}")

    print(f"Generated {len(all_chunks)} chunks total.")

    if not all_chunks:
        print("No chunks generated. Exiting.")
        return

    if test_mode:
        print("\n[TEST] Sample Chunk:")
        print(all_chunks[0].page_content[:200])
        print(f"Metadata: {all_chunks[0].metadata}")
        print("\n[TEST] Skipping ChromaDB insertion with actual Google API calls.")
        return

    print("Initializing Google Generative AI Embeddings...")
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    print("Storing chunks in ChromaDB...")
    vector_store = Chroma.from_documents(
        documents=all_chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DB_DIR,
        collection_name=COLLECTION_NAME
    )
    print("Data ingestion complete!")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Ingest docs into ChromaDB.")
    parser.add_argument("--test", action="store_true", help="Run in test mode with mock API calls.")
    args = parser.parse_args()

    main(test_mode=args.test)
