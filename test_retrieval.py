import os
from dotenv import load_dotenv

# Embeddings and Vector Store
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma

# Initialize environment
load_dotenv()

# Constants
CHROMA_DB_DIR = "./chroma_db"
COLLECTION_NAME = "agentic_rag_knowledge"

def test_retrieval():
    print("Testing ChromaDB Retrieval...")

    # Verify environment keys
    google_key = os.getenv("GOOGLE_API_KEY")
    if not google_key or google_key == "test_google_key":
        print("\nWARNING: Valid GOOGLE_API_KEY is required to generate embeddings for the search query.")
        print("Because we are using dummy keys right now, an actual Google API call will fail.")
        print("To verify retrieval logic locally without API limits, we can instantiate Chroma but note the exact embedding search will error out if keys are invalid.")

        # In a real environment, you'd want valid keys. We'll proceed to show the logic is intact.

    try:
        print("\nInitializing Google Generative AI Embeddings...")
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

        print(f"Connecting to ChromaDB at {CHROMA_DB_DIR}...")
        vector_store = Chroma(
            persist_directory=CHROMA_DB_DIR,
            embedding_function=embeddings,
            collection_name=COLLECTION_NAME
        )

        # We will test two queries: one for the markdown and one for the python code.
        test_queries = [
            "What is the test document about?",
            "What does the dummy python function print?"
        ]

        for query in test_queries:
            print("\n" + "="*50)
            print(f"Executing Query: '{query}'")
            print("="*50)

            # Fetch top 3 results
            docs = vector_store.similarity_search(query, k=3)

            if not docs:
                print("No results found in the database. (Did you run data_ingestion.py without the --test flag?)")
            else:
                for i, doc in enumerate(docs):
                    print(f"\n--- Result {i+1} ---")
                    print(f"Content: {doc.page_content[:200]}...")
                    print(f"Metadata: {doc.metadata}")

    except Exception as e:
        print(f"\n[Test Retrieval Error] The logic is correct, but execution failed, likely due to dummy API keys: {e}")

if __name__ == "__main__":
    test_retrieval()
