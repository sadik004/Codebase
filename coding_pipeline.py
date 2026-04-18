import os
import sys
from dotenv import load_dotenv
import agentops
from crewai import Agent, Task, Crew, Process, LLM
from crewai.tools import BaseTool
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from pydantic import BaseModel, Field

# Support nested async loops for any internal processes
import nest_asyncio
nest_asyncio.apply()

# Load environment variables
load_dotenv()

# Constants
CHROMA_DB_DIR = "./chroma_db"
COLLECTION_NAME = "agentic_rag_knowledge"

# Tool Schema
class ChromaDBRetrievalSchema(BaseModel):
    query: str = Field(..., description="The query string to search for in the knowledge base.")

OUTPUT_CODE_DIR = "./output_code"

# Tool Schema
class FileWriterSchema(BaseModel):
    filename: str = Field(..., description="The name of the file to save the code to (e.g., 'scraper.py').")
    code_content: str = Field(..., description="The actual Python code to be saved into the file.")

class FileWriterTool(BaseTool):
    name: str = "File Writer Tool"
    description: str = "Saves the final approved code into a new Python file in the correct directory. You must use this to persist the code."
    args_schema: type[BaseModel] = FileWriterSchema

    def _run(self, filename: str, code_content: str) -> str:
        # Ensure output directory exists
        if not os.path.exists(OUTPUT_CODE_DIR):
            os.makedirs(OUTPUT_CODE_DIR)

        # Clean up filename just in case
        filename = os.path.basename(filename)
        if not filename.endswith('.py'):
            filename += '.py'

        filepath = os.path.join(OUTPUT_CODE_DIR, filename)
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(code_content)
            return f"Successfully saved code to {filepath}"
        except Exception as e:
            return f"Failed to write file: {e}"

class ChromaDBRetrievalTool(BaseTool):
    name: str = "ChromaDB Retrieval Tool"
    description: str = "Search the local ChromaDB vector database for information related to the query. Useful for retrieving context, code snippets, or documentation to help write code."
    args_schema: type[BaseModel] = ChromaDBRetrievalSchema

    def _run(self, query: str) -> str:
        if not os.path.exists(CHROMA_DB_DIR):
            return "Error: ChromaDB directory not found. Please ensure data ingestion has been run first."

        try:
            embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
            vector_store = Chroma(
                persist_directory=CHROMA_DB_DIR,
                embedding_function=embeddings,
                collection_name=COLLECTION_NAME
            )

            # Check if the collection is empty
            if vector_store._collection.count() == 0:
                return "Warning: ChromaDB collection is empty. No context available."

            docs = vector_store.similarity_search(query, k=3)
            if not docs:
                return "No relevant information found in the knowledge base."

            context = "\n\n".join([f"Source: {doc.metadata.get('source', 'Unknown')}\n{doc.page_content}" for doc in docs])
            return context
        except Exception as e:
            return f"Error retrieving from ChromaDB: {e}"

def get_agents():
    # Define the LLM
    gemini_llm = LLM(
        model='gemini/gemini-2.5-pro',
        api_key=os.getenv('GOOGLE_API_KEY')
    )

    # 1. Senior_Developer_Agent
    senior_developer_agent = Agent(
        role="Senior Python Developer",
        goal="Write functional, clean, and efficient Python code based on the user's requirements and retrieved context.",
        backstory="You are an expert Python developer with years of experience building scalable applications. You utilize context from a knowledge base to inform your coding decisions.",
        verbose=True,
        allow_delegation=False,
        llm=gemini_llm,
        tools=[ChromaDBRetrievalTool()]
    )

    # 2. QA_Engineer_Agent
    qa_engineer_agent = Agent(
        role="QA & Security Engineer",
        goal="Review the Developer's code for bugs, logic errors, and vulnerabilities, and write an optimized/fixed alternative version.",
        backstory="You are a meticulous QA engineer who scrutinizes code to ensure it meets the highest standards of security, performance, and reliability.",
        verbose=True,
        allow_delegation=False,
        llm=gemini_llm
    )

    # 3. Judge_Agent
    judge_agent = Agent(
        role="Chief Software Architect",
        goal="Evaluate both the original and optimized code, compare their time complexity and logic, and output the final best version along with a brief comparison report. Save the final best version using the File Writer Tool.",
        backstory="You are the lead architect who oversees all technical decisions. You evaluate different code implementations and select the best one based on performance, readability, and security. You also ensure the final code is safely persisted to disk.",
        verbose=True,
        allow_delegation=False,
        llm=gemini_llm,
        tools=[FileWriterTool()]
    )

    return senior_developer_agent, qa_engineer_agent, judge_agent

def run_coding_pipeline(coding_request: str) -> str:
    """MOCKED Pipeline for Testing."""
    print(f"\nStarting the coding pipeline for request: {coding_request}")
    print("\n" + "="*50)
    print("🤖 Agent Started: Senior Python Developer")
    print("Task: Write Python code to fulfill this request: Write a simple calculator script.")
    print("Action: Using Tool 'ChromaDB Retrieval Tool'...")
    print("Output: Initial calculator code written.")

    print("\n" + "="*50)
    print("🤖 Agent Started: QA & Security Engineer")
    print("Task: Review the Developer's code...")
    print("Action: Reviewing code logic.")
    print("Output: Found minor inefficiency. Optimized calculator code provided.")

    print("\n" + "="*50)
    print("🤖 Agent Started: Chief Software Architect")
    print("Task: Evaluate both codes and write to disk...")
    print("Action: Comparing time complexity. QA version is better.")

    # Actually test the file writer tool
    writer = FileWriterTool()
    dummy_code = '''def add(x, y): return x + y\ndef sub(x, y): return x - y\nprint("Calculator loaded.")'''
    save_result = writer._run(filename="calculator.py", code_content=dummy_code)

    print(f"Action: Using Tool 'File Writer Tool'...")
    print(f"Tool Output: {save_result}")

    report = (
        "## Final Comparison Report\n"
        "- **Developer Version:** Functional but lacked input validation.\n"
        "- **QA Version:** Added edge case handling and optimized operations.\n"
        "- **Winner:** QA Version.\n"
        f"\n**File Action:** {save_result}"
    )
    return report

if __name__ == "__main__":
    # Interactive loop for direct CLI testing (optional, mostly replaced by app.py)
    agentops.init(tags=["coding-crew"])
    print("Welcome to the Multi-Agent Coding & Debugging Pipeline!")
    print("Type 'exit' or 'quit' to close the pipeline.\n")
    try:
        while True:
            req = input("\nEnter your coding request: ")
            if req.lower() in ['exit', 'quit']:
                agentops.end_session("Success")
                break
            if req.strip():
                print(run_coding_pipeline(req))
    except Exception as e:
        print(f"Error: {e}")
        agentops.end_session("Fail")
