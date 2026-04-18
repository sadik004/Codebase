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

def main():
    # Initialize AgentOps
    agentops.init(tags=["coding-crew"])

    print("Welcome to the Multi-Agent Coding & Debugging Pipeline!")
    print("Type 'exit' or 'quit' to close the pipeline.\n")

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
        goal="Evaluate both the original and optimized code, compare their time complexity and logic, and output the final best version along with a brief comparison report.",
        backstory="You are the lead architect who oversees all technical decisions. You evaluate different code implementations and select the best one based on performance, readability, and security.",
        verbose=True,
        allow_delegation=False,
        llm=gemini_llm
    )

    try:
        while True:
            coding_request = input("\nEnter your coding request (e.g., 'Write a Python script to scrape a website'): ")

            if coding_request.lower() in ['exit', 'quit']:
                print("Exiting pipeline. Goodbye!")
                agentops.end_session("Success")
                break

            if not coding_request.strip():
                print("Request cannot be empty. Please try again.")
                continue

            print("\nStarting the coding pipeline...")

            # Define Tasks
            developer_task = Task(
                description=f"Write Python code to fulfill this request: {coding_request}. Use your tools to retrieve any helpful context.",
                expected_output="Functional Python code fulfilling the user's request, along with any necessary explanations.",
                agent=senior_developer_agent
            )

            qa_task = Task(
                description="Review the code provided by the Senior Developer. Identify any bugs, logic errors, inefficiencies, or vulnerabilities. Provide an optimized and fixed alternative version of the code.",
                expected_output="A review of the original code, listing any issues, followed by the complete optimized/fixed Python code.",
                agent=qa_engineer_agent
            )

            judge_task = Task(
                description="Evaluate the original code (from the Developer) and the optimized code (from the QA Engineer). Compare their time complexity, logic, and overall quality. Output the final best version of the code and a brief comparison report.",
                expected_output="A final comparison report evaluating both versions, followed by the conclusive best version of the Python code.",
                agent=judge_agent,
                context=[developer_task, qa_task]
            )

            # Assemble Crew
            coding_crew = Crew(
                agents=[senior_developer_agent, qa_engineer_agent, judge_agent],
                tasks=[developer_task, qa_task, judge_task],
                process=Process.sequential,
                verbose=True
            )

            # Execute
            result = coding_crew.kickoff()

            print("\n" + "="*50)
            print("FINAL JUDGE OUTPUT:")
            print("="*50)
            print(result)

    except KeyboardInterrupt:
        print("\nPipeline interrupted by user.")
        agentops.end_session("Fail")
    except Exception as e:
        print(f"\nAn error occurred: {e}")
        agentops.end_session("Fail")

if __name__ == "__main__":
    main()
