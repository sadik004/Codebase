import os
import sys

from dotenv import load_dotenv

# Langchain & CrewAI
from crewai import Agent, Task, Crew, Process, LLM
from crewai.tools import tool
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.tools import DuckDuckGoSearchRun

# Support nested async loops just in case
import nest_asyncio
nest_asyncio.apply()

# Initialize environment variables
load_dotenv()

# Constants
CHROMA_DB_DIR = "./chroma_db"
COLLECTION_NAME = "agentic_rag_knowledge"

# Initialize Google Generative AI (Gemini) LLM via CrewAI's LLM wrapper
# Note: Since CrewAI uses LiteLLM under the hood, we map our GOOGLE_API_KEY to the api_key param
llm = LLM(
    model="gemini/gemini-1.5-pro",
    api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0, # Low temperature for accurate, deterministic reasoning
    max_tokens=2048
)

# ---------------------------------------------------------
# Custom Tools definition
# ---------------------------------------------------------

@tool("ChromaDB_Codebase_Search")
def chromadb_search_tool(query: str) -> str:
    """
    Searches the local ChromaDB vector database which contains the internal documentation
    and codebase files. Use this tool when the query is about our internal codebase, documents,
    project-specific logic, or internal architecture.
    """
    print(f"\n[Tool Execution] ChromaDB_Codebase_Search called with query: '{query}'")

    # Fail gracefully if keys aren't set correctly
    google_key = os.getenv("GOOGLE_API_KEY")
    if not google_key or google_key == "test_google_key":
        return "ERROR: Valid GOOGLE_API_KEY is required to generate embeddings for the search query."

    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        vector_store = Chroma(
            persist_directory=CHROMA_DB_DIR,
            embedding_function=embeddings,
            collection_name=COLLECTION_NAME
        )

        # Perform similarity search
        docs = vector_store.similarity_search(query, k=4)

        if not docs:
            return "No relevant internal documents or code were found in the local database."

        results = []
        for i, doc in enumerate(docs):
            source = doc.metadata.get('source', 'Unknown Source')
            results.append(f"--- Document {i+1} (Source: {source}) ---\n{doc.page_content}\n")

        return "\n".join(results)
    except Exception as e:
        return f"Error executing ChromaDB search: {str(e)}"

@tool("Web_Search")
def web_search_tool(query: str) -> str:
    """
    Searches the internet using DuckDuckGo. Use this tool when the query is about general
    knowledge, external libraries, latest news, public APIs, or concepts that wouldn't be
    found in the internal private codebase.
    """
    print(f"\n[Tool Execution] Web_Search called with query: '{query}'")
    try:
        search = DuckDuckGoSearchRun()
        return search.run(query)
    except Exception as e:
        return f"Error executing Web search: {str(e)}"

# ---------------------------------------------------------
# Agent Definitions
# ---------------------------------------------------------

router_agent = Agent(
    role="Senior Codebase Analyst & Router",
    goal="Analyze the user's query and intelligently decide whether to search the internal codebase/documents using ChromaDB or to search the internet using the Web Search tool. Retrieve the necessary context and provide it to the synthesizer.",
    backstory=(
        "You are a highly experienced software architect and technical analyst. You possess a deep understanding "
        "of when to look at internal, proprietary documentation and code versus when to consult the broader internet "
        "for external knowledge or general library documentation. You are meticulous in retrieving accurate "
        "and highly relevant context to answer complex technical queries."
    ),
    verbose=True,
    allow_delegation=False,
    llm=llm,
    tools=[chromadb_search_tool, web_search_tool]
)

synthesizer_agent = Agent(
    role="Technical Content Synthesizer",
    goal="Take the raw context retrieved by the Router Agent and synthesize it into a clear, accurate, and well-reasoned technical explanation or solution.",
    backstory=(
        "You are an expert technical writer and developer advocate. You excel at taking raw code snippets, "
        "documentation fragments, or web search results and turning them into highly readable, cohesive, "
        "and actionable explanations. You structure your output logically, often using code blocks and bullet points."
    ),
    verbose=True,
    allow_delegation=False,
    llm=llm
)

# ---------------------------------------------------------
# Main Execution Loop
# ---------------------------------------------------------

def process_query(user_query: str):
    """Creates tasks for the query and executes the crew."""

    routing_task = Task(
        description=(
            f"Analyze the user query: '{user_query}'.\n"
            "1. Determine if the query is asking about internal codebase logic, proprietary architecture, or specific internal documents. If so, use the 'ChromaDB_Codebase_Search' tool.\n"
            "2. If the query asks about general knowledge, external libraries, common programming concepts, or recent news, use the 'Web_Search' tool.\n"
            "3. Retrieve the most relevant and comprehensive information based on your tool choice.\n"
            "4. Return the raw retrieved data along with a brief note on where it was sourced from."
        ),
        expected_output="Raw context data retrieved from either the local codebase or the web, along with a note identifying the source.",
        agent=router_agent
    )

    synthesis_task = Task(
        description=(
            f"Based on the context retrieved by the Senior Codebase Analyst regarding the user query: '{user_query}', "
            "generate a final, polished response. The response should be well-reasoned, clearly formatted (using Markdown where appropriate), "
            "and directly answer the user's question without hallucinating information outside the provided context."
        ),
        expected_output="A clear, well-reasoned, and formatted technical explanation or solution directly answering the user query.",
        agent=synthesizer_agent
    )

    rag_crew = Crew(
        agents=[router_agent, synthesizer_agent],
        tasks=[routing_task, synthesis_task],
        process=Process.sequential,
        verbose=True
    )

    print("\n==================================================")
    print(f"Executing Agentic RAG Pipeline for query: '{user_query}'")
    print("==================================================\n")

    result = rag_crew.kickoff()
    return result

if __name__ == "__main__":
    print("=========================================================================")
    print("               Agentic RAG System Interactive Terminal                   ")
    print("Type 'exit' or 'quit' to terminate the session.")
    print("=========================================================================\n")

    while True:
        try:
            user_input = input("\nEnter your query: ").strip()

            if user_input.lower() in ['exit', 'quit']:
                print("Exiting Agentic RAG System. Goodbye!")
                break

            if not user_input:
                continue

            final_response = process_query(user_input)

            print("\n" + "="*50)
            print("FINAL SYNTHESIZED RESPONSE:")
            print("="*50)
            print(final_response)

        except KeyboardInterrupt:
            print("\nExiting Agentic RAG System. Goodbye!")
            break
        except EOFError:
            print("\nExiting Agentic RAG System (EOF).")
            break
        except Exception as e:
            print(f"\nAn error occurred during execution: {e}")
