import os
import sys
from dotenv import load_dotenv
import agentops
from crewai import Agent, Task, Crew, Process, LLM
from coding_pipeline import ChromaDBRetrievalTool, FileWriterTool

# Support nested async loops for any internal processes
import nest_asyncio
nest_asyncio.apply()

# Load environment variables
load_dotenv()

def get_enterprise_agents():
    # Define the LLM
    gemini_llm = LLM(
        model='gemini/gemini-2.5-pro',
        api_key=os.getenv('GOOGLE_API_KEY')
    )

    # 1. Researcher Agent
    researcher_agent = Agent(
        role="Principal Researcher",
        goal="Query the vector database for the best practices, patterns, and existing documentation relevant to the user's request.",
        backstory="You are an elite researcher who always consults internal knowledge bases before any engineering work begins. You gather critical context to ensure all code adheres to company standards.",
        verbose=True,
        allow_delegation=False,
        llm=gemini_llm,
        tools=[ChromaDBRetrievalTool()]
    )

    # 2. Planner Agent
    planner_agent = Agent(
        role="Systems Architect & Planner",
        goal="Take the research data and user request and break it down into a clear, step-by-step technical execution plan.",
        backstory="You are a seasoned software architect. You take raw requirements and research, and structure them into a foolproof blueprint for developers to follow.",
        verbose=True,
        allow_delegation=False,
        llm=gemini_llm
    )

    # 3. Senior Developer Agent
    senior_developer_agent = Agent(
        role="Senior Python Developer",
        goal="Write functional, clean, and efficient Python code based strictly on the Architect's execution plan.",
        backstory="You are an expert Python developer with years of experience building scalable applications. You follow plans meticulously and write beautiful code.",
        verbose=True,
        allow_delegation=False,
        llm=gemini_llm
    )

    # 4. QA Reviewer Agent
    qa_reviewer_agent = Agent(
        role="QA & Security Engineer",
        goal="Critique the Developer's code for bugs, logic errors, and vulnerabilities, and output the final optimized version. Use the File Writer Tool to save the final approved code.",
        backstory="You are a meticulous QA engineer who scrutinizes code. You debate with developers internally to ensure the final product meets the highest standards of security, performance, and reliability. You also persist the final code to disk.",
        verbose=True,
        allow_delegation=False,
        llm=gemini_llm,
        tools=[FileWriterTool()]
    )

    return researcher_agent, planner_agent, senior_developer_agent, qa_reviewer_agent

def run_enterprise_pipeline(coding_request: str) -> str:
    """Executes the Enterprise Agentic RAG Pipeline with Collaborative Debate."""
    print(f"\nStarting the Enterprise coding pipeline for request: {coding_request}")

    researcher_agent, planner_agent, senior_developer_agent, qa_reviewer_agent = get_enterprise_agents()

    # Define Tasks
    research_task = Task(
        description=f"Query the knowledge base using the ChromaDB Retrieval Tool for best practices and context related to this request: {coding_request}",
        expected_output="A detailed summary of relevant best practices, code snippets, and architectural guidelines retrieved from the database.",
        agent=researcher_agent
    )

    planning_task = Task(
        description=f"Analyze the original request ('{coding_request}') alongside the research gathered. Create a detailed, step-by-step technical execution plan.",
        expected_output="A structured markdown document outlining the exact steps, components, and logic required to build the solution.",
        agent=planner_agent,
        context=[research_task]
    )

    developer_task = Task(
        description="Write the initial Python code strictly following the Architect's step-by-step plan. Ensure the code is complete and functional.",
        expected_output="The complete, functional Python script fulfilling the requirements.",
        agent=senior_developer_agent,
        context=[planning_task]
    )

    # Collaborative Debate: QA critiques the Developer's output, fixes it, and saves it.
    qa_debate_task = Task(
        description="Critique the Developer's code. Identify inefficiencies, missing edge cases, or security flaws. Refactor and fix the code to resolve these issues. Once the code is perfect, save it to a dynamically chosen, appropriate filename using the File Writer Tool. Output a final report summarizing the original flaws, the fixes applied, and the path to the saved file.",
        expected_output="A comprehensive QA report detailing the critique and fixes, along with confirmation of the file successfully saved to disk.",
        agent=qa_reviewer_agent,
        context=[developer_task]
    )

    # Configure persistent memory for the Crew
    embedder_config = {
        'provider': 'google-generativeai',
        'config': {
            'model': 'models/embedding-001',
            'api_key': os.getenv('GOOGLE_API_KEY')
        }
    }

    # Assemble Enterprise Crew with Persistent Memory
    enterprise_crew = Crew(
        agents=[researcher_agent, planner_agent, senior_developer_agent, qa_reviewer_agent],
        tasks=[research_task, planning_task, developer_task, qa_debate_task],
        process=Process.sequential,
        memory=True,
        embedder=embedder_config,
        verbose=True
    )

    # Execute
    result = enterprise_crew.kickoff()
    return result

if __name__ == "__main__":
    agentops.init(tags=["enterprise-coding-crew"])
    print("Welcome to the Enterprise Multi-Agent Coding Pipeline!")
    print("Type 'exit' or 'quit' to close the pipeline.\n")
    try:
        while True:
            req = input("\nEnter your enterprise coding request: ")
            if req.lower() in ['exit', 'quit']:
                agentops.end_session("Success")
                break
            if req.strip():
                print(run_enterprise_pipeline(req))
    except Exception as e:
        print(f"Error: {e}")
        agentops.end_session("Fail")
