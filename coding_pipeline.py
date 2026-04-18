import os
from dotenv import load_dotenv

# AgentOps for monitoring
import agentops

# CrewAI
from crewai import Agent, Task, Crew, Process, LLM

# Initialize environment variables
load_dotenv()

# Initialize AgentOps
agentops.init(tags=["coding-crew"])

# Initialize Google Generative AI (Gemini) LLM via CrewAI's LLM wrapper
llm = LLM(
    model="gemini/gemini-1.5-pro",
    api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0.2, # Slight creativity for coding, but mostly deterministic
    max_tokens=4096
)

# ---------------------------------------------------------
# Agent Definitions
# ---------------------------------------------------------

senior_developer = Agent(
    role="Senior Python Developer",
    goal="Write functional, clean, and efficient Python code based on the user's requirements and retrieved context.",
    backstory=(
        "You are an elite Python developer with over 10 years of experience building scalable applications. "
        "Your code is known for being elegant, well-documented, and following PEP 8 standards perfectly. "
        "You focus on writing the initial implementation to strictly meet the user's requirements."
    ),
    verbose=True,
    allow_delegation=False,
    llm=llm
)

qa_engineer = Agent(
    role="QA & Security Engineer",
    goal="Review the Developer's code for bugs, logic errors, and vulnerabilities, and write an optimized/fixed alternative version.",
    backstory=(
        "You are a meticulous Quality Assurance and Security Engineer. You possess a sharp eye for edge cases, "
        "inefficient loops, security vulnerabilities, and logic flaws that developers often miss. Your job is to "
        "take the developer's raw code, critique it, and provide a hardened, optimized alternative version."
    ),
    verbose=True,
    allow_delegation=False,
    llm=llm
)

chief_architect = Agent(
    role="Chief Software Architect",
    goal="Evaluate both the original and optimized code, compare their time complexity and logic, and output the final best version along with a brief comparison report.",
    backstory=(
        "You are the Chief Software Architect of the engineering team. You have the final say on all code that goes into production. "
        "You evaluate multiple implementations of a feature, weigh their pros and cons regarding time/space complexity, readability, and security, "
        "and synthesize the ultimate, production-ready version of the code. You communicate your decisions clearly."
    ),
    verbose=True,
    allow_delegation=False,
    llm=llm
)

# ---------------------------------------------------------
# Main Execution Function
# ---------------------------------------------------------

def run_coding_pipeline(coding_request: str):
    """Creates tasks for the query and executes the coding crew."""

    dev_task = Task(
        description=(
            f"Write the initial Python implementation for the following request:\n"
            f"'{coding_request}'\n"
            "Ensure the code is complete, functional, and includes docstrings."
        ),
        expected_output="A complete, well-documented Python script that fulfills the initial user requirements.",
        agent=senior_developer
    )

    qa_task = Task(
        description=(
            "Review the output provided by the Senior Python Developer.\n"
            "1. Identify any potential bugs, unhandled edge cases, or inefficiencies.\n"
            "2. Write a newly optimized and hardened version of the code that fixes these issues.\n"
            "3. Clearly separate your critique from the new code block."
        ),
        expected_output="A critique of the original code followed by an optimized, hardened Python script.",
        agent=qa_engineer,
        context=[dev_task] # Explicitly pass the output of the dev task
    )

    judge_task = Task(
        description=(
            "Review both the original implementation (from the Developer) and the optimized implementation (from the QA Engineer).\n"
            "1. Compare both versions based on time complexity, space complexity, security, and readability.\n"
            "2. Decide which version is objectively better, or merge the best parts of both.\n"
            "3. Output a brief comparison report outlining your reasoning.\n"
            "4. Finally, output the definitive, production-ready Python code block."
        ),
        expected_output="A brief comparison report followed by the definitive, final production-ready Python code.",
        agent=chief_architect,
        context=[dev_task, qa_task] # Provide both the original and the QA versions
    )

    coding_crew = Crew(
        agents=[senior_developer, qa_engineer, chief_architect],
        tasks=[dev_task, qa_task, judge_task],
        process=Process.sequential,
        verbose=True
    )

    print("\n==================================================")
    print(f"Executing Multi-Agent Coding Pipeline for request:\n'{coding_request}'")
    print("==================================================\n")

    try:
        result = coding_crew.kickoff()
        agentops.end_session("Success")
        return result
    except Exception as e:
        agentops.end_session("Fail")
        return f"Pipeline failed: {e}"

if __name__ == "__main__":
    # Dummy coding request to test the simulation
    dummy_request = (
        "Write a Python function to find all prime numbers up to a given integer N. "
        "It should be somewhat efficient, but I'm not an expert so just get it working."
    )

    final_output = run_coding_pipeline(dummy_request)

    print("\n" + "="*50)
    print("FINAL ARCHITECT DECISION & CODE:")
    print("="*50)
    print(final_output)
