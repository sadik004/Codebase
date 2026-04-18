import streamlit as st
import agentops
import os

st.set_page_config(page_title="Multi-Agent Coding Pipeline", page_icon="🤖", layout="wide")

st.title("🤖 Agentic Multi-Agent Coding Pipeline")
st.markdown("Enter your coding request below. The CrewAI agents (Developer, QA, and Judge) will write, review, and evaluate the code, and then save the best version to the `./output_code/` directory.")

coding_request = st.text_area("What would you like the agents to build?", height=150, placeholder="e.g., Write a Python script to scrape a website and save data to a CSV...")

if st.button("Generate Code", type="primary"):
    if not coding_request.strip():
        st.warning("Please enter a coding request.")
    else:
        # We run the pipeline in a subprocess to avoid asyncio/uvloop conflicts with Streamlit
        import subprocess

        try:
            agentops.init(tags=["streamlit-ui"])
        except Exception as e:
            st.error(f"Failed to initialize AgentOps: {e}")

        with st.spinner("Agents are working... This may take a few minutes as they write, evaluate, and optimize code."):
            try:
                # Create a temporary script to run the pipeline.
                # Use sys.argv to safely pass the input to avoid code injection vulnerabilities.
                temp_script = """
import sys
from coding_pipeline import run_coding_pipeline
import agentops

try:
    coding_request = sys.argv[1]
    agentops.init(tags=["coding-crew"])
    result = run_coding_pipeline(coding_request)
    print("SUCCESS_MARKER")
    print(result)
    agentops.end_session("Success")
except Exception as e:
    print(f"Error: {e}")
    agentops.end_session("Fail")
    sys.exit(1)
"""
                with open("run_temp.py", "w") as f:
                    f.write(temp_script)

                # Execute in subprocess safely
                process = subprocess.run(["python", "run_temp.py", coding_request], capture_output=True, text=True)

                if "SUCCESS_MARKER" in process.stdout:
                    result = process.stdout.split("SUCCESS_MARKER")[1].strip()
                    st.success("Pipeline execution complete!")
                    st.markdown("### 🏆 Judge Agent Final Report")
                    st.markdown(result)
                else:
                    st.error(f"Pipeline failed. Error Output:\n{process.stderr}\n{process.stdout}")

                agentops.end_session("Success")

            except Exception as e:
                st.error(f"An error occurred during pipeline execution: {e}")
                try:
                    agentops.end_session("Fail")
                except:
                    pass
            finally:
                if os.path.exists("run_temp.py"):
                    os.remove("run_temp.py")

st.markdown("---")
st.markdown("### 📁 Recent Generated Files")
OUTPUT_CODE_DIR = "./output_code"
if os.path.exists(OUTPUT_CODE_DIR):
    files = os.listdir(OUTPUT_CODE_DIR)
    if files:
        for f in files:
            file_path = os.path.join(OUTPUT_CODE_DIR, f)
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
            with st.expander(f"📄 {f}"):
                st.code(content, language='python')
    else:
        st.info("No generated files yet.")
else:
    st.info("No generated files yet.")
