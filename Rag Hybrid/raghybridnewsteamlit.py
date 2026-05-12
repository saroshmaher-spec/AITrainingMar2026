import streamlit as st
import os
import json
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.tools import tool
from tavily import TavilyClient
from dotenv import load_dotenv

# ===============================
# Load environment variables
# ===============================
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

# ===============================
# Initialize LLM
# ===============================
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.3,
    openai_api_key=OPENAI_API_KEY
)

# ===============================
# Tools
# ===============================

@tool
def check_kafka_log_exists(file_path: str) -> str:
    """Check whether the uploaded kafka.log exists and report size."""
    print(f"Checking file at: {file_path}")
    if os.path.isfile(file_path):
        return f"FOUND: {file_path} ({os.path.getsize(file_path)} bytes)"
    return f"NOT FOUND: {file_path}"

@tool
def analyze_kafka_log(file_path: str) -> str:
    """Analyze Kafka replication logs for production failure patterns."""
    if not os.path.isfile(file_path):
        return f"kafka.log not found at {file_path}"

    with open(file_path, "r") as f:
        logs = f.read().lower()

    findings = []

    failure_signatures = {
        "ISR Shrink": ["shrinking isr", "isr shrunk", "isr shrink"],
        "ReplicaFetcher Lag": ["replicafetcherthread stuck", "replicafetcher is slow"],
        "Not Enough Replicas": ["notenoughreplicasexception"],
        "Leader Not Available": ["leader not available"],
        "Request Timeout": ["requesttimedoutexception", "timed out waiting"]
    }

    for key, patterns in failure_signatures.items():
        for p in patterns:
            if p in logs:
                findings.append(f"{key} detected → {p}")
                break

    if not findings:
        return "No critical Kafka replication issues detected."

    return "\n".join(findings)

@tool
def tavily_search(query: str) -> str:
    """Search Kafka issues using Tavily API."""
    client = TavilyClient(api_key=TAVILY_API_KEY)
    res = client.search(query=query, max_results=5)
    return json.dumps(res.get("results", []), indent=2)

web_search = DuckDuckGoSearchRun()

# ===============================
# Create Agent
# ===============================
tools = [
    check_kafka_log_exists,
    analyze_kafka_log,
    tavily_search,
    web_search
]

agent = create_agent(
    model="gpt-4o-mini",
    tools=tools,
    system_prompt="""
You are a Kafka SRE AI.
Always run analyze_kafka_log first.
If severity is CRITICAL, you MUST call tavily_search with the detected issue.
If severity is OK, never use web search tools.
provide me the immediate fixes we need to do
provide me the long term fixes we need to do
"""
)

# ===============================
# Streamlit UI
# ===============================
st.set_page_config(page_title="Kafka SRE Copilot", layout="wide")
st.title(" Kafka SRE Copilot")
st.caption("Upload your kafka.log to automatically detect replication issues and ISR problems.")

uploaded_file = st.file_uploader("Upload kafka.log file", type=["log", "txt"])

run_btn = st.button("Run Diagnosis")

if uploaded_file and run_btn:
    temp_path = os.path.join("kafka.log")
    st.info(f"Uploaded file: {uploaded_file.name} ({uploaded_file.size} bytes)")
    st.info(temp_path)
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    with st.spinner("Analyzing Kafka logs..."):
        query = "Check uploaded kafka.log existence and diagnose replication or ISR issues."
        response = agent.invoke({
            "messages": [{"role": "user", "content": query}],
            "file_path": temp_path
        })

        # Display only the clean output
        st.subheader("Diagnosis Result")
        st.code(response["messages"][-1].content)