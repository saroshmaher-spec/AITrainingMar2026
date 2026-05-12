from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.agents import create_agent
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains import RetrievalQA
from langchain.tools import tool
from tavily import TavilyClient # Import the official SDK
import os
import json
from dotenv import load_dotenv

# ===============================
# Setup
# ===============================
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.3,
    openai_api_key=OPENAI_API_KEY
)

# ===============================
# Kafka Tools
# ===============================

@tool
def check_kafka_log_exists(directory: str = ".", filename: str = "/Users/premkumargontrand/AITraining2026/langchain/kafka/kafka.log") -> str:
    """Check whether kafka.log exists in the given directory."""
    path = os.path.join(directory, filename)
    if os.path.isfile(path):
        return f"FOUND: {path} ({os.path.getsize(path)} bytes)"
    return f"NOT FOUND: {path}\nFiles: {os.listdir(directory)}"


@tool
def analyze_kafka_log(file_path: str = "/Users/premkumargontrand/AITraining2026/langchain/kafka/kafka.log") -> str:
    """Analyze Kafka replication logs for ISR / fetcher / timeout issues."""
    if not os.path.isfile(file_path):
        return f"kafka.log not found at {file_path}"

    with open(file_path, "r") as f:
        logs = f.read().lower()

    findings = []

    if "isr" in logs and "shrink" in logs:
        findings.append("ISR shrink → Broker GC pause / network latency.")
    if "replicafetcher" in logs:
        findings.append("ReplicaFetcher lag → Disk IO bottleneck.")
    if "notenoughreplicas" in logs:
        findings.append("NotEnoughReplicas → Under replicated partitions.")
    if "timeout" in logs:
        findings.append("Request timeout → Network or broker overload.")

    if not findings:
        findings.append("No critical Kafka replication issues detected.")

    return "\n".join(findings)

# ===============================
# External Search Tools
# ===============================

@tool
def tavily_search(query: str) -> str:
    """Search Kafka issues using Tavily API."""
    # client = TavilyClient(api_key=TAVILY_API_KEY)
    # res = client.search(query=query, max_results=5)
    TavilySearchResults(max_result)
    return json.dumps(res["results"], indent=2)

web_search = DuckDuckGoSearchRun()

# ===============================
# Agent (v1 compatible)
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
    system_prompt="You are an AI assistant. Use tools when needed to answer queries."
)

# ===============================
# Run
# ===============================

if __name__ == "__main__":
    query = "Check kafka.log existence and diagnose replication or ISR issues."
    response = agent.invoke({"messages": [{"role": "user", "content": query}]})
    print("\n====== Kafka SRE Diagnosis ======\n")
    print(response['messages'][-1].content)

    