import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_tavily import TavilySearch

load_dotenv()


@tool
def analyze_logs(query: str) -> str:
    """Analyze application logs for errors or issues"""
    try:
        with open("../data/kafka.log") as f:
            logs = f.read()

        if "error" in logs.lower():
            return f"Error found in logs:\n{logs[:1000]}"
        return "No major errors found in logs"

    except Exception as e:
        return str(e)



@tool
def local_rag_search(query: str) -> str:
    """Search internal documents"""
    return "Simulated RAG result: Relevant internal knowledge about " + query


# ------------------------
# TOOL 3: Web Search
# ------------------------
tavily_tool = TavilySearch(max_results=3)



# Log Agent
log_agent = create_agent(
    model="gpt-4o-mini",
    tools=[analyze_logs],
    system_prompt="You are a log analysis expert. Focus only on logs."
)

# RAG Agent
rag_agent = create_agent(
    model="gpt-4o-mini",
    tools=[local_rag_search],
    system_prompt="You are a knowledge retrieval expert using internal data."
)

# Web Agent
web_agent = create_agent(
    model="gpt-4o-mini",
    tools=[tavily_tool],
    system_prompt="You are a web research agent. Use Tavily for latest info."
)


def supervisor(query: str):
    """
    Decide which agent to use based on query
    """

    query_lower = query.lower()

    if "log" in query_lower or "error" in query_lower or "kafka" in query_lower:
        print("→ Routing to LOG AGENT")
        return log_agent.invoke({
            "messages": [{"role": "user", "content": query}]
        })

    elif "internal" in query_lower or "document" in query_lower:
        print("→ Routing to RAG AGENT")
        return rag_agent.invoke({
            "messages": [{"role": "user", "content": query}]
        })

    else:
        print("→ Routing to WEB AGENT")
        return web_agent.invoke({
            "messages": [{"role": "user", "content": query}]
        })



query = "Investigate Kafka error and latest fixes"

response = supervisor(query)

print("\nFinal Response:\n")
print(response["messages"][-1].content)