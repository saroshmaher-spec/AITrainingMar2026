from langchain_tavily import TavilySearch
from dotenv import load_dotenv
load_dotenv()
tool = TavilySearch(
    max_results=1,
    topic="general")


response = tool.invoke("latest AI news 2026")

print(response)