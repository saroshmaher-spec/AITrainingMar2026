from tavily import TavilyClient
from dotenv import load_dotenv
load_dotenv()
client = TavilyClient()
response = client.search(
    query="how to build a aws cloud server with vip and load balancer",
    search_depth="advanced"
)
print(response)