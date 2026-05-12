from tavily import TavilyClient
from dotenv import load_dotenv
load_dotenv()
client = TavilyClient()
response = client.crawl(
    url="https://huggingface.co/models",
    extract_depth="advanced"
)
print(response)