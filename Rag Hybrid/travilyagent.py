from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain_tavily import TavilySearch
from dotenv import load_dotenv
load_dotenv()


tool = TavilySearch(max_results=1)

agent = create_agent(
    model="gpt-4o-mini",
    tools=[tool],
    system_prompt="You are a helpful AI assistant"
)
user_input = input("Enter your query: ")
result = agent.invoke({
    "messages": [
        {"role": "user", "content": user_input}
    ]
})

# print(result.message[-1].content)
print(result['messages'][-1].content)