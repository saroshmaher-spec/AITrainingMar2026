from langchain.chat_models import init_chat_model
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain.tools import tool
from langchain.agents import create_agent
import pprint
from dotenv import load_dotenv
load_dotenv()

model = init_chat_model("gpt-4.1-mini", model_provider="openai", temperature=0)
search = GoogleSerperAPIWrapper()


@tool
def intermediate_answer(query: str) -> str:
    """Useful for when you need to ask with search."""
    return search.run(query)


tools = [intermediate_answer]
agent = create_agent(model, tools)

# events = agent.stream(
#     {
#         "messages": [
#             ("user", "latest AI news 2026"),
#         ]
#     },
#     stream_mode="values",
# )

# for event in events:
#     event["messages"][-1].pretty_print()

response = agent.invoke({
        "messages": [
            ("user", "latest AI news 2026"),
        ]})

# print(response)
pprint.pp(response)