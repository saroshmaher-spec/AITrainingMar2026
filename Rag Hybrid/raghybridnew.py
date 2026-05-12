import os
import json
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_tavily import TavilySearch
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader, PyPDFLoader,CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.chains import RetrievalQA
from dotenv import load_dotenv
load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
tavily_api_key = os.getenv("TAVILY_API_KEY")

print(f"Tavily API Key Loaded: {bool(tavily_api_key)}")
print(f"OpenAI API Key Loaded: {bool(openai_api_key)}")

# LLM
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.4,
    api_key=openai_api_key
)


@tool("local_doc_search")
def local_doc_search(query: str, directory: str = "../data") -> str:
    """Search for answers in local PDF or TXT documents within the given directory."""
    try:
        if not os.path.exists(directory):
            return f"Directory not found: {directory}"

        docs = []
        for file in os.listdir(directory):
            path = os.path.join(directory, file)

            if file.endswith(".txt") or file.endswith(".log"):
                docs.extend(TextLoader(path).load())
            elif file.endswith(".pdf"):
                docs.extend(PyPDFLoader(path).load())
            elif file.endswith(".csv"):
                docs.extend(CSVLoader(path).load())

        if not docs:
            return f"No readable files found in {directory}"

        # Split documents
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        splits = splitter.split_documents(docs)

        # Embeddings + Vector DB
        embeddings = OpenAIEmbeddings(api_key=openai_api_key)
        vectorstore = FAISS.from_documents(splits, embeddings)

        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

        # QA Chain
        qa = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever
        )

        answer = qa.invoke(query)
        return f"Local Search Result:\n{answer}"

    except Exception as e:
        return f"Error in local search: {str(e)}"



tavily_search = TavilySearch(
    max_results=5,
    tavily_api_key=tavily_api_key,
    name="tavily_search",
    description="Search the web for real-time and technical information"
)

web_search = DuckDuckGoSearchRun(name="web_search")


tools = [local_doc_search, tavily_search, web_search]

agent = create_agent(
    model="gpt-4o-mini",
    tools=tools,
    system_prompt="You are an AI assistant. Use tools intelligently when needed."
)


query = "Investigate Kafka replication log. Search local logs, Tavily, and the web for related insights."

response = agent.invoke({
    "messages": [
        {"role": "user", "content": query}
    ]
})


last_message = response['messages'][-1]

result = {
    "content": last_message.content,
    "type": last_message.type,
    "additional_kwargs": last_message.additional_kwargs,
    "response_metadata": last_message.response_metadata
}

print(json.dumps(result, indent=2))