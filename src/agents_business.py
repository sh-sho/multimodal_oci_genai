from langchain_core.documents import Document
from langchain_core.runnables import RunnablePassthrough
from langgraph.graph import START, StateGraph
from typing_extensions import List, TypedDict
from langchain_community.chat_models import ChatOCIGenAI
from langchain.prompts.chat import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langfuse.callback import CallbackHandler

import os

from dotenv import find_dotenv, load_dotenv

from retriever.custom_retriever import CustomMarkdownRetriever

_ = load_dotenv(find_dotenv())

OCI_GENAI_ENDPOINT = os.getenv("OCI_GENAI_ENDPOINT")
OCI_COMPARTMENT_ID = os.getenv("OCI_COMPARTMENT_ID")
LANGFUSE_HOST = os.getenv("LANGFUSE_HOST")
LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY")
LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY")

llm = ChatOCIGenAI(
    model_id="cohere.command-r-plus-08-2024",
    service_endpoint="https://inference.generativeai.us-chicago-1.oci.oraclecloud.com",
    compartment_id=OCI_COMPARTMENT_ID,
    )

langfuse_handler = CallbackHandler(
    secret_key=LANGFUSE_SECRET_KEY,
    public_key=LANGFUSE_PUBLIC_KEY,
    host=LANGFUSE_HOST,
)

text_retriever = CustomMarkdownRetriever()


# Define state for application
class State(TypedDict):
    question: str
    context: List[Document]
    answer: str


# Define application steps
def retrieve(state: State):
    retrieved_docs = text_retriever.invoke(state["question"])
    return {
        "context": retrieved_docs,
        }
    
def chat_text(state: State):
    context = "\n\n".join(doc.page_content for doc in state["context"])
    llm = ChatOCIGenAI(
        model_id="cohere.command-r-08-2024",
        service_endpoint="https://inference.generativeai.us-chicago-1.oci.oraclecloud.com",
        compartment_id=OCI_COMPARTMENT_ID,
        )
    
    prompt = ChatPromptTemplate([
        ("system", "あなたは質疑応答のAIアシスタントです。必ず日本語で答えてください。"),
        ("human", """
         以下のMarkdownのコンテキストに基づいて質問に答えてください。
         回答は数字だけを回答してください。
         ** 質問 **
          {question} 
          
        ** コンテキスト **
        """ + context),
    ])

    chain = {'question': RunnablePassthrough()} | prompt | llm | StrOutputParser()
    return {
        "answer": chain.invoke(state["question"]),
        }


def chat_business_agent(question: str):
    # Compile application and test
    graph_builder = StateGraph(State).add_sequence([retrieve, chat_text])
    graph_builder.add_edge(START, "retrieve")
    graph = graph_builder.compile(name="business_agent")
    result = graph.invoke({"question": question})
    print(result)
    return result["answer"]

graph_builder = StateGraph(State).add_sequence([retrieve, chat_text])
graph_builder.add_edge(START, "retrieve")
graph_business = graph_builder.compile(name="business_agent")

if __name__ == "__main__":
    # Get user input
    question = input("質問を入力してください: ")
    result = chat_business_agent(question)
    print(f"Answer: {result}")