import base64
import bs4
from langchain import hub
from langchain_community.document_loaders import WebBaseLoader

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import START, StateGraph
from typing_extensions import List, TypedDict
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.documents import Document
from langchain_community.embeddings import OCIGenAIEmbeddings
from langchain_community.chat_models import ChatOCIGenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain.prompts.chat import ChatPromptTemplate
from IPython.display import Image, display
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool
from langfuse.callback import CallbackHandler

import getpass
import os

from dotenv import find_dotenv, load_dotenv

from retriever.custom_retriever import CustomImageRetriever, CustomTextRetriever
from utils.utils import chat_cohere, chat_with_image, summarize_image_to_text

_ = load_dotenv(find_dotenv())

OCI_GENAI_ENDPOINT = os.getenv("OCI_GENAI_ENDPOINT")
OCI_COMPARTMENT_ID = os.getenv("OCI_COMPARTMENT_ID")
LANGFUSE_HOST = os.getenv("LANGFUSE_HOST")
LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY")
LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY")

llm = ChatOCIGenAI(
    model_id="cohere.command-r-08-2024",
    service_endpoint="https://inference.generativeai.us-chicago-1.oci.oraclecloud.com",
    compartment_id=OCI_COMPARTMENT_ID,

    )

langfuse_handler = CallbackHandler(
    secret_key=LANGFUSE_SECRET_KEY,
    public_key=LANGFUSE_PUBLIC_KEY,
    host=LANGFUSE_HOST,
)

image_retriever = CustomImageRetriever()

# Define state for application
class State(TypedDict):
    question: str
    image_path: str
    context: List[Document]
    answer: str


# Define application steps
def retrieve(state: State):
    retrieved_docs = image_retriever.invoke(state["question"])
    return {
        "context": retrieved_docs, 
        "image_path": retrieved_docs[0].metadata["file_path"]
        }


def chat_image(state: State):
    image_path = state["image_path"]
    context = "\n\n".join(doc.page_content for doc in state["context"])
    chat_prompt = ChatPromptTemplate.from_messages([
        ("human", "以下の文章を英語に翻訳してください。解答だけを返してください。\n {input}"),
    ])
    en_question= llm.invoke(chat_prompt.invoke({"input": state["question"]}))
    print(f"Question: {state['question']}")
    print(f"Translated Question: {en_question.content}")
    prompt = f"""
        Please answer the following question based on the image.\n
        Be sure to answer based on the image and context.\n
        
        ** Question**
        {en_question.content}\n
        
        **Context**
        {context}\n
        """

    system_prompt = "You are a AI assistant. Please answer the question based on the image."
    print(f"image_path: {image_path}")
    response = chat_with_image(
        image_path=image_path,
        prompt=prompt,
        system_prompt=system_prompt
        )
    print(f"Response: {response}")
    return {"answer": response}


def translate(state: State):
    prompt = ChatPromptTemplate([
        ("system", "あなたは質疑応答のAIアシスタントです。必ず日本語で答えてください。"),
        ("human", "以下のコンテキストを日本語に翻訳してください。必ず答えのみ出力してください。{input} "),
    ])
    messages = prompt.invoke({"input": state["answer"]})
    response = llm.invoke(messages, config={"callbacks": [langfuse_handler]})
    return {"answer": response.content}


def chat_image_agent(question: str):
    # Compile application and test
    graph_builder = StateGraph(State).add_sequence([retrieve, chat_image, translate])
    graph_builder.add_edge(START, "retrieve")
    graph = graph_builder.compile(name="image_agent")
    result = graph.invoke({"question": question})
    return result["answer"]

graph_builder = StateGraph(State).add_sequence([retrieve, chat_image, translate])
graph_builder.add_edge(START, "retrieve")
graph_image = graph_builder.compile(name="image_agent")

if __name__ == "__main__":
    # Get user input
    question = input("質問を入力してください: ")
    result = chat_image_agent(question)
    print(f"Answer: {result}")