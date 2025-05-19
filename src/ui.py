import os
import streamlit as st
from dotenv import find_dotenv, load_dotenv
from langchain_community.chat_models import ChatOCIGenAI
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_core.messages import HumanMessage
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain.chains import RetrievalQA
from langgraph.prebuilt import create_react_agent

from agents_image import chat_image_agent
from retriever.custom_retriever import CustomImageRetriever, CustomTextRetriever

_ = load_dotenv(find_dotenv())
OCI_CONFIG_FILE = "~/.oci/config"
OCI_GENAI_ENDPOINT = os.getenv("OCI_GENAI_ENDPOINT")
OCI_COMPARTMENT_ID = os.getenv("OCI_COMPARTMENT_ID")

st.title("Multimodalチャット")


    
# Chat
chat = ChatOCIGenAI(
    model_id="cohere.command-r-08-2024",
    service_endpoint="https://inference.generativeai.us-chicago-1.oci.oraclecloud.com",
    compartment_id=OCI_COMPARTMENT_ID,
    model_kwargs={"temperature": 0.7, "max_tokens": 500},
)

retriever = CustomTextRetriever()

if "messages" not in st.session_state:
    st.session_state["messages"] = []


def post_message(message, role, save=True):
    with st.chat_message(role):
        st.write(message)

    if save:
        st.session_state["messages"].append({"message": message, "role": role})

def post_image(image_path, role, save=True):
    with st.chat_message(role):
        st.image(image_path, caption=image_path.name)

    if save:
        st.session_state["messages"].append({"image": image_path, "role": role})

def show_message_history():
    for message in st.session_state["messages"]:
        post_message(
            message["message"],
            message["role"],
            save=False,
        )

def join_docs(docs: list[Document]) -> str:
    return "\n".join(document.page_content for document in docs)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "あなたのタスクはユーザーの質問に明確に答えることです。"),
        ("human", "{input} 以下のコンテキストに基づいて答えてください.{context}"),
    ]
)

col1, col2 = st.columns([1, 3])
with col1:
    if st.button("履歴クリア"):
        st.session_state.messages.clear()

with col2:
    if st.button("画像をアップロード", key="upload_image"):
        uploaded_file = st.file_uploader(
            "画像をアップロードしてください",
            type=["png", "jpg", "jpeg"],
            accept_multiple_files=False
        )

        if uploaded_file:
            st.image(uploaded_file, caption=uploaded_file.name)

        


post_message("こんにちは！", "ai", save=False)

show_message_history()

message = st.chat_input("なんでも聞いて..")

retrieve_docs = (lambda x: x["input"]) | retriever

rag_chain_from_docs = (
    RunnablePassthrough.assign(context=(lambda x: join_docs(x["context"])))
    | prompt
    | chat
    | StrOutputParser()
)

# if message:
#     post_message(message, "user")
#     rag_chain = RunnablePassthrough.assign(
#             context=retrieve_docs,
#         ).assign(
#             content=rag_chain_from_docs,
#         )
#     response = rag_chain.invoke({"input": message})
#     print(response)
#     post_message(response["content"], "ai")

if message:
    post_message(message, "user")
    response = chat_image_agent(question=message)
    post_message(response, "ai")

     
# retriever_image = CustomImageRetriever()

# # retrieve_docs = (lambda x: x["input"]) | retriever_image
