import os
import streamlit as st
from dotenv import find_dotenv, load_dotenv
from langchain_community.chat_models import ChatOCIGenAI
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

_ = load_dotenv(find_dotenv())
OCI_CONFIG_FILE = "~/.oci/config"
OCI_GENAI_ENDPOINT = os.getenv("OCI_GENAI_ENDPOINT")
OCI_COMPARTMENT_ID = os.getenv("OCI_COMPARTMENT_ID")

st.title("Multimodalチャット")

if "session_id" not in st.session_state:
    st.session_state.session_id = "session_id"

if "history" not in st.session_state:
    st.session_state.history = StreamlitChatMessageHistory(key="langchain_messages")

if "chain" not in st.session_state:
    prompt = ChatPromptTemplate.from_messages(
        [
          ("system", "あなたのタスクはユーザーの質問に明確に答えることです。"),
          MessagesPlaceholder(variable_name="messages"),
          MessagesPlaceholder(variable_name="human_message"),
        ]
    )
    
    # Chat
    chat = ChatOCIGenAI(
        model_id="cohere.command-r-08-2024",
        service_endpoint="https://inference.generativeai.ap-osaka-1.oci.oraclecloud.com",
        compartment_id=OCI_COMPARTMENT_ID,
        model_kwargs={"temperature": 0.7, "max_tokens": 500},
    )
    
    chain = prompt | chat
    st.session_state.chain = chain

col1, col2 = st.columns([1, 3])
with col1:
    if st.button("履歴クリア"):
        st.session_state.history.clear()

with col2:
    if st.button("画像をアップロード", key="upload_image"):
        uploaded_image = st.file_uploader("画像をアップロードしてください", type=["png", "jpg", "jpeg"])

        if uploaded_image is not None:
            st.image(uploaded_image, caption="アップロードされた画像", clamp=True)

for message in st.session_state.history.messages:
    with st.chat_message(message.type):
        st.markdown(message.content)

if prompt := st.chat_input("なんでも聞いて.."):
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        response = st.write_stream(
          st.session_state.chain.stream(
            {
                "messages": st.session_state.history.messages,
                "human_message": [HumanMessage(content=prompt)]
            },
            config={"configurable": {"session_id": st.session_state.session_id}},
          )
        )
    st.session_state.history.add_user_message(prompt)
    st.session_state.history.add_ai_message(response)

