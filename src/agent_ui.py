import os

import streamlit as st
from dotenv import find_dotenv, load_dotenv
from langchain.agents import AgentType, initialize_agent, load_tools
from langchain.callbacks import StreamlitCallbackHandler
from langchain_community.chat_models import ChatOCIGenAI
from langchain.memory import ConversationBufferMemory
from langchain.prompts import MessagesPlaceholder

_ = load_dotenv(find_dotenv())
OCI_CONFIG_FILE = "~/.oci/config"
OCI_GENAI_ENDPOINT = os.getenv("OCI_GENAI_ENDPOINT")
OCI_COMPARTMENT_ID = os.getenv("OCI_COMPARTMENT_ID")

def create_agent_chain():
    chat = ChatOCIGenAI(
        model_id="cohere.command-r-08-2024",
        service_endpoint="https://inference.generativeai.us-chicago-1.oci.oraclecloud.com",
        compartment_id=OCI_COMPARTMENT_ID,
        model_kwargs={"temperature": 0.7, "max_tokens": 500},
    )

    agent_kwargs = {
        "extra_prompt_messages": [MessagesPlaceholder(variable_name="memory")],
    }
    memory = ConversationBufferMemory(memory_key="memory", return_messages=True)

    tools = load_tools(["ddg-search", "wikipedia"])
    return initialize_agent(
        tools,
        chat,
        agent_kwargs=agent_kwargs,
        memory=memory,
    )


if "agent_chain" not in st.session_state:
    st.session_state.agent_chain = create_agent_chain()


st.title("langchain-streamlit-app")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

prompt = st.chat_input("What is up?")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        callback = StreamlitCallbackHandler(st.container())
        response = st.session_state.agent_chain.run(prompt, callbacks=[callback])
        st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})