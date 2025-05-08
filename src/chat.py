import base64
import os
import re
from typing import List
from dotenv import find_dotenv, load_dotenv

from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.chat_models import ChatOCIGenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.callbacks.manager import CallbackManagerForRetrieverRun
from langchain.schema import BaseRetriever
from langchain_core.documents import Document
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import oracledb
import numpy as np
from pandas import array

from retriever.custom_retriever import CustomImageRetriever, CustomTextRetriever
from utils.utils import chat_with_image, get_embedding, summarize_image_to_text

_ = load_dotenv(find_dotenv())
oracledb.init_oracle_client()

UN = os.environ.get("UN")
PW = os.environ.get("PW")
DSN = os.environ.get("DSN")

OCI_CONFIG_FILE = "~/.oci/config"
OCI_GENAI_ENDPOINT = os.getenv("OCI_GENAI_ENDPOINT")
OCI_COMPARTMENT_ID = os.getenv("OCI_COMPARTMENT_ID")
MOVIE_DIRECTORY_PATH = os.getenv("MOVIE_DIRECTORY_PATH")
SPLIT_MOVIE_DIRECTORY_PATH = os.getenv("SPLIT_MOVIE_DIRECTORY_PATH")


## Text -> Text
def get_text_by_text():
    llm = ChatOCIGenAI(
        model_id="cohere.command-r-08-2024",
        service_endpoint="https://inference.generativeai.ap-osaka-1.oci.oraclecloud.com",
        compartment_id=OCI_COMPARTMENT_ID,
        model_kwargs={"temperature": 0.7, "max_tokens": 500},
        )
    
    prompt = ChatPromptTemplate([
        ("system", "あなたは質疑応答のAIアシスタントです。必ず日本語で答えてください。"),
        ("human", "{query} 以下のコンテキストに基づいて答えてください。{context}"),
    ])

    retriever = CustomTextRetriever()
    chain = {'query': RunnablePassthrough(), 'context': retriever} | prompt | llm | StrOutputParser()

    result = chain.invoke("洗濯機の性能を教えてください。")
    print(result)

## Text -> Image
def get_text_by_image():
    llm = ChatOCIGenAI(
        model_id="cohere.command-r-08-2024",
        service_endpoint="https://inference.generativeai.ap-osaka-1.oci.oraclecloud.com",
        compartment_id=OCI_COMPARTMENT_ID,
        model_kwargs={"temperature": 0.7, "max_tokens": 500},
        )
    
    prompt = ChatPromptTemplate([
        ("system", "あなたは質疑応答のAIアシスタントです。必ず日本語で答えてください。"),
        ("human", "{query} 以下のコンテキストに基づいて答えてください。{context}"),
    ])

    retriever = CustomImageRetriever()
    chain = {'query': RunnablePassthrough(), 'context': retriever} | prompt | llm | StrOutputParser()

    result_images = chain.invoke("洗濯機の性能を教えてください。")
    print(result_images)
    
    file_id = result_images[0].metadata['file_id']
    try:
        with oracledb.connect(user=UN, password=PW, dsn=DSN) as connection:
            with connection.cursor() as cursor:
                cursor.setinputsizes(oracledb.DB_TYPE_VECTOR)
                select_sql = f"""
                    SELECT
                        image_blob
                    FROM
                        image_contents
                    WHERE file_id = :1
                """
                cursor.execute(select_sql, [file_id])
                blob, = cursor.fetchone()
                offset = 1
                bufsize = 65536
                with open('tmp_image.png', 'wb') as f:
                    while True:
                        data = blob.read(offset, bufsize)
                        if data:
                            f.write(data)
                        if len(data) < bufsize:
                            break
                        offset += bufsize
            connection.close()
                    
    except oracledb.DatabaseError as e:
        print(f"Database error: {e}")
        raise
    except Exception as e:
        print("Error Vector Search:", e)
    


## Image -> Image
def get_image_by_image():
    llm = ChatOCIGenAI(
        model_id="cohere.command-r-08-2024",
        service_endpoint="https://inference.generativeai.ap-osaka-1.oci.oraclecloud.com",
        compartment_id=OCI_COMPARTMENT_ID,
        model_kwargs={"temperature": 0.7, "max_tokens": 500},
        )
    
    prompt = ChatPromptTemplate([
        ("system", "あなたは質疑応答のAIアシスタントです。必ず日本語で答えてください。"),
        ("human", "{query} 以下のコンテキストに基づいて答えてください。{context}"),
    ])

    retriever = CustomImageRetriever()

    tmp_image = "./images/washing_machine.png"
    image_text = summarize_image_to_text(image_path=tmp_image, uploaded_url="")
    result_images = retriever.invoke(image_text)
    print(result_images)

    file_id = result_images[0].metadata['file_id']
    try:
        with oracledb.connect(user=UN, password=PW, dsn=DSN) as connection:
            with connection.cursor() as cursor:
                cursor.setinputsizes(oracledb.DB_TYPE_VECTOR)
                select_sql = f"""
                    SELECT
                        image_blob
                    FROM
                        image_contents
                    WHERE file_id = :1
                """
                cursor.execute(select_sql, [file_id])
                blob, = cursor.fetchone()
                offset = 1
                bufsize = 65536
                with open('tmp_image.png', 'wb') as f:
                    while True:
                        data = blob.read(offset, bufsize)
                        if data:
                            f.write(data)
                        if len(data) < bufsize:
                            break
                        offset += bufsize
            connection.close()
                    
    except oracledb.DatabaseError as e:
        print(f"Database error: {e}")
        raise
    except Exception as e:
        print("Error Vector Search:", e)

## Text -> Movie
def simulate_rename_files_in_directory(file_path: str, pattern: str, replacement: str = "") -> List[str]:
    """
    Simulates renaming files in the specified directory by replacing a pattern with a replacement string.
    Does not actually rename the files, only returns the new file names.

    Args:
        directory_path (str): Path to the directory containing files to simulate renaming.
        pattern (str): Regular expression pattern to match in the file names.
        replacement (str): String to replace the matched pattern with. Defaults to an empty string.

    Returns:
        List[str]: List of new file names after the simulated renaming.
    """
    simulated_renamed_files = []
    new_filename = re.sub(pattern, replacement, file_path)
    if new_filename != file_path:
        simulated_renamed_files.append(new_filename)
        print(f"Simulated rename: {file_path} -> {new_filename}")
    return simulated_renamed_files

def get_movie_by_text():
    llm = ChatOCIGenAI(
        model_id="cohere.command-r-08-2024",
        service_endpoint="https://inference.generativeai.ap-osaka-1.oci.oraclecloud.com",
        compartment_id=OCI_COMPARTMENT_ID,
        model_kwargs={"temperature": 0.7, "max_tokens": 500},
    )
    
    prompt = ChatPromptTemplate([
        ("system", "あなたは質疑応答のAIアシスタントです。必ず日本語で答えてください。"),
        ("human", "{query} 以下のコンテキストに基づいて答えてください。{context}"),
    ])
    query = "洗濯機の使い方を説明する動画を探してください。"
    
    retriever = CustomImageRetriever()
    result_images = retriever.invoke(query)
    print(result_images)

    file_url = result_images[0].metadata['file_path']
    movie_path = simulate_rename_files_in_directory(
        file_path=file_url,
        pattern=r"_[0-9]+_[0-9]+\.[0-9]+\.png",
        replacement=f".mp4"
    )
    print(movie_path)
    


## Text -> Text with Image
def get_text_by_text_with_image(question: str):
    
    prompt = ChatPromptTemplate([
        ("system", "あなたは言語翻訳のAIアシスタントです。日本語を英語に翻訳してください。"),
        ("human", "{input} "),
    ])
    
    llm = ChatOCIGenAI(
        model_id="cohere.command-r-08-2024",
        service_endpoint="https://inference.generativeai.ap-osaka-1.oci.oraclecloud.com",
        compartment_id=OCI_COMPARTMENT_ID,
    )
    chain = {'input': RunnablePassthrough()} | prompt | llm | StrOutputParser()
    question_en = chain.invoke({"input":question})
    
    retriever = CustomImageRetriever()
    result_images = retriever.invoke(question)
    print(result_images)
    
    image_path = result_images[0].metadata['file_path']

    res = chat_with_image(
        image_path=image_path,
        prompt=question_en,
        system_prompt="You are a AI assistant. Please answer the question based on the image."
    )
    return res

  
if __name__ == "__main__":
    response = get_text_by_text_with_image(question = "洗濯機の使い方を説明する情報を探してください。")
    print(response)
