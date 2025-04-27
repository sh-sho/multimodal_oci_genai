import os
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
from pandas import array

from utils.utils import get_embedding, summarize_image_to_text

_ = load_dotenv(find_dotenv())
oracledb.init_oracle_client()

UN = os.environ.get("UN")
PW = os.environ.get("PW")
DSN = os.environ.get("DSN")

OCI_CONFIG_FILE = "~/.oci/config"
OCI_GENAI_ENDPOINT = os.getenv("OCI_GENAI_ENDPOINT")
OCI_COMPARTMENT_ID = os.getenv("OCI_COMPARTMENT_ID")

class CustomTextRetriever(BaseRetriever):
    """
    Custom retriever.
    """

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        docs: List[Document] = []
        embed_query = str(get_embedding(query))
        try:
            with oracledb.connect(user=UN, password=PW, dsn=DSN) as connection:
                with connection.cursor() as cursor:
                    cursor.setinputsizes(oracledb.DB_TYPE_VECTOR)
                    select_sql = f"""
                        SELECT
                            file_id,
                            summary
                        FROM
                            docs_contents
                        ORDER BY VECTOR_DISTANCE(embedding, to_vector(:1, 1024, FLOAT32), COSINE)
                        FETCH FIRST 3 ROWS ONLY
                    """
                    cursor.execute(select_sql, [embed_query])
                    index = 1
                    for row in cursor:
                        doc = Document(
                            page_content=row[1],
                            metadata={'file_id':row[0], 'vector_index': index}
                            )
                        docs.append(doc)
                        index += 1
                    connection.commit()
                        
        except oracledb.DatabaseError as e:
            print(f"Database error: {e}")
            raise
        except Exception as e:
            print("Error Vector Search:", e)
        
        return docs

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

# retriever = CustomTextRetriever()
# chain = {'query': RunnablePassthrough(), 'context': retriever} | prompt | llm | StrOutputParser()

# result = chain.invoke("洗濯機の性能を教えてください。")
# print(result)



class CustomImageRetriever(BaseRetriever):
    
    """
    Custom image retriever.
    """

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        docs: List[Document] = []
        embed_query = str(get_embedding(query))
        try:
            with oracledb.connect(user=UN, password=PW, dsn=DSN) as connection:
                with connection.cursor() as cursor:
                    cursor.setinputsizes(oracledb.DB_TYPE_VECTOR)
                    select_sql = f"""
                        SELECT
                            file_id,
                            summary
                        FROM
                            image_contents
                        ORDER BY VECTOR_DISTANCE(embedding, to_vector(:1, 1024, FLOAT32), COSINE)
                        FETCH FIRST 3 ROWS ONLY
                    """
                    cursor.execute(select_sql, [embed_query])
                    index = 1
                    for row in cursor:
                        doc = Document(
                            page_content=row[1],
                            metadata={'file_id':row[0], 'vector_index': index}
                            )
                        docs.append(doc)
                        index += 1
                    connection.commit()
                    cursor.close()
                connection.close()
                        
        except oracledb.DatabaseError as e:
            print(f"Database error: {e}")
            raise
        except Exception as e:
            print("Error Vector Search:", e)
        
        return docs



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
result_image = retriever.invoke(image_text)
print(result_image)

