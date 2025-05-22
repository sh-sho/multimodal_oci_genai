import os
import re
import base64
from typing import List
import pandas as pd
from mcp.server.fastmcp import FastMCP
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from dotenv import find_dotenv, load_dotenv

from langchain.callbacks.manager import CallbackManagerForRetrieverRun
from langchain.schema import BaseRetriever
from langchain.chat_models import init_chat_model
from langchain_core.documents import Document
from langchain.prompts.chat import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.chat_models import ChatOCIGenAI
from langchain_community.embeddings import OCIGenAIEmbeddings
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.prebuilt import create_react_agent
from langgraph_supervisor import create_supervisor
from langchain.tools import tool
from langchain_mcp_adapters.tools import load_mcp_tools


from oci.generative_ai_inference import GenerativeAiInferenceClient
from oci.generative_ai_inference.models import (
    EmbedTextDetails,
    OnDemandServingMode,
)
import oracledb

_ = load_dotenv(find_dotenv())
oracledb.init_oracle_client()
mcp = FastMCP("RAG")

UN = os.getenv("UN")
PW = os.getenv("PW")
DSN = os.getenv("DSN")
OCI_COMPARTMENT_ID = os.getenv("OCI_COMPARTMENT_ID")



def get_embedding(text: str) -> list:
  embeddings = OCIGenAIEmbeddings(
    model_id="cohere.embed-multilingual-image-v3.0",
    service_endpoint="https://inference.generativeai.us-chicago-1.oci.oraclecloud.com",
    compartment_id=OCI_COMPARTMENT_ID,
  )
  return embeddings.embed_query(text)

class CustomMarkdownRetriever(BaseRetriever):
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
                    df = pd.DataFrame()
                    cursor.setinputsizes(oracledb.DB_TYPE_VECTOR)
                    select_sql = f"""
                        SELECT
                            file_id,
                            markdown
                        FROM
                            docs_contents
                        ORDER BY VECTOR_DISTANCE(embedding, to_vector(:1, 1024, FLOAT32), COSINE)
                        FETCH FIRST 3 ROWS ONLY
                    """
                    cursor.execute(select_sql, [embed_query])
                    for row in cursor:
                        df_tmp = pd.DataFrame([[row[0], row[1].read()]],
                                                columns=["file_id", "markdown"])
                        df = pd.concat([df, df_tmp], ignore_index=True)
                    
                    for i in range(len(df)):
                        file_id = df.iloc[i, 0]
                        markdown = df.iloc[i, 1]
                        # print(f"file_id: {file_id}, markdown: {markdown}")
                        doc = Document(
                            page_content=markdown,
                            metadata={'file_id':file_id, 'vector_index': i}
                            )
                        docs.append(doc)
                connection.close()
        except oracledb.DatabaseError as e:
            print(f"Database error: {e}")
            raise
        except Exception as e:
            print("Error Vector Search:", e)

        return docs


@tool
def get_text_with_markdown(query: str) -> str:
    """
    Get the value of department expenses, sales, and operating income by text with markdown retriever
    Args:
        query (str): The query to ask the model. YYYY/MM format is required.
    Returns:
        str: The answer from the model.
    """
    
    llm = ChatOCIGenAI(
        model_id="cohere.command-a-03-2025",
        service_endpoint="https://inference.generativeai.us-chicago-1.oci.oraclecloud.com",
        compartment_id=OCI_COMPARTMENT_ID,
        )
    
    prompt = ChatPromptTemplate([
        ("system", "あなたは質疑応答のAIアシスタントです。必ず日本語で答えてください。"),
        ("human", """
         以下のMarkdownのコンテキストに基づいて質問に答えてください。
         回答は数字だけを回答してください。
         ** 質問 **
          {query} 
          
        ** コンテキスト **
        {context}
        """),
    ])

    retriever = CustomMarkdownRetriever()
    chain = {'query': RunnablePassthrough(), 'context': retriever} | prompt | llm | StrOutputParser()

    result = chain.invoke(query)
    return result



@mcp.tool()
def retrieve_answers_with_markdown(query: str) -> str:
    """
    Given a query, get the value of department expenses, sales, and operating income by text with markdown retriever
    """
    return get_text_with_markdown(query)

if __name__ == "__main__":
    # Run the MCP server with a chosen transport method; stdio is used for demonstration.
    mcp.run(transport="stdio")