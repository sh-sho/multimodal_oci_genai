import os
import re
from typing import List


from dotenv import find_dotenv, load_dotenv
from langchain_core.documents import Document
from langchain.callbacks.manager import CallbackManagerForRetrieverRun
from langchain.schema import BaseRetriever
import oracledb

from utils.utils import get_embedding, summarize_image_to_text

_ = load_dotenv(find_dotenv())
oracledb.init_oracle_client()

UN = os.environ.get("UN")
PW = os.environ.get("PW")
DSN = os.environ.get("DSN")

## Text -> Text
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
                    # connection.commit()
                        
        except oracledb.DatabaseError as e:
            print(f"Database error: {e}")
            raise
        except Exception as e:
            print("Error Vector Search:", e)
        
        return docs
    
## Text -> Image
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
                            image_path,
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
                            page_content=row[2],
                            metadata={
                                'file_id':row[0], 
                                'file_path': row[1], 
                                'vector_index': index
                                }
                            )
                        docs.append(doc)
                        index += 1
                    # connection.commit()
                connection.close()
                        
        except oracledb.DatabaseError as e:
            print(f"Database error: {e}")
            raise
        except Exception as e:
            print("Error Vector Search:", e)
        
        return docs
