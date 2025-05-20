import base64
import os
from dotenv import find_dotenv, load_dotenv
from langchain_community.embeddings import OCIGenAIEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain_community.chat_models import ChatOCIGenAI
from langchain_core.output_parsers import StrOutputParser


_ = load_dotenv(find_dotenv())
OCI_COMPARTMENT_ID = os.getenv("OCI_COMPARTMENT_ID")

def get_embedding(text: str) -> list:
  embeddings = OCIGenAIEmbeddings(
    model_id="cohere.embed-multilingual-v3.0",
    service_endpoint="https://inference.generativeai.us-chicago-1.oci.oraclecloud.com",
    compartment_id=OCI_COMPARTMENT_ID,
  )
  return embeddings.embed_query(text)


def summarize_text(text: str) -> str:
  prompt = PromptTemplate(
      input_variables=["text"],
      template="以下の内容を要約してください:\n{text}"
  )
  llm = ChatOCIGenAI(
    model_id="cohere.command-r-08-2024",
    service_endpoint="https://inference.generativeai.us-chicago-1.oci.oraclecloud.com",
    compartment_id=OCI_COMPARTMENT_ID,
    model_kwargs={"temperature": 0.7, "max_tokens": 500},
    )

  chain = ({"text": RunnablePassthrough() } 
           | prompt 
           | llm 
           | StrOutputParser()
  )
  result = chain.invoke(text)
  return result


def summarize_image_to_text(image_path: str) -> str:
  with open(image_path, "rb") as img_file:
    image_data = base64.b64encode(img_file.read()).decode("utf-8")

  prompt = {
    "role": "user",
    "content": [
        {
            "type": "text",
            "text": "Summarize the following this image. Answer in English.",
        },
        {
            "type": "image_url",
            "image_url": {
              "url": "data:image/png;base64,"+image_data,
            }
        },
    ],
  }
  llm = ChatOCIGenAI(
    model_id="meta.llama-4-scout-17b-16e-instruct",
    service_endpoint="https://inference.generativeai.us-chicago-1.oci.oraclecloud.com",
    compartment_id=OCI_COMPARTMENT_ID,
    )
  try:
      result = llm.invoke([prompt])
      print(f"Image Summary: {result}")
      return result.content
  except Exception as e:
      print(f"Error in image summarization: {e}")
      return None


if __name__ == "__main__":
  print("test")
