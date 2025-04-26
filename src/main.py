import os
import oci
from oci.generative_ai_inference.models import (
    ChatDetails, 
    CohereChatRequest, 
    OnDemandServingMode, 
    CohereTool, 
    CohereParameterDefinition,
    LlamaLlmInferenceRequest,
    GenericChatRequest,
    TextContent,
    ImageContent,
    ImageUrl,
    Message,
    ChatContent,
    BaseChatRequest
)
from oci.generative_ai_inference import GenerativeAiInferenceClient
from oci.object_storage import ObjectStorageClient
import oracledb
import requests
from langchain_community.chat_models import ChatOCIGenAI
from langchain_core.messages import HumanMessage
from langchain_core.prompts import PromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain_community.llms.oci_generative_ai import OCIGenAI
from langchain.chains.llm import LLMChain


from langchain_core.documents import Document
from langchain_community.embeddings import OCIGenAIEmbeddings
import pandas as pd
from openpyxl import load_workbook
from openpyxl.drawing.image import Image as XLImage
from PIL import Image
import io
import os
import base64
from tabulate import tabulate
from dotenv import load_dotenv, find_dotenv

from utils.utils import get_embedding, summarize_image_to_text, summarize_text, upload_image_to_oci

_ = load_dotenv(find_dotenv())
oracledb.init_oracle_client()

UN = os.environ.get("UN")
PW = os.environ.get("PW")
DSN = os.environ.get("DSN")

OCI_CONFIG_FILE = "~/.oci/config"
OCI_COMPARTMENT_ID = os.getenv("OCI_COMPARTMENT_ID")
OCI_GENAI_ENDPOINT = os.getenv("OCI_GENAI_ENDPOINT")
OCI_OS_NAMESPACE = os.getenv("OCI_OS_NAMESPACE")
OCI_OS_BUCKET_NAME = os.getenv("OCI_OS_BUCKET_NAME")
OCI_OS_BUCKET_URL = os.getenv("OCI_OS_BUCKET_URL")

config_osaka = oci.config.from_file(file_location=OCI_CONFIG_FILE, profile_name="OSAKA")
config = oci.config.from_file(file_location=OCI_CONFIG_FILE, profile_name="DEFAULT")
generative_ai_inference_client = GenerativeAiInferenceClient(config_osaka, sevice_endpoint=OCI_GENAI_ENDPOINT)
model_id="ocid1.generativeaimodel.oc1.ap-osaka-1.amaaaaaask7dceyayobenuynf4om42tyrr7scxwijpzwfajc6i6w5wjcmjbq"

preamble = """
## 回答のスタイル
日本語で回答してください。
質問に対してできる限り詳細な回答をしてください。
"""
input_text = "Oracleのクラウドについて教えてください"

# chat_detail = ChatDetails(
#     chat_request=CohereChatRequest(
#         preamble_override=preamble,
#         message=input_text,
#         max_tokens=500,
#         is_force_single_step=False,
#         ),
#     compartment_id=OCI_COMPARTMENT_ID,
#     serving_mode=OnDemandServingMode(
#         model_id=model_id
#     ))
# chat_response = generative_ai_inference_client.chat(chat_detail)

# res_chat = chat_response.data.chat_response
# print(f"Cohere Response: {res_chat}")



# generative_ai_inference_client = GenerativeAiInferenceClient(config=config, service_endpoint=OCI_GENAI_ENDPOINT, retry_strategy=oci.retry.NoneRetryStrategy(), timeout=(10,240))
# chat_detail = ChatDetails()

# content = TextContent()
# content.text = input_text
# message = Message()
# message.role = "USER"
# message.content = [content]

# chat_request = GenericChatRequest()
# chat_request.api_format = BaseChatRequest.API_FORMAT_GENERIC
# chat_request.messages = [message]
# chat_request.max_tokens = 600
# chat_request.temperature = 1
# chat_request.frequency_penalty = 0
# chat_request.presence_penalty = 0
# chat_request.top_p = 0.75

# chat_detail.serving_mode = OnDemandServingMode(model_id="ocid1.generativeaimodel.oc1.ap-osaka-1.amaaaaaask7dceyac2pavq6pya22whj4gvy5l7mpdyrlm646dt7n3cppfxcq")
# chat_detail.chat_request = chat_request
# chat_detail.compartment_id = OCI_COMPARTMENT_ID

# chat_response = generative_ai_inference_client.chat(chat_detail)
# res_chat = chat_response.data.chat_response
# print(f"Llama Response: {res_chat}")




# AUTH_TYPE = "API_KEY"
# CONFIG_PROFILE = "OSAKA"

# prompt="Oracleのクラウドについて教えてください"

# # Service endpoint
# endpoint = "https://inference.generativeai.ap-osaka-1.oci.oraclecloud.com"

# # initialize interface
# chat = ChatOCIGenAI(
#   model_id="ocid1.generativeaimodel.oc1.ap-osaka-1.amaaaaaask7dceyac2pavq6pya22whj4gvy5l7mpdyrlm646dt7n3cppfxcq",
#   service_endpoint=endpoint,
#   compartment_id=OCI_COMPARTMENT_ID,
#   provider="meta",
#   model_kwargs={
#     "temperature": 1,
#     "max_tokens": 600,
#     "frequency_penalty": 0,
#     "presence_penalty": 0,
#     "top_p": 0.75
#   },
#   auth_type=AUTH_TYPE,
#   auth_profile=CONFIG_PROFILE
# )

# messages = [
#   HumanMessage(content=prompt),
# ]

# response = chat.invoke(messages)
# res_chat = response.content
# print(f"ChatOCIGenAI Response: {res_chat}")




def extract_tables_and_images(file_path, output_dir="images"):
  os.makedirs(output_dir, exist_ok=True)
  wb = load_workbook(file_path)
  result = []

  for sheet_name in wb.sheetnames:
    ws = wb[sheet_name]
    print(f"worksheet: {ws}")
    # --- 表の抽出 ---
    df = pd.read_excel(file_path, sheet_name=sheet_name)
    if not df.empty:
      markdown = tabulate(df.head(10), headers='keys', tablefmt='github')
      print(f"Markdown Table:\n{markdown}")
      result.append({
        "type": "table",
        "sheet": sheet_name,
        "markdown": markdown,
        "raw": df
      })

    # --- 画像の抽出 ---
    for image in ws._images:
      if isinstance(image, XLImage):
        img_bytes = image._data()
        img = Image.open(io.BytesIO(img_bytes))

        # ファイル名決定
        img_name = f"{sheet_name}_{image.anchor._from.row}_{image.anchor._from.col}.png"
        img_path = os.path.join(output_dir, img_name)
        img.save(img_path)
        print(f"Image Path: {img_path}")

        # Base64に変換（任意）
        with open(img_path, "rb") as f:
          base64_str = base64.b64encode(f.read()).decode("utf-8")
          data_url = f"data:image/png;base64,{base64_str}"

        result.append({
          "type": "image",
          "sheet": sheet_name,
          "image_path": img_path,
          "image_url": data_url
        })
  return result

def save_file_info(file_path):
  try:
    with oracledb.connect(user=UN, password=PW, dsn=DSN) as conn:
      cur = conn.cursor()
      file_name = file_path.split("/")[-1]
      file_id_var = cur.var(oracledb.NUMBER)
      sql = """INSERT INTO uploaded_files (file_name, file_path) VALUES (:1, :2) RETURNING id INTO :3"""
      data = [file_name, file_path, file_id_var]
      cur.execute(sql, data)
      file_id = file_id_var.getvalue()[0]
      conn.commit()
      return file_id
  except oracledb.DatabaseError as e:
    error, = e.args
    print(f"Error at save_file_info")
    print(f"Oracle error code: {error.code}")
    print(f"Oracle error message: {error.message}")
    return None

def save_docs_content(file_id, markdown, summary, embedding):
  sql = """
          INSERT INTO docs_contents (file_id, summary, embedding, markdown)
          VALUES (:file_id, :summary, :embedding, :markdown)
          """
  params = {
    "file_id": file_id,
    "summary": summary,
    "embedding": str(embedding),
    "markdown": markdown,
  }
  try:
    with oracledb.connect(user=UN, password=PW, dsn=DSN) as conn:
      with conn.cursor() as cursor:
        cursor.execute(sql, params)
        print(f"Success insert {file_id} into docs_contents")
        conn.commit()
  except oracledb.DatabaseError as e:
    error, = e.args
    print(f"Error at save_content")
    print(f"Oracle error code: {error.code}")
    print(f"Oracle error message: {error.message}")
  except Exception as e:
    print(f"Error:save_docs_content: {e}")
    return None

def save_image_content(file_id, image_path, image_url, summary, embedding):
  sql = """
        INSERT INTO image_contents (file_id, image_url, summary, embedding, image_blob)
        VALUES (:file_id, :image_url, :summary, :embedding, empty_blob())
        returning image_blob into :blobdata
      """
  with open(image_path, "rb") as img_file:
    image_blob = base64.b64encode(img_file.read()).decode("utf-8")
  
  try:
    with oracledb.connect(user=UN, password=PW, dsn=DSN) as conn:
      with conn.cursor() as cursor:
        blobdata = cursor.var(oracledb.DB_TYPE_BLOB)
        params = {
          'file_id': file_id,
          'image_url': image_url,
          'summary': summary,
          'embedding': str(embedding),
          'blobdata': blobdata
        }
        cursor.execute(sql, params)
        blobdata.setvalue(0, image_blob)
        
        print(f"Success insert {file_id} into image_contents")
        conn.commit()
  except oracledb.DatabaseError as e:
    error, = e.args
    print(f"Error at save_content")
    print(f"Oracle error code: {error.code}")
    print(f"Oracle error message: {error.message}")
  except Exception as e:
    print(f"Error:save_image_content: {e}")
    return None

def summarize_to_db_and_upload_image(image_path, object_name=None):
  with open(image_path, "rb") as img_file:
    image_lob = img_file.read()
    print(f"type: {type(image_lob)}")
  
  uploaded_url = upload_image_to_oci(image_path, object_name)
  print(f"Uploaded Image URL: {uploaded_url}")
  
  image_summary = summarize_image_to_text(image_path, uploaded_url)
  print(f"Image Summary: {image_summary}")
  image_embedding = get_embedding(image_summary)
  
  image_id = save_file_info(image_path)
  save_image_content(image_id, image_path, uploaded_url, image_summary, image_embedding)
  return {"summary": image_summary, "uploaded_url": uploaded_url}

def process_excel_with_images(file_path):
  file_id = save_file_info(file_path)
  contents = extract_tables_and_images(file_path)

  for content in contents:
    if content["type"] == "table":
      print(f"content_type:table: {content["type"]}")
      summary = summarize_text(content["markdown"])
      embedding = get_embedding(summary)
      save_docs_content(file_id, content["markdown"], summary, embedding)

    elif content["type"] == "image":
      print(f"content_type:image: {content["type"]}")
      res = summarize_to_db_and_upload_image(
        image_path=content["image_path"], 
        object_name=os.path.basename(content["image_path"])
        )


if __name__ == "__main__":
  process_excel_with_images("./data/sample_sales_infos.xlsx")
  # t_url = upload_image_to_oci("./data/sample_sales_infos.xlsx", "sample_sales_infos.xlsx")
  # print(t_url)
  # d_name=download_image_from_oci("sample_sales_infos.xlsx", "./data/tmp_sample_sales_infos.xlsx")
  # print(d_name)