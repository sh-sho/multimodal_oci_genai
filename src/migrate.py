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
from moviepy import  VideoFileClip, ColorClip, CompositeVideoClip, concatenate_videoclips
from moviepy.audio.io.AudioFileClip import AudioFileClip
import cv2

from utils.utils import dir_check, get_embedding, summarize_image_to_text, summarize_text, upload_image_to_oci

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
MOVIE_DIRECTORY_PATH = os.getenv("MOVIE_DIRECTORY_PATH")
SPLIT_MOVIE_DIRECTORY_PATH = os.getenv("SPLIT_MOVIE_DIRECTORY_PATH")


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


def extract_tables_and_images(file_path, output_dir="images"):
  os.makedirs(output_dir, exist_ok=True)
  wb = load_workbook(file_path)
  e_wb = pd.ExcelFile(file_path)
  result = []

  # check_sheets = {"Segmental info & Opex", "Corporate_Overview"}
  for sheet_name in wb.sheetnames:
    # if not sheet_name in check_sheets:
    #   continue
    ws = wb[sheet_name]
    print(f"worksheet: {ws}")
    # --- 表の抽出 ---
    df = pd.read_excel(e_wb, sheet_name=sheet_name)
    markdown = df.to_markdown(index=False)
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
        INSERT INTO image_contents (file_id, image_path, image_url, summary, embedding, image_blob)
        VALUES (:file_id, :image_path, :image_url, :summary, :embedding, empty_blob())
        returning image_blob into :blobdata
      """
  
  try:
    with oracledb.connect(user=UN, password=PW, dsn=DSN) as conn:
      with conn.cursor() as cursor:
        blobdata = cursor.var(oracledb.DB_TYPE_BLOB)
        params = {
          'file_id': file_id,
          'image_path': image_path,
          'image_url': image_url,
          'summary': summary,
          'embedding': str(embedding),
          'blobdata': blobdata
        }
        cursor.execute(sql, params)
        blob, = blobdata.getvalue()
        offset = 1
        bufsize = 65536
        with open(image_path, 'rb') as f:
            while True:
                data = f.read(bufsize)
                if data:
                    blob.write(data, offset)
                if len(data) < bufsize:
                    break
                offset += bufsize
        
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
  
  print(f"image_path: {image_path}")
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


# Movies
def resize_image(input_path, output_path, max_width=2560, max_height=1440):
  image = cv2.imread(input_path)
  if image is None:
    print(f"Failed to load image: {input_path}")
    return

  height, width = image.shape[:2]

  if width <= max_width and height <= max_height:
      print(f"Image is already within the size limits: {input_path}")
      cv2.imwrite(output_path, image)
      return

  aspect_ratio = width / height
  if width / max_width > height / max_height:
    new_width = max_width
    new_height = int(max_width / aspect_ratio)
  else:
    new_height = max_height
    new_width = int(max_height * aspect_ratio)

  resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)

  cv2.imwrite(output_path, resized_image)
  print(f"Resized image saved to: {output_path}")

def summarize_to_db_and_upload_movie(image_path, object_name=None):
  with open(image_path, "rb") as img_file:
    image_lob = img_file.read()
    print(f"type: {type(image_lob)}")
  
  uploaded_url = upload_image_to_oci(image_path, object_name)
  print(f"Uploaded Image URL: {uploaded_url}")
  
  image_summary = summarize_image_to_text(image_path, uploaded_url)
  print(f"Image Summary: {image_summary}")
  if image_summary is not None:
    image_embedding = get_embedding(image_summary)
  
  image_id = save_file_info(image_path)
  save_image_content(image_id, image_path, uploaded_url, image_summary, image_embedding)
  return {"summary": image_summary, "uploaded_url": uploaded_url}


def save_frames_at_intervals(
  video_path: str, 
  interval_sec: int, 
  output_dir: str, 
  base_filename: str
  ) -> None:
  try:
    video = VideoFileClip(video_path)
    duration_sec = video.duration
    current_sec = 0
    frame_idx = 0
    digit = len(str(int(duration_sec // interval_sec) + 1))

    while current_sec < duration_sec:
      frame = video.get_frame(current_sec)
      output_filename = '{}_{}_{:.2f}.png'.format(base_filename, str(frame_idx).zfill(digit), current_sec)
      output_path = os.path.join(output_dir, output_filename)

      frame_image = Image.fromarray(frame)
      frame_image.save(output_path)
      print(f"Saved frame at {current_sec} sec to {output_path}")
    
      current_sec += interval_sec
      frame_idx += 1

    video.close()

  except Exception as e:
    print(f"Error processing video {video_path}: {e}")

def split_movie(movie_file: str) -> None:
  try:
    if movie_file.endswith(".mp4"):
      movie_path = os.path.join(MOVIE_DIRECTORY_PATH, movie_file)
      base_filename = movie_file.replace(".mp4", "")
      save_frames_at_intervals(movie_path, 10, SPLIT_MOVIE_DIRECTORY_PATH, base_filename)
      print(f"split {MOVIE_DIRECTORY_PATH}/{movie_file}")
  except Exception as e:
    print(f"Error splitting movie {movie_file}: {e}")
  
def split_movies() -> None:
    dir_check(SPLIT_MOVIE_DIRECTORY_PATH, ".png")
    movie_files = os.listdir(MOVIE_DIRECTORY_PATH)
    print(movie_files)
    try:
      for movie_file in movie_files:
        split_movie(movie_file)
    except Exception as e:
      print("Error split movies", e)

def encode_image(image_path: str):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode("utf-8")

def process_movies() -> None:
  
  split_movies()

  for movie_file in os.listdir(SPLIT_MOVIE_DIRECTORY_PATH):
    if movie_file.endswith(".png"):
      image_path = os.path.join(SPLIT_MOVIE_DIRECTORY_PATH, movie_file)
      print(f"image_path: {image_path}")
      resize_image(
        input_path=os.path.join(image_path),
        output_path=os.path.join(image_path),
        max_width=2560,
        max_height=1440
      )
      
      summarize_to_db_and_upload_movie(image_path, object_name=os.path.basename(image_path))

def excel_markdown(file_path):
  wb = load_workbook(file_path)
  e_wb = pd.ExcelFile(file_path)
  result = []

  for sheet_name in wb.sheetnames:
    # if not sheet_name in check_sheets:
    #   continue
    ws = wb[sheet_name]
    print(f"worksheet: {ws}")
    df = pd.read_excel(e_wb, sheet_name=sheet_name)
    markdown = df.to_markdown(index=False)
    print(f"Markdown Table:\n{markdown}")
    result.append({
      "type": "table",
      "sheet": sheet_name,
      "markdown": markdown,
      "raw": df
    })
  return result

def process_excel(file_path: str) -> None:
  file_id = save_file_info(file_path)
  contents = excel_markdown(file_path)
  for content in contents:
    summary = summarize_text(content["markdown"])
    embedding = get_embedding(summary)
    save_docs_content(file_id, content["markdown"], summary, embedding)


if __name__ == "__main__":
  process_excel_with_images("./data/fy25q3-supplemental.xlsx")
  summarize_to_db_and_upload_image(image_path="./images/net_income.png", object_name="net_income.png")
  summarize_to_db_and_upload_image(image_path="./images/revenue.png", object_name="revenue.png")
