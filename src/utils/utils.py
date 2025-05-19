import base64
import datetime
import json
import os
import shutil
from dotenv import find_dotenv, load_dotenv
import oci
from oci.generative_ai_inference import GenerativeAiInferenceClient
from oci.generative_ai_inference.models import (
    EmbedTextDetails,
    OnDemandServingMode,
)
from oci.object_storage import ObjectStorageClient
from oci.ai_speech import AIServiceSpeechClient
from oci.ai_speech.models import (
  CreateTranscriptionJobDetails,
  ObjectListInlineInputLocation,
  ObjectLocation,
  OutputLocation,
  TranscriptionModelDetails,
  TranscriptionJobSummary,

)
from langchain_community.embeddings import OCIGenAIEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain_community.chat_models import ChatOCIGenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, SystemMessage
from langchain.prompts.chat import ChatPromptTemplate


_ = load_dotenv(find_dotenv())
OCI_CONFIG_FILE = "~/.oci/config"
OCI_GENAI_ENDPOINT = os.getenv("OCI_GENAI_ENDPOINT")
OCI_COMPARTMENT_ID = os.getenv("OCI_COMPARTMENT_ID")
OCI_OS_NAMESPACE = os.getenv("OCI_OS_NAMESPACE")
OCI_OS_BUCKET_NAME = os.getenv("OCI_OS_BUCKET_NAME")
OCI_OS_BUCKET_URL = os.getenv("OCI_OS_BUCKET_URL")

config = oci.config.from_file(file_location=OCI_CONFIG_FILE, profile_name="DEFAULT")
object_storage = ObjectStorageClient(config)
speech_client = AIServiceSpeechClient(config)
generative_ai_inference_client = GenerativeAiInferenceClient(
    config=config,
    service_endpoint="https://inference.generativeai.us-chicago-1.oci.oraclecloud.com"
    )   

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
  # chain = LLMChain(llm=llm, prompt=prompt)
  chain = ({"text": RunnablePassthrough() } 
           | prompt 
           | llm 
           | StrOutputParser()
  )
  result = chain.invoke(text)
  return result


def summarize_image_to_text(image_path: str, uploaded_url: str = None) -> str:
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
    model_id="meta.llama-3.2-90b-vision-instruct",
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

def chat_with_image(image_path: str, prompt: str, system_prompt: str = None) -> str:
  with open("/home/opc/multimodal_oci_genai/" + image_path, "rb") as img_file:
    image_data = base64.b64encode(img_file.read()).decode("utf-8")
  
  prompt_with_image = [
    SystemMessage(
        content=system_prompt
    ),
    HumanMessage(
        content=[
            {"type": "text", "text": prompt},
            {
              "type": "image_url",
              "image_url": {
                "url": "data:image/png;base64,"+image_data,
            }
        },
        ]
      )
  ]

  llm = ChatOCIGenAI(
      model_id="meta.llama-3.2-90b-vision-instruct",
      service_endpoint="https://inference.generativeai.us-chicago-1.oci.oraclecloud.com",
      compartment_id=OCI_COMPARTMENT_ID,
      )
  result = llm.invoke(prompt_with_image)
  print(f"Result: {result}") 
  return result.content
  

def chat_cohere(prompt: str, system_prompt: str = "") -> str:
  
  chat_prompt = ChatPromptTemplate.from_messages([
    ("system", "{system_prompt}"),
    ("human", "{input}"),
  ])


  llm = ChatOCIGenAI(
      model_id="cohere.command-r-08-2024",
      service_endpoint="https://inference.generativeai.us-chicago-1.oci.oraclecloud.com",
      compartment_id=OCI_COMPARTMENT_ID,
      )
  # chain = chat_prompt | llm | StrOutputParser()
  # result = chain.invoke({"input": prompt, "system_prompt": system_prompt})
  result = llm.invoke(chat_prompt)
  return result.content

def upload_image_to_oci(image_path, object_name=None):
  if not object_name:
    object_name = os.path.basename(image_path)

  with open(image_path, "rb") as f:
    data = f.read()

  object_storage.put_object(
    namespace_name=OCI_OS_NAMESPACE,
    bucket_name=OCI_OS_BUCKET_NAME,
    object_name=object_name,
    put_object_body=data
  )
  object_url = OCI_OS_BUCKET_URL + object_name
  return object_url

def download_image_from_oci(object_name: str, destination_path: str) -> str:
    response = object_storage.get_object(
        namespace_name=OCI_OS_NAMESPACE,
        bucket_name=OCI_OS_BUCKET_NAME,
        object_name=object_name
    )

    with open(destination_path, "wb") as out_file:
      for chunk in response.data.raw.stream(1024 * 1024, decode_content=False):
        out_file.write(chunk)

    return destination_path

def delete_files(directory: str, file_type: str) -> None:
    """ Delete files """
    files = os.listdir(directory)
    try:
        for file in files:
            if file.endswith(file_type):
                os.remove(os.path.join(directory, file))
    except FileNotFoundError:
        print("no target files")
    except Exception as e:
        print("Error delete files", e)

def dir_check(directory: str, file_type: str) -> None:
    """ Check directory """
    if os.path.exists(directory):
        try:
            delete_files(directory, file_type)
            print(f"success delete files in {directory}")
        except FileNotFoundError:
            print("no target files")
        except Exception as e:
            print("Error delete files", e)
    else:
        try:
            os.makedirs(directory, exist_ok=True)
        except Exception as e:
            print("Error make dirctory ", e)

def delete_dir(directory: str) -> None:
    """ Delete directory """
    if os.path.exists(directory):
        try:
            shutil.rmtree(directory)
            print("success delete directory")
        except Exception as e:
            print("Error delete directory", e)
            
            
## Whisper
def audio_to_text(uploaded_file_name: str):
  create_transcription_job_details = CreateTranscriptionJobDetails(
      compartment_id=OCI_COMPARTMENT_ID,
      display_name="PythonSDKSampleTranscriptionJob",
      description="Transcription job created by Python SDK",
      input_location=ObjectListInlineInputLocation(
          location_type="OBJECT_LIST_INLINE_INPUT_LOCATION",
          object_locations=[ObjectLocation(
              namespace_name=OCI_OS_NAMESPACE,
              bucket_name=OCI_OS_BUCKET_NAME,
              object_names=[uploaded_file_name])]),
      output_location=OutputLocation(
          namespace_name=OCI_OS_NAMESPACE, bucket_name=OCI_OS_BUCKET_NAME),
      model_details=TranscriptionModelDetails(
          model_type="WHISPER_MEDIUM", language_code="ja")
  )   
  response = speech_client.create_transcription_job(
      create_transcription_job_details)
  job_id = response.data.id


  job_finished = False
  while not job_finished:
    job_details = speech_client.get_transcription_job(job_id).data
    job_status = job_details.lifecycle_state
    if job_status in ["SUCCEEDED"]:
      job_finished = True
      output_location = job_details.output_location
      bucket_name = output_location.bucket_name
      namespace_name = output_location.namespace_name
      prefix = output_location.prefix
      object_storage_client = ObjectStorageClient(config)
      list_objects_response = object_storage_client.list_objects(
          namespace_name=namespace_name,
          bucket_name=bucket_name,
          prefix=prefix,
          fields="name"
      )
      file_name = list_objects_response.data.objects[0].name
      get_object_response = object_storage_client.get_object(
          namespace_name=namespace_name,
          bucket_name=bucket_name,
          object_name=file_name
      )
      file_content = get_object_response.data.text
      transcriptions_json = json.loads(file_content)
      print(f"Trascription result: {transcriptions_json['transcriptions']}")
      return transcriptions_json['transcriptions']
          
def embedding_image(image_path: str):
  with open("/home/opc/multimodal_oci_genai/" + image_path, "rb") as img_file:
    image_data = base64.b64encode(img_file.read()).decode("utf-8")
    data_uri = f"data:image/png;base64,{image_data}"

  print(f"data_uri: {data_uri}")
  model_id = "cohere.embed-multilingual-image-v3.0"
  embed_image_detail = EmbedTextDetails()
  embed_image_detail.serving_mode = OnDemandServingMode(model_id=model_id)
  embed_image_detail.compartment_id = OCI_COMPARTMENT_ID

  embed_image_detail.inputs = [data_uri]
  embed_image_detail.input_type = "IMAGE"
  embedding = generative_ai_inference_client.embed_text(embed_image_detail)
  print(f"Embedding result: {embedding.data.embeddings}")
  return embedding.data.embeddings
  


if __name__ == "__main__":
  # response = get_embedding(
  #     text="This is a test text for embedding."
  # )
  # print(response)
  
  # response = summarize_image_to_text(
  #     image_path="./images/製品概要_1_7.png",
  #     uploaded_url=None
  # )
  # print(response)
  
  # res = chat_with_image(
  #     image_path="images/net_income.png",
  #     prompt="Tell me the operating profit for FY24.",
  #     system_prompt="You are a helpful assistant."
  # )
  # print(res)
  
  res = embedding_image(
      image_path="images/製品概要_1_7.png"
  )
  print(res)
