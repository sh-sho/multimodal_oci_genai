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
    Message,
    BaseChatRequest
)
from oci.generative_ai_inference import GenerativeAiInferenceClient
import requests
from langchain_community.chat_models import ChatOCIGenAI
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())

OCI_CONFIG_FILE = "~/.oci/config"
OCI_COMPARTMENT_ID = os.getenv("OCI_COMPARTMENT_ID")
OCI_GENAI_ENDPOINT = os.getenv("OCI_GENAI_ENDPOINT")
config = oci.config.from_file(file_location=OCI_CONFIG_FILE, profile_name="OSAKA")
generative_ai_inference_client = GenerativeAiInferenceClient(config, sevice_endpoint=OCI_GENAI_ENDPOINT)
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




AUTH_TYPE = "API_KEY" # The authentication type to use, e.g., API_KEY (default), SECURITY_TOKEN, INSTANCE_PRINCIPAL, RESOURCE_PRINCIPAL.
CONFIG_PROFILE = "OSAKA"

prompt="Oracleのクラウドについて教えてください"

# Service endpoint
endpoint = "https://inference.generativeai.ap-osaka-1.oci.oraclecloud.com"

# initialize interface
chat = ChatOCIGenAI(
  model_id="ocid1.generativeaimodel.oc1.ap-osaka-1.amaaaaaask7dceyac2pavq6pya22whj4gvy5l7mpdyrlm646dt7n3cppfxcq",
  service_endpoint=endpoint,
  compartment_id=OCI_COMPARTMENT_ID,
  provider="meta",
  model_kwargs={
    "temperature": 1,
    "max_tokens": 600,
    "frequency_penalty": 0,
    "presence_penalty": 0,
    "top_p": 0.75
  },
  auth_type=AUTH_TYPE,
  auth_profile=CONFIG_PROFILE
)

messages = [
  HumanMessage(content=prompt),
]

response = chat.invoke(messages)
res_chat = response.content
print(f"ChatOCIGenAI Response: {res_chat}")