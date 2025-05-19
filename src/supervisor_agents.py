import os
from dotenv import find_dotenv, load_dotenv
from langgraph.prebuilt import create_react_agent
from langgraph_supervisor import create_supervisor
from langchain.chat_models import init_chat_model
from langchain_core.documents import Document
from langchain.prompts.chat import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.chat_models import ChatOCIGenAI

from langchain.tools import tool

from retriever.custom_retriever import CustomImageRetriever, CustomMarkdownRetriever
from utils.utils import chat_with_image

_ = load_dotenv(find_dotenv())
OCI_COMPARTMENT_ID = os.getenv("OCI_COMPARTMENT_ID")

model = ChatOCIGenAI(
    model_id="cohere.command-r-plus-08-2024",
    service_endpoint="https://inference.generativeai.us-chicago-1.oci.oraclecloud.com",
    compartment_id=OCI_COMPARTMENT_ID,
    model_kwargs={"temperature": 1.0, "max_tokens": 500},
    )



def text_retrieve(question: str) -> dict:
    """
    Retrieve documents based on the question.
    Arguments:
        question: str: Question to ask.
    """
    text_retriever = CustomMarkdownRetriever()
    retrieved_docs = text_retriever.invoke(question)
    context = "\n\n".join(doc.page_content for doc in retrieved_docs)
    return {
        "context": context,
    }


def chat_business(context: str, question: str) -> dict:
    """
    Chat with business related tasks.
    Arguments:
        context: str: Context to use.
        question: str: Question to ask.
    """
    
    prompt = ChatPromptTemplate([
        ("system", "あなたは質疑応答のAIアシスタントです。必ず日本語で答えてください。"),
        ("human", """
         以下のMarkdownのコンテキストに基づいて質問に答えてください。
         回答は数字だけを回答してください。
         ** 質問 **
          {question} 
          
        ** コンテキスト **
        """ + context),
    ])

    chain = {'question': RunnablePassthrough()} | prompt | model | StrOutputParser()
    return {
        "answer": chain.invoke(question),
        }
    

chat_business_agent = create_react_agent(
    model=model,
    tools=[text_retrieve, chat_business],
    prompt=(
        """あなたは質疑応答のAIアシスタントです。\n
        あなたは2つのツールを順番に呼び出す必要があります。\n
        Step 1, テキスト検索ツール。ドキュメント検索タスクをこのツールに割り当てます。\n
           引数のquestionには、ユーザーの質問をそのまま渡してください。\n
        Step 2, ビジネスチャットツール。ビジネス関連のタスクをこのツールに割り当てます。売上や経費に関するタスクを割り当ててください。\n
              引数のcontextには、Step 1の戻り値contextをそのまま渡してください。\n
              引数のquestionには、ユーザーの質問をそのまま渡してください。\n
        """
    ),
    name="Business Agent",
    debug=True,
)
result = chat_business_agent.invoke({
    "message": [
        {
            "role": "user",
            "content": "2025年のFacilityのTotalの経費を教えてください。",
        },
    ]
})

def image_retrieve(question: str) -> dict:
    """
    Retrieve image based on the question.
    Arguments:
        question: str: Question to ask.
    """
    image_retriever = CustomImageRetriever()
    retrieved_docs = image_retriever.invoke(question)

    return {
        "context": retrieved_docs[0].page_content,
        "image_path": retrieved_docs[0].metadata["file_path"]
    }


def chat_image(context: str, image_path: str, question: str) -> dict:
    """
    Chat with image related tasks.
    Arguments:
        context: str: Context to use.
        image_path: str: Image path to use.
        question: str: Question to ask.
    """
    chat_prompt = ChatPromptTemplate.from_messages([
        ("human", "以下の文章を英語に翻訳してください。解答だけを返してください。\n {input}"),

    ])
    en_question= model.invoke(chat_prompt.invoke({"input": question}))
    print(f"Question: {question}")
    print(f"Translated Question: {en_question.content}")
    prompt = f"""
        Please answer the following question based on the image.\n
        Be sure to answer based on the image and context.\n
        ** Question**
        {en_question.content}\n
        **Context**
        {context}\n
        """
    system_prompt = "You are a AI assistant. Please answer the question based on the image."
    print(f"image_path: {image_path}")
    response = chat_with_image(
        image_path=image_path,
        prompt=prompt,
        system_prompt=system_prompt
        )
    print(f"Response: {response}")
    answer_prompt = ChatPromptTemplate.from_messages([
        ("human", "以下の文章を日本語に翻訳してください。解答だけを返してください。\n {input}"),

    ])
    ja_answer = model.invoke(answer_prompt.invoke({"input": response.content}))
    return {"answer": ja_answer}

chat_image_agent = create_react_agent(
    model=model,
    tools=[image_retrieve, chat_image],
    prompt=(
        """あなたは質疑応答のAIアシスタントです。\n
        あなたは2つのツールを順番に呼び出す必要があります。\n
        Step 1, Image検索ツール。Image検索タスクをこのツールに割り当てます。\n
           引数のquestionには、ユーザーの質問をそのまま渡してください。\n
        Step 2, Imageチャットツール。Imageの説明関連のタスクをこのツールに割り当てます。画像に関するタスクを割り当ててください。\n
              引数のcontextには、Step 1の戻り値contextをそのまま渡してください。\n
              引数のimage_pathには、Step 1の戻り値image_pathをそのまま渡してください。\n
              引数のquestionには、受け取ったプロンプトをそのまま渡してください。\n
        """
    ),
    name="Business Agent",
    debug=True,
)
result = chat_image_agent.invoke({
    "message": [
        {
            "role": "user",
            "content": "2025年のFacilityのTotalの経費を教えてください。",
        },
    ]
})
print(result.content)



supervisor = create_supervisor(
    model=model,
    agents=[chat_business_agent, chat_image_agent],
    prompt=(
        "You are a chat agent. You can call two agents:\n"
        "- a chat business agent. Assign business-related tasks to this agent\n"
        "- a chat image agent. Assign image-related tasks to this agent\n"
        "Do not do any work yourself."
    ),
    add_handoff_back_messages=True,
    output_mode="full_history",
).compile()

# result = supervisor.invoke({
#     "message": [
#         {
#             "role": "user",
#             "content": "2025年のFacilityのTotalの経費を教えてください。",
#         },
#     ]
# })
# print(result)