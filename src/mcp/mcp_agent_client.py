import os
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
import asyncio
from dotenv import find_dotenv, load_dotenv

from langchain_mcp_adapters.tools import load_mcp_tools
from langgraph.prebuilt import create_react_agent

from langchain_community.chat_models.oci_generative_ai import ChatOCIGenAI
_ = load_dotenv(find_dotenv())
OCI_COMPARTMENT_ID = os.getenv("OCI_COMPARTMENT_ID")

llm = ChatOCIGenAI(
    model_id="cohere.command-a-03-2025",
    service_endpoint="https://inference.generativeai.us-chicago-1.oci.oraclecloud.com",
    compartment_id=OCI_COMPARTMENT_ID,
    model_kwargs={"temperature": 0,"max_tokens": 4000}
    )

server_params = StdioServerParameters(
    command="python",
    args=["src/retrieve_answer_with_markdown.py"],
)

async def main():
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            # Initialize the connection
            await session.initialize()

            tools = await load_mcp_tools(session)

            agent = create_react_agent(
                model=llm,
                tools=tools,
                prompt=(
                    "You are a chat agent. You can call two tools:\n"
                    "- a get_text_with_markdown. Assign getting department expenses, sales, and operating income to this tool\n"
                    "- a get_text_with_image. Assign getting revenue or net income for the past years tasks to this tool\n"
                    "Do not do any work yourself."
                    "If you need to use a tool, please use the tool and return the result.\n"
                    "You must answer in Japanese.\n"
                ),
                debug=True,
                )
            agent_response = await agent.ainvoke({"messages": "2024/5のQ4のFacilityの経費を教えてください。"})
            print("Response: ",agent_response['messages'][-1].content)

if __name__ == "__main__":
    asyncio.run(main())