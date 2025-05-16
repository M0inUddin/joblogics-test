import os
from dotenv import load_dotenv
from langgraph.prebuilt import create_react_agent
from utils.logger import logger
from langchain_community.agent_toolkits.load_tools import load_tools
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage
from graphql import build_schema, parse
import requests

load_dotenv()


# def load_sdl_schema():
#     response = requests.get(
#         os.getenv("GRAPHQL_SCHEMA_URL") or "http://localhost:8000/graphql"
#     )
#     response.raise_for_status()
#     return response.text


# def get_graphql_schema():
#     sdl = load_sdl_schema()
#     return build_schema(sdl)


def call_graphql_api(query: str):
    try:
        model = init_chat_model(
            "gpt-4o", model_provider="azure_openai", api_version="2024-12-01-preview"
        )

        # Read the local SDL schema file content
        # schema_path = "service/graphqlv2schema.schema.graphql"
        # with open(schema_path, "r", encoding="utf-8") as f:
        #     graphql_schema_sdl = f.read()

        # Load GraphQL tools with explicit schema content
        tools = load_tools(
            ["graphql"],
            graphql_endpoint="https://graphqlzero.almansi.me/api",
            # graphql_sdl_schema_endpoint=os.getenv("GRAPHQL_SCHEMA_URL"),
            # graphql_introspection_disabled=True,
            # graphql_sdl_schema_content=graphql_schema_sdl,
        )

        agent = create_react_agent(model, tools)

        system = SystemMessage(
            content=(
                "You are a GraphQL agent. "
                "Introspection is disabled; you must use the provided schema. "
                "Translate the user's request into the correct GraphQL query or mutation, "
                "call the GraphQL tool, and return the tool's results."
            )
        )
        user = HumanMessage(content=query)

        # 7. send both to the agent
        result = agent.invoke({"messages": [system, user]})

        # 8. extract and return the final assistant message
        return result

    except Exception as e:
        logger.error(f"Error initializing agent: {e}")
        return f"Error: {e}"
