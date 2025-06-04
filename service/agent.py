import os
from typing import Dict, Any, Optional, Union
from pathlib import Path
from dotenv import load_dotenv
from langgraph.prebuilt import create_react_agent
from langchain_community.agent_toolkits.load_tools import load_tools
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage
from langchain_core.language_models.chat_models import BaseChatModel
from dataclasses import dataclass
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()


@dataclass
class GraphQLConfig:
    """Configuration for GraphQL agent"""
    endpoint: str
    schema_path: str
    model_name: str = "gpt-4o"
    model_provider: str = "azure_openai"
    api_version: str = "2024-12-01-preview"
    introspection_disabled: bool = True
    
    @classmethod
    def from_env(cls) -> 'GraphQLConfig':
        """Create config from environment variables"""
        return cls(
            endpoint=os.getenv("GRAPHQL_ENDPOINT", "https://graphqlzero.almansi.me/api"),
            schema_path=os.getenv("GRAPHQL_SCHEMA_PATH", "service/graphqlv2schema.schema.graphql"),
            model_name=os.getenv("MODEL_NAME", "gpt-4o"),
            model_provider=os.getenv("MODEL_PROVIDER", "azure_openai"),
            api_version=os.getenv("API_VERSION", "2024-12-01-preview")
        )


class GraphQLAgent:
    """Enhanced GraphQL agent with better error handling and structure"""
    
    def __init__(self, config: Optional[GraphQLConfig] = None):
        """Initialize the GraphQL agent with configuration"""
        self.config = config or GraphQLConfig.from_env()
        self._model: Optional[BaseChatModel] = None
        self._agent = None
        self._schema_content: Optional[str] = None
        
    @property
    def model(self) -> BaseChatModel:
        """Lazy load the language model"""
        if self._model is None:
            try:
                self._model = init_chat_model(
                    self.config.model_name,
                    model_provider=self.config.model_provider,
                    api_version=self.config.api_version
                )
                logger.info(f"Initialized {self.config.model_name} model")
            except Exception as e:
                logger.error(f"Failed to initialize model: {e}")
                raise RuntimeError(f"Model initialization failed: {e}")
        return self._model
    
    @property
    def schema_content(self) -> str:
        """Load and cache the GraphQL schema"""
        if self._schema_content is None:
            self._schema_content = self._load_schema()
        return self._schema_content
    
    def _load_schema(self) -> str:
        """Load GraphQL schema from file with error handling"""
        schema_path = Path(self.config.schema_path)
        
        if not schema_path.exists():
            error_msg = f"Schema file not found: {schema_path}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)
        
        try:
            with open(schema_path, "r", encoding="utf-8") as f:
                content = f.read()
            
            if not content.strip():
                raise ValueError("Schema file is empty")
            
            logger.info(f"Loaded GraphQL schema from {schema_path}")
            return content
            
        except Exception as e:
            logger.error(f"Error reading schema file: {e}")
            raise RuntimeError(f"Failed to read schema: {e}")
    
    @property
    def agent(self):
        """Lazy load the agent with tools"""
        if self._agent is None:
            try:
                # Load GraphQL tools
                tools = load_tools(
                    ["graphql"],
                    graphql_endpoint=self.config.endpoint,
                    graphql_sdl_schema_endpoint=os.getenv("GRAPHQL_SCHEMA_URL"),
                    graphql_introspection_disabled=self.config.introspection_disabled,
                    graphql_sdl_schema_content=self.schema_content,
                )
                
                # Create the agent
                self._agent = create_react_agent(self.model, tools)
                logger.info("Created GraphQL agent with tools")
                
            except Exception as e:
                logger.error(f"Failed to create agent: {e}")
                raise RuntimeError(f"Agent creation failed: {e}")
                
        return self._agent
    
    def _create_messages(self, query: str) -> list[BaseMessage]:
        """Create system and user messages for the agent"""
        system_message = SystemMessage(
            content=(
                "You are a specialized GraphQL agent with the following capabilities:\n"
                "1. Parse and understand user requests related to GraphQL operations\n"
                "2. Translate natural language queries into proper GraphQL syntax\n"
                "3. Execute GraphQL queries and mutations using the provided schema\n"
                "4. Handle errors gracefully and provide helpful feedback\n\n"
                "Important notes:\n"
                "- Introspection is disabled; use only the provided schema\n"
                "- Validate queries against the schema before execution\n"
                "- Return structured results with clear formatting\n"
                "- If a query fails, explain why and suggest corrections"
            )
        )
        
        user_message = HumanMessage(content=query)
        
        return [system_message, user_message]
    
    def execute_query(self, query: str) -> Dict[str, Any]:
        """
        Execute a GraphQL query through the agent
        
        Args:
            query: Natural language query or GraphQL query string
            
        Returns:
            Dictionary containing the result or error information
        """
        try:
            # Validate input
            if not query or not query.strip():
                return {
                    "success": False,
                    "error": "Query cannot be empty",
                    "data": None
                }
            
            # Create messages
            messages = self._create_messages(query)
            
            # Execute through agent
            logger.info(f"Executing query: {query[:100]}...")
            result = self.agent.invoke({"messages": messages})
            
            # Extract the response
            response = self._extract_response(result)
            
            return {
                "success": True,
                "data": response,
                "error": None
            }
            
        except FileNotFoundError as e:
            return {
                "success": False,
                "error": f"Configuration error: {str(e)}",
                "data": None
            }
        except RuntimeError as e:
            return {
                "success": False,
                "error": f"Runtime error: {str(e)}",
                "data": None
            }
        except Exception as e:
            logger.error(f"Unexpected error executing query: {e}")
            return {
                "success": False,
                "error": f"Unexpected error: {str(e)}",
                "data": None
            }
    
    def _extract_response(self, result: Dict[str, Any]) -> Any:
        """Extract the relevant response from agent result"""
        if isinstance(result, dict):
            # Check for messages in the result
            if "messages" in result and result["messages"]:
                last_message = result["messages"][-1]
                if hasattr(last_message, "content"):
                    return last_message.content
            
            # Check for direct output
            if "output" in result:
                return result["output"]
        
        # Return raw result if structure is unexpected
        return result
    
    def validate_schema(self) -> bool:
        """Validate that the schema can be loaded and is valid"""
        try:
            _ = self.schema_content
            return True
        except Exception as e:
            logger.error(f"Schema validation failed: {e}")
            return False


# Backward compatible function
def call_graphql_api(query: str) -> Union[Dict[str, Any], str]:
    """
    Backward compatible function to call GraphQL API
    
    Args:
        query: The GraphQL query or natural language request
        
    Returns:
        Result dictionary or error string
    """
    try:
        agent = GraphQLAgent()
        result = agent.execute_query(query)
        
        if result["success"]:
            return result["data"]
        else:
            return f"Error: {result['error']}"
            
    except Exception as e:
        logger.error(f"Error in call_graphql_api: {e}")
        return f"Error: {str(e)}"


# Example usage and testing
if __name__ == "__main__":
    # Example 1: Using the class directly
    agent = GraphQLAgent()
    
    # Validate schema first
    if agent.validate_schema():
        print("Schema validated successfully")
    
    # Execute a query
    result = agent.execute_query("Get all users with their posts")
    print(f"Result: {result}")
    
    # Example 2: Using the backward compatible function
    response = call_graphql_api("Get user with id 1")
    print(f"Response: {response}")