"""
Agent for Code Q&A using MCP Server

This agent provides an interface for interacting with code repositories
through the MCP server.
"""

import json
import sys
from typing import Dict, TypedDict, Optional

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from code_qa.config.settings import get_settings

# Constants for prompts
ANALYSIS_PROMPT = """
You are analyzing a user's request about code or repository. The user might provide:
1. A repository URL (GitHub, GitLab, etc.) and a question about the code or repository
2. A local path and a question about the code or repository

Additionally, determine if the question is about:
a) Repository structure/statistics (e.g., "How many files are there?")
b) Code content/implementation (e.g., "How does this function work?")

Extract:
- repository_url: The repository URL or path (if provided)
- question: The specific question about the code
- question_type: Either "structure" or "code"

Respond in JSON format:
{
    "repository_url": "extracted_url_or_null",
    "question": "extracted_question",
    "question_type": "structure_or_code"
}
"""

RESPONSE_PROMPT = """
You are a helpful assistant. Based on the context provided below, 
provide a clear, well-structured answer to the user's question about the code repository.

User's Question: {question}
Repository: {repository_url}

Context:
{mcp_response}

Guidelines:
1. Be conversational while maintaining technical accuracy
2. If the user's question is about the repository structure, provide a summary of the repository structure.
3. If the user's question is about the code, provide a detailed explanation of the code.
4. If the user's question is about the repository, provide a summary of the repository.
5. Do not use your own knowledge to answer the question. Only use the context provided.
"""

class AgentState(TypedDict):
    """State for the LangGraph agent."""
    repository_url: Optional[str]
    current_question: Optional[str]
    response: Optional[str]
    question_type: Optional[str]

class CodeQAAgent:
    """LangGraph agent that uses MCP server for code Q&A."""
    
    def __init__(self, openai_api_key: str = None):
        """Initialize the Code Q&A agent."""
        self.settings = get_settings()
        self.openai_api_key = openai_api_key or self.settings.openai_api_key
        if not self.openai_api_key:
            raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY environment variable.")
        
        self.llm = ChatOpenAI(
            api_key=self.openai_api_key,
            model=self.settings.openai_model,
            temperature=self.settings.openai_temperature
        )
        
        self.mcp_server_path = self.settings.mcp_server_path
        self.graph = self._create_graph()
    
    def _create_graph(self) -> StateGraph:
        """Create the LangGraph workflow."""
        workflow = StateGraph(AgentState)
        
        workflow.add_node("analyze_question", self._analyze_question_node)
        workflow.add_node("query_mcp", self._query_mcp_node)
        workflow.add_node("generate_response", self._generate_response_node)
        
        workflow.set_entry_point("analyze_question")
        workflow.add_edge("analyze_question", "query_mcp")
        workflow.add_edge("query_mcp", "generate_response")
        workflow.add_edge("generate_response", END)
        
        return workflow.compile()
    
    async def _analyze_question_node(self, state: AgentState) -> Dict:
        """Analyze the user's question to extract repository URL and question."""
        question = state["current_question"]
        
        response = await self.llm.ainvoke([
            SystemMessage(content=ANALYSIS_PROMPT),
            HumanMessage(content=question)
        ])
        
        try:
            analysis = json.loads(response.content)
            return {
                "repository_url": analysis.get("repository_url") or state.get("repository_url"),
                "current_question": analysis.get("question", question),
                "question_type": analysis.get("question_type", "code"),
                "response": None
            }
        except json.JSONDecodeError:
            return {**state, "question_type": "code"}
    
    async def _query_mcp_node(self, state: AgentState) -> Dict:
        """Query the MCP server with the extracted question and repository."""
        if not state.get("repository_url") or not state.get("current_question"):
            return {**state, "response": "Error: Repository URL and question are required."}
        
        try:
            tool_name = "answer_repo_structure_question" if state["question_type"] == "structure" else "answer_code_question"
            mcp_response = await self._call_mcp_server(
                state["current_question"],
                state["repository_url"],
                tool_name
            )
            return {**state, "response": mcp_response}
        except Exception as e:
            return {**state, "response": f"Error querying MCP server: {str(e)}"}
    
    async def _generate_response_node(self, state: AgentState) -> Dict:
        """Generate a final response based on the MCP server results."""
        response = await self.llm.ainvoke([
            SystemMessage(content=RESPONSE_PROMPT.format(
                question=state.get("current_question", ""),
                repository_url=state.get("repository_url", ""),
                mcp_response=state.get("response", "")
            ))
        ])
        return {**state, "response": response.content}
    
    async def _call_mcp_server(self, question: str, repository_url: str, tool_name: str) -> str:
        """Call the MCP server with the question and repository URL."""
        server_cmd = [sys.executable, str(self.mcp_server_path)]
        
        async with stdio_client(StdioServerParameters(command=server_cmd[0], args=server_cmd[1:])) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                
                arguments = {
                    "question": question,
                    "repository_url": repository_url,
                }
                
                if tool_name == "answer_code_question":
                    arguments.update({
                        "max_results": self.settings.default_max_results,
                        "max_tokens": self.settings.default_max_tokens,
                    })
                
                result = await session.call_tool(tool_name, arguments=arguments)
                return result.content[0].text if result.content else "No response from MCP server"
    
    async def answer_question(self, question: str, repository_url: str = None) -> str:
        """Answer a single question about code."""
        final_state = await self.graph.ainvoke({
            "repository_url": repository_url,
            "current_question": question,
            "response": None
        })
        return final_state["response"] 