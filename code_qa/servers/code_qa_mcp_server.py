#!/usr/bin/env python3
"""
MCP Server for Code Q&A with Repository Processing

This server provides tools for:
1. Downloading and processing code repositories
2. Creating semantic code chunks using tree-sitter
3. Storing chunks in a vector database for efficient retrieval
4. Answering questions about code using RAG
"""

from typing import Dict

import mcp
import mcp.server.stdio

from mcp.server.fastmcp import FastMCP

from code_qa.scripts.vector_database import CodeVectorStore
from code_qa.config.settings import get_settings
from code_qa.servers.utils import process_repository, get_repo_id, generate_repo_analysis

import logging

logger = logging.getLogger(__name__)

# Get application settings
settings = get_settings()

# Create the MCP server
mcp = FastMCP("Code QA Server")

# Cache to store vector store instances and processed URLs to avoid re-processing the same repository
vector_stores: Dict[str, CodeVectorStore] = {}
processed_repos: Dict[str, str] = {}  # URL -> repo_id mapping for caching

# Base directories for data storage - use from settings
BASE_DIR = settings.base_dir
DOWNLOADS_DIR = settings.downloads_dir
PROCESSED_DIR = settings.processed_dir
VECTOR_DB_DIR = settings.vector_db_dir

# Create base directories if they don't exist
for dir_path in [DOWNLOADS_DIR, PROCESSED_DIR, VECTOR_DB_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

@mcp.tool()
def answer_code_question(question: str, repository_url: str, 
                        max_results: int = None, 
                        max_tokens: int = None) -> str:
    """
    Answer a question about code in a repository.
    
    Args:
        question: The question to answer about the code
        repository_url: URL or path to the repository (Git URL, HTTP archive, or local path)
        max_results: Maximum number of search results to consider
        max_tokens: Maximum tokens for reconstructed code context
    
    Returns:
        A formatted response with relevant code context and analysis
    """
    # Use settings for default values
    max_results = max_results or settings.default_max_results
    max_tokens = max_tokens or settings.default_max_tokens
    
    repo_id = get_repo_id(repository_url)
    vector_db_path = VECTOR_DB_DIR / repo_id
    logger.info(f"Vector DB path: {vector_db_path}")
    logger.info(f"Processed repos: {processed_repos}")
    logger.info(f"VB exists: {vector_db_path.exists()}")
    try:
        # First check if we have this repo in memory cache
        if repository_url in processed_repos:
            repo_id = processed_repos[repository_url]
            vector_store = vector_stores[repo_id]
            logger.info(f"Using in-memory cache for repository: {repository_url}")
        # Then check if we have it persisted on disk
        elif vector_db_path.exists():
            logger.info(f"Loading persisted vector store for repository: {repository_url}")
            vector_store = CodeVectorStore(
                db_path=str(vector_db_path),
                collection_name=f"repo_{repo_id}",
                embedding_model=settings.embedding_model
            )
            # Cache it in memory for future use
            vector_stores[repo_id] = vector_store
            processed_repos[repository_url] = repo_id
        else:
            # Download and process the repository
            logger.info(f"Processing new repository: {repository_url}")
            vector_store = process_repository(repository_url)
            vector_stores[repo_id] = vector_store
            processed_repos[repository_url] = repo_id
        
        # Search for relevant code chunks
        search_results = vector_store.search(question, n_results=max_results)
        
        if not search_results:
            return f"No relevant code found for the question: {question}"
        
        # Reconstruct relevant files with context
        reconstructed_files = vector_store.search_and_reconstruct(
            query=question,
            n_results=max_results,
            max_token_limit=max_tokens
        )
        
        return reconstructed_files
    except Exception as e:
        return f"Error: {str(e)}"

@mcp.tool()
def answer_repo_structure_question(repository_url: str) -> str:
    """
    Answer questions about repository structure and statistics using the analysis report.
    
    Args:
        question: The question about repository structure or statistics
        repository_url: URL or path to the repository
    
    Returns:
        A response based on the repository analysis report
    """
    try:
        analysis_report = generate_repo_analysis(repository_url)
            
        return f"Based on the repository analysis:\n\n{analysis_report}"
        
    except Exception as e:
        return f"Error analyzing repository structure: {str(e)}"

if __name__ == "__main__":
    mcp.run(transport='stdio') 