"""Configuration management for the Code QA agent."""

import os
from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    """Application settings."""
    
    # OpenAI Configuration
    openai_api_key: str = os.getenv("OPENAI_API_KEY")
    openai_model: str = "gpt-4o-mini"
    openai_temperature: float = 0.1
    
    # MCP Server Configuration
    mcp_server_path: Path = Path(__file__).parent.parent / "servers" / "code_qa_mcp_server.py"
    
    # Repository Configuration
    max_returned_chunks: int = 5
    default_max_results: int = 5
    default_max_tokens: int = 2500
    # Each file should not exceed 500 tokens during reconstruction
    # 5 returned chunks, if each chunk is 500 tokens, then the total context should not exceed 2500 tokens
    max_token_per_file: int = 500
    
    # Vector Store Configuration
    embedding_model: str = "all-MiniLM-L6-v2"
    chunk_size: int = 1500
    
    # Paths
    base_dir: Path = Path(__file__).parent.parent.parent / "data"
    downloads_dir: Path = base_dir / "downloads"
    processed_dir: Path = base_dir / "processed"
    vector_db_dir: Path = base_dir / "vector_db"
    
    # Evaluation Configuration
    qa_pairs_dir: Path = "<file_path>"
    repo_url: str = "<file_path>"
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Create necessary directories
        for dir_path in [self.downloads_dir, self.processed_dir, self.vector_db_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

def get_settings() -> Settings:
    """Get application settings."""
    return Settings() 