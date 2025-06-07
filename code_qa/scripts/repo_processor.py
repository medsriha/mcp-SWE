#!/usr/bin/env python3
"""
Enhanced Repository Processor with Tree-Sitter Chunking

This enhanced version of the repository processor integrates tree-sitter based
chunking for Python files, creating optimal chunks for RAG (Retrieval Augmented Generation).

Features:
- Tree-sitter based semantic chunking for Python files
- Optimized chunk sizes for embedding models
"""

import os
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any

import logging

from code_qa.scripts.python_chunker import PythonChunker

logger = logging.getLogger(__name__)


class RepoProcessor:
    """Enhanced processor with tree-sitter based chunking for Python files."""
    
    def __init__(self, 
                 repo_path: str, 
                 output_path: str, 
                 max_chunk_size: int):
        """
        Initialize the enhanced repository processor.
        
        Args:
            repo_path: Path to the repository to process
            output_path: Path for output files
            max_chunk_size: Maximum characters per chunk.
                Chunk count is determined using gpt tokenizer.
        """
        self.repo_path = Path(repo_path)
        self.output_path = Path(output_path)
        self.max_chunk_size = max_chunk_size
        
        self.python_chunker = PythonChunker()
        logger.info(f"Initialized repository processor for: {self.repo_path}")
        
        # File extension mappings
        self.code_extensions = {
            '.py': 'python',
            '.js': 'javascript',
            '.ts': 'typescript',
            '.jsx': 'javascript',
            '.tsx': 'typescript',
            '.java': 'java',
            '.cpp': 'cpp',
            '.c': 'c',
            '.h': 'c',
            '.cs': 'csharp',
            '.php': 'php',
            '.rb': 'ruby',
            '.go': 'go',
            '.rs': 'rust',
            '.kt': 'kotlin',
            '.swift': 'swift',
            '.scala': 'scala',
            '.html': 'html',
            '.css': 'css',
            '.scss': 'scss',
            '.sass': 'sass',
            '.vue': 'vue',
            '.sql': 'sql',
            '.sh': 'shell',
            '.bash': 'shell'
        }
        
        # Track all chunks
        self.all_chunks: List[Dict[str, Any]] = []
    
    def should_process_file(self, file_path: Path) -> bool:
        """Check if file should be processed based on extension only."""
        return not file_path.is_dir() and file_path.suffix.lower() in self.code_extensions
    
    def get_file_language(self, file_path: Path) -> str:
        """Determine the programming language of a file."""
        extension = file_path.suffix.lower()
        return self.code_extensions.get(extension, None)
    
    def read_file_content(self, file_path: Path) -> str:
        """Read file content as text."""
        encodings = ['utf-8', 'latin-1', 'cp1252']
        
        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    content = f.read()
                logger.debug(f"Successfully read {file_path} with {encoding} encoding")
                return content
            except UnicodeDecodeError:
                logger.debug(f"Failed to read {file_path} with {encoding} encoding")
                continue
        
        error_msg = f"[ERROR: Could not decode file {file_path.name}]"
        logger.error(f"Could not decode file {file_path} with any encoding")
        return error_msg
    
    def process_python_file(self, file_path: Path, content: str) -> List[Dict[str, Any]]:
        """Process a Python file using tree-sitter chunking."""
        try:
            logger.debug(f"Processing Python file with tree-sitter: {file_path}")
            # Use the Python chunker to create semantic chunks
            code_chunks = self.python_chunker.chunk_code(content, str(file_path))
            
            processed_chunks = []
            for chunk in code_chunks:
                chunk_data = {
                    'content': chunk.content,
                    'chunk_type': chunk.chunk_type,  # Already a string
                    'file_path': str(file_path.relative_to(self.repo_path)),
                    'language': 'python',
                    
                    # Position information
                    'start_line': chunk.start_line,
                    'end_line': chunk.end_line,
                    
                    # Essential metadata for RAG
                    'function_name': chunk.function_name,
                    'class_name': chunk.class_name,
                    'docstring': chunk.docstring,
                }
                
                processed_chunks.append(chunk_data)
            
            logger.debug(f"Created {len(processed_chunks)} chunks from Python file {file_path}")
            return processed_chunks
            
        except Exception as e:
            logger.error(f"Error processing Python file {file_path}: {e}")
            logger.debug(f"Falling back to text chunking for {file_path}")
            # Fall back to simple text chunking
            return []
    

    def process_file(self, file_path: Path) -> None:
        """Process a single file and add chunks to collection."""
        relative_path = file_path.relative_to(self.repo_path)
        logger.info(f"Processing file: {relative_path}")
        
        content = self.read_file_content(file_path)
        language = self.get_file_language(file_path)
        
        # Create chunks based on file type
        if language == 'python' and content.strip():
            chunks = self.process_python_file(file_path, content)
        
            # Store chunks globally
            self.all_chunks.extend(chunks)
            logger.debug(f"Added {len(chunks)} chunks from {relative_path}")
    
    def collect_files(self) -> List[Path]:
        """Collect all processable files based on extensions only."""
        logger.info("Collecting processable files from repository")
        files = []
        
        for file_path in self.repo_path.rglob('*'):
            if self.should_process_file(file_path):
                files.append(file_path)
        
        logger.info(f"Found {len(files)} processable files")
        return files
    
    def process_repository(self) -> Dict[str, Any]:
        """Process entire repository with enhanced chunking."""
        if not self.repo_path.exists():
            logger.error(f"Repository path does not exist: {self.repo_path}")
            raise FileNotFoundError(f"Repository path does not exist: {self.repo_path}")
        
        logger.info(f"Starting repository processing: {self.repo_path}")
        
        # Reset state
        self.all_chunks = []
        
        # Collect and process files
        files_to_process = self.collect_files()
        
        processed_count = 0
        error_count = 0
        
        for file_path in files_to_process:
            try:
                self.process_file(file_path)
                processed_count += 1
            except Exception as e:
                logger.error(f"Error processing {file_path}: {e}")
                error_count += 1
        
        logger.info(f"Repository processing complete: {processed_count} files processed, {error_count} errors")
        logger.info(f"Total chunks created: {len(self.all_chunks)}")
        
        # Return simple data structure
        return {
            'repository_info': {
                'path': str(self.repo_path),
                'processed_at': datetime.now().isoformat(),
                'total_chunks': len(self.all_chunks),
                'files_processed': processed_count,
                'processing_errors': error_count,
            },
            'chunks': self.all_chunks
        }
    
    def save_results(self, data: Dict[str, Any]) -> None:
        """Save processed data to JSON file."""
        logger.info(f"Saving results to: {self.output_path}")
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        # Save main repository data
        output_file = self.output_path / 'repository_chunks.json'
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            logger.info(f"Results successfully saved to: {output_file}")
            logger.debug(f"Saved {data['repository_info']['total_chunks']} chunks to JSON file")
        except Exception as e:
            logger.error(f"Error saving results to {output_file}: {e}")
            raise
    
    def generate_repo_analysis(self) -> str:
        """Generate a text report analyzing the repository structure and patterns.
        
        Returns:
            str: The generated repository analysis report as a string
        """
        logger.info("Generating repository analysis report")
        
        # Collect statistics
        total_files = 0
        total_lines = 0
        
        # Process each file for basic stats
        for file_path in self.repo_path.rglob('*'):
            if file_path.is_file():  # Include all files, not just code files
                total_files += 1
                try:
                    content = self.read_file_content(file_path)
                    lines = content.split('\n')
                    total_lines += len(lines)
                except Exception as e:
                    logger.error(f"Error analyzing file {file_path}: {e}")
        
        # Generate report as string
        report_lines = []
        report_lines.append("Repository Analysis Report")
        report_lines.append("=========================\n")
        
        report_lines.append("1. Repository Overview")
        report_lines.append("---------------------")
        report_lines.append(f"Total Files: {total_files}")
        report_lines.append(f"Total Lines: {total_lines}\n")
        
        report_lines.append("2. Repository Structure")
        report_lines.append("----------------------")
        
        def write_tree(directory: Path, lines: list, prefix: str = "", is_last: bool = True):
            # Add the current directory name
            dir_marker = "└──" if is_last else "├──"
            relative_path = directory.relative_to(self.repo_path)
            lines.append(f"{prefix}{dir_marker} {relative_path}/")
            
            # Prepare prefix for children
            new_prefix = prefix + ("    " if is_last else "│   ")
            
            # Get all items in the directory
            items = list(directory.iterdir())
            # Sort directories first, then files
            items.sort(key=lambda x: (not x.is_dir(), x.name))
            
            # Process directories
            dirs = [d for d in items if d.is_dir() and not d.name.startswith('.')]
            files = [f for f in items if f.is_file() or (f.is_dir() and f.name.startswith('.'))]
            
            # Process directories
            for idx, dir_path in enumerate(dirs):
                is_last_dir = (idx == len(dirs) - 1) and not files
                write_tree(dir_path, lines, new_prefix, is_last_dir)
            
            # Process files
            for idx, file_path in enumerate(files):
                file_marker = "└──" if idx == len(files) - 1 else "├──"
                lines.append(f"{new_prefix}{file_marker} {file_path.name}")
        
        # Start the tree from the root
        report_lines.append("Complete repository structure:")
        write_tree(self.repo_path, report_lines)
        report_lines.append("")
        
        # Join all lines with newlines
        report = "\n".join(report_lines)
        
        logger.info("Repository analysis report generated")
        return report
