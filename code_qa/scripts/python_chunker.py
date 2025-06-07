#!/usr/bin/env python3
"""
Function-Focused Python Code Chunker for RAG

Chunks code by functions with rich metadata, and handles non-functional code by token size.
"""

import tree_sitter_python as tspython
from tree_sitter import Language, Parser
from typing import List, Optional
from dataclasses import dataclass
import tiktoken
import os
import logging

# Set up logging
logger = logging.getLogger(__name__)

@dataclass
class CodeChunk:
    """Code chunk with rich metadata for reconstruction."""
    # Core attributes
    content: str
    start_line: int
    end_line: int
    file_path: str
    chunk_index: int
    total_chunks: int
    token_count: int
    # Essential metadata
    chunk_type: str  # 'function', 'method', 'class_header', 'module_code'
    function_name: Optional[str] = None
    class_name: Optional[str] = None
    module_name: Optional[str] = None
    docstring: Optional[str] = None


class PythonChunker:
    """Function-focused Python code chunker."""
    
    def __init__(self):
        """Initialize the chunker."""
        # Initialize tree-sitter
        self.language = Language(tspython.language())
        self.parser = Parser(self.language)
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
    
    def _extract_module_docstring(self, node) -> Optional[str]:
        """Extract module-level docstring if present."""
        for child in node.children:
            if child.type == 'expression_statement':
                expr = child.children[0] if child.children else None
                if expr and expr.type == 'string':
                    return self._get_node_text(expr).strip().strip('"\'')
        return None
    
    def _count_tokens(self, content: str) -> int:
        """Count tokens in the content."""
        try:
            return len(self.tokenizer.encode(content))
        except Exception as e:
            logger.warning(f"Error counting tokens, using fallback method: {e}")
            return len(content) // 4
    
    def chunk_code(self, source_code: str, file_path: str = "unknown_path") -> List[CodeChunk]:
        """
        Chunk Python source code from a string.
        
        Args:
            source_code: Python source code to chunk
            file_path: Virtual file path for metadata
            
        Returns:
            List of CodeChunk objects
        """
        logger.info(f"Chunking code from string, file_path: {file_path}")
        
        tree = self.parser.parse(bytes(source_code, "utf8"))
        
        self.source_code = source_code
        self.file_path = file_path
        self.module_name = self._extract_module_name(file_path)
        
        chunks = []
        processed_ranges = set()
        
        # Extract module docstring first
        module_docstring = self._extract_module_docstring(tree.root_node)
        if module_docstring:
            first_node = tree.root_node.children[0] if tree.root_node.children else None
            if first_node:
                chunk = CodeChunk(
                    content=self._get_node_text(first_node),
                    start_line=first_node.start_point[0] + 1,
                    end_line=first_node.end_point[0] + 1,
                    file_path=self.file_path,
                    token_count=self._count_tokens(self._get_node_text(first_node)),
                    chunk_index=0,
                    total_chunks=0,
                    chunk_type='module_docstring',
                    docstring=module_docstring
                )
                chunks.append(chunk)
                processed_ranges.add((chunk.start_line, chunk.end_line))
                logger.debug("Extracted module docstring")
        
        # Extract functions and methods
        logger.debug("Extracting functions and methods")
        self._extract_functions_and_methods(tree.root_node, chunks, None, processed_ranges)
        
        # Extract remaining code (imports, module-level code, etc.)
        logger.debug("Extracting non-function code")
        self._extract_non_function_code(chunks, processed_ranges)
        
        # Sort and finalize chunks
        chunks.sort(key=lambda c: c.start_line)
        for i, chunk in enumerate(chunks):
            chunk.chunk_index = i
            chunk.total_chunks = len(chunks)
            chunk.module_name = self.module_name
        
        logger.info(f"Successfully created {len(chunks)} chunks from {file_path}")
        logger.debug(f"Chunk types: {[chunk.chunk_type for chunk in chunks]}")
        
        return chunks
    
    def _extract_functions_and_methods(self, node, chunks: List[CodeChunk], 
                                     current_class: Optional[str], processed_ranges: set):
        """Extract all functions and methods recursively."""
        
        if node.type == 'class_definition':
            class_name = self._get_identifier_name(node)
            logger.debug(f"Processing class: {class_name}")
            
            # Create class header chunk
            class_header = self._extract_class_header(node)
            if class_header:
                chunk = self._create_class_header_chunk(class_header, node, class_name)
                chunks.append(chunk)
                processed_ranges.add((chunk.start_line, chunk.end_line))
                logger.debug(f"Created class header chunk for {class_name}")
            
            # Process methods in this class
            for child in node.children:
                self._extract_functions_and_methods(child, chunks, class_name, processed_ranges)
        
        elif node.type in ['function_definition', 'async_function_definition']:
            function_name = self._get_identifier_name(node)
            logger.debug(f"Processing {'async ' if node.type == 'async_function_definition' else ''}function: {function_name}")
            chunk = self._create_function_chunk(node, current_class)
            chunks.append(chunk)
            processed_ranges.add((chunk.start_line, chunk.end_line))
        
        elif node.type == 'decorated_definition':
            decorators = self._extract_decorators(node)
            target = node.children[-1] if node.children else None
            
            if target and target.type in ['function_definition', 'async_function_definition']:
                function_name = self._get_identifier_name(target)
                logger.debug(f"Processing decorated function: {function_name} with decorators: {decorators}")
                chunk = self._create_function_chunk(target, current_class, decorators)
                chunks.append(chunk)
                processed_ranges.add((node.start_point[0] + 1, node.end_point[0] + 1))
            elif target and target.type == 'class_definition':
                class_name = self._get_identifier_name(target)
                logger.debug(f"Processing decorated class: {class_name}")
                for child in target.children:
                    self._extract_functions_and_methods(child, chunks, class_name, processed_ranges)
        
        else:
            for child in node.children:
                self._extract_functions_and_methods(child, chunks, current_class, processed_ranges)
    
    def _extract_non_function_code(self, chunks: List[CodeChunk], processed_ranges: set):
        """Extract non-function code."""
        unprocessed_segments = self._find_unprocessed_segments(processed_ranges)
        logger.debug(f"Found {len(unprocessed_segments)} unprocessed segments")
        
        for start_line, end_line in unprocessed_segments:
            content = '\n'.join(self.source_code.splitlines()[start_line-1:end_line])
            
            if not content.strip():
                continue
            
            # Try to identify if this is a significant module-level statement
            # like setup() calls or similar
            is_significant = any(keyword in content for keyword in [
                'setup(',
                'main(',
                'if __name__ == "__main__":'
            ])
            
            chunk_type = 'module_statement' if is_significant else 'module_code'
            
            chunk = CodeChunk(
                content=content,
                start_line=start_line,
                end_line=end_line,
                file_path=self.file_path,
                token_count=self._count_tokens(content),
                chunk_index=0,
                total_chunks=0,
                chunk_type=chunk_type
            )
            chunks.append(chunk)
            logger.debug(f"Created {chunk_type} chunk")
    
    def _find_unprocessed_segments(self, processed_ranges: set) -> List[tuple]:
        """Find segments of code that haven't been processed yet."""
        total_lines = len(self.source_code.splitlines())
        segments = []
        sorted_ranges = sorted(processed_ranges)
        
        current_line = 1
        for start, end in sorted_ranges:
            if current_line < start:
                segments.append((current_line, start - 1))
            current_line = max(current_line, end + 1)
        
        if current_line <= total_lines:
            segments.append((current_line, total_lines))
        
        return segments
       
    
    def _create_function_chunk(self, node, class_name: Optional[str] = None, 
                             decorators: List[str] = None) -> CodeChunk:
        """Create a chunk for a function with rich metadata."""
        content = self._get_node_text(node)
        function_name = self._get_identifier_name(node)
        docstring = self._extract_docstring(node)
        chunk_type = 'method' if class_name else 'function'
        
        return CodeChunk(
            content=content,
            start_line=node.start_point[0] + 1,
            end_line=node.end_point[0] + 1,
            file_path=self.file_path,
            token_count=self._count_tokens(content),
            chunk_index=0,
            total_chunks=0,
            chunk_type=chunk_type,
            function_name=function_name,
            class_name=class_name,
            docstring=docstring,
            module_name=self.module_name
        )
    
    def _create_class_header_chunk(self, content: str, node, class_name: str) -> CodeChunk:
        """Create a chunk for class header."""
        lines = content.split('\n')
        
        return CodeChunk(
            content=content,
            start_line=node.start_point[0] + 1,
            end_line=node.start_point[0] + len(lines),
            file_path=self.file_path,
            token_count=self._count_tokens(content),
            chunk_index=0,
            total_chunks=0,
            chunk_type='class_header',
            class_name=class_name,
            docstring=self._extract_docstring(node)
        )
    
    def _create_module_code_chunk(self, start_line: int, end_line: int, content: str) -> CodeChunk:
        """Create a chunk for module-level code."""
        return CodeChunk(
            content=content,
            start_line=start_line,
            end_line=end_line,
            file_path=self.file_path,
            token_count=self._count_tokens(content),
            chunk_index=0,
            total_chunks=0,
            chunk_type='module_code'
        )
    
    def _extract_class_header(self, node) -> Optional[str]:
        """Extract class header (definition + docstring + class variables, but no methods)."""
        
        lines = self._get_node_text(node).split('\n')
        header_lines = [lines[0]] if lines else []
        
        # Include content until first method
        for line in lines[1:]:
            stripped = line.strip()
            
            if stripped.startswith(('def ', 'async def ', 'class ')):
                break
            
            header_lines.append(line)
        
        return '\n'.join(header_lines) if len(header_lines) > 1 else None
    
    def _get_identifier_name(self, node) -> str:
        """Extract identifier name from a node."""
        for child in node.children:
            if child.type == 'identifier':
                return self._get_node_text(child)
        return "unknown"

    
    def _extract_docstring(self, node) -> Optional[str]:
        """Extract docstring from function or class."""
        for child in node.children:
            if child.type == 'block':
                for stmt in child.children:
                    if stmt.type == 'expression_statement':
                        expr = stmt.children[0] if stmt.children else None
                        if expr and expr.type == 'string':
                            return self._get_node_text(expr).strip().strip('"\'')
        return None
    
    def _extract_decorators(self, decorated_node) -> List[str]:
        """Extract decorator names from decorated definition."""
        decorators = []
        if decorated_node and decorated_node.type == 'decorated_definition':
            for child in decorated_node.children:
                if child.type == 'decorator':
                    decorators.append(self._get_node_text(child))
        return decorators
    
    def _get_node_text(self, node) -> str:
        """Extract text content from a tree-sitter node."""
        start_line = node.start_point[0]
        end_line = node.end_point[0]
        lines = self.source_code.splitlines()[start_line:end_line + 1]
        return '\n'.join(lines)
    
    def _extract_module_name(self, file_path: str) -> str:
        """Extract module name from file path."""
        module_name = os.path.splitext(os.path.basename(file_path))[0]
        logger.debug(f"Extracted module name: {module_name}")
        return module_name