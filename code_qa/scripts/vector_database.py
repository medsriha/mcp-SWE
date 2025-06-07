#!/usr/bin/env python3
"""
Vector Database Storage for Python Code Chunks

Stores code chunks from PythonChunker into a vector database for semantic search and retrieval.
"""

import os
from typing import List, Dict, Any, Optional
from dataclasses import asdict
import chromadb
from chromadb.config import Settings as ChromaSettings
from sentence_transformers import SentenceTransformer
from code_qa.scripts.python_chunker import CodeChunk

import logging
import uuid

# Set up logging
logger = logging.getLogger(__name__)

class CodeVectorStore:
    """Vector database for storing and retrieving code chunks."""
    
    # Default metadata fields for embedding
    DEFAULT_EMBED_FIELDS = ['chunk_type', 'function_name', 'docstring']
    
    # All available metadata fields
    AVAILABLE_METADATA_FIELDS = [
        'chunk_type', 'function_name', 'class_name', 'module_name', 
        'docstring'
    ]
    
    def __init__(self, 
                 db_path: str = None,
                 collection_name: str = "python_code_chunks",
                 embedding_model: str = None,
                 embed_metadata: List[str] = None):
        """
        Initialize the vector store.
        
        Args:
            db_path: Path to store the ChromaDB database.
            collection_name: Name of the collection to store chunks
            embedding_model: Sentence transformer model for embeddings.
            embed_metadata: List of metadata fields to include in embeddings
        """
        self.db_path = db_path
        self.collection_name = collection_name
        self.embed_metadata = embed_metadata or self.DEFAULT_EMBED_FIELDS
        self.embedding_model_name = embedding_model
        
        self._initialize_database()
        self._initialize_embedding_model(self.embedding_model_name)
        self._get_or_create_collection()
    
    
    def _initialize_database(self) -> None:
        """Initialize ChromaDB client."""
        self.client = chromadb.PersistentClient(
            path=self.db_path,
            settings=ChromaSettings(anonymized_telemetry=False, allow_reset=True)
        )
    
    def _initialize_embedding_model(self, model_name: str) -> None:
        """Initialize the sentence transformer model."""
        logger.info(f"Loading embedding model: {model_name}")
        self.embedding_model = SentenceTransformer(model_name)
        logger.debug(f"Successfully loaded embedding model: {model_name}")
    
    def _get_or_create_collection(self) -> None:
        """Get existing collection or create new one."""
        try:
            self.collection = self.client.get_collection(name=self.collection_name)
            logger.info(f"Loaded existing collection: {self.collection_name}")
        except:
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"description": "Python code chunks with rich metadata"}
            )
            logger.info(f"Created new collection: {self.collection_name}")
    
    def add_chunks(self, chunks: List[CodeChunk], embed_metadata: List[str] = None) -> None:
        """Add code chunks to the vector database."""
        if not chunks:
            logger.warning("No chunks to add")
            return
        
        metadata_fields = embed_metadata or self.embed_metadata
        logger.info(f"Adding {len(chunks)} chunks to vector database...")
        logger.debug(f"Embedding metadata fields: {metadata_fields}")
        
        # Prepare data for batch insertion (without IDs)
        documents, metadatas = self._prepare_batch_data(chunks, metadata_fields)
        
        # Generate embeddings and add to database
        logger.debug("Generating embeddings...")
        embeddings = self.embedding_model.encode(documents, show_progress_bar=True)
        
        # Let ChromaDB auto-generate IDs by not providing the ids parameter
        self.collection.add(
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings.tolist(),
            ids=[str(uuid.uuid4()) for _ in range(len(documents))]
        )
        
        logger.info(f"Successfully added {len(chunks)} chunks to the database")
    
    def _prepare_batch_data(self, chunks: List[CodeChunk], metadata_fields: List[str]) -> tuple:
        """Prepare documents and metadata for batch insertion."""
        documents, metadatas = [], []
        
        for chunk in chunks:
            enhanced_text = self._create_enhanced_text(chunk, metadata_fields)
            documents.append(enhanced_text)
            
            metadata = self._prepare_metadata(chunk)
            metadatas.append(metadata)
        
        return documents, metadatas
    
    def _create_enhanced_text(self, chunk: CodeChunk, metadata_fields: List[str]) -> str:
        """Create enhanced text with metadata for better embeddings."""
        metadata_parts = []
        
        # Add unique location identifier
        location_id = f"Location: {chunk.file_path}:{chunk.start_line}-{chunk.end_line}"
        metadata_parts.append(location_id)
        
        # Add all available metadata fields, not just the requested ones
        for field in self.AVAILABLE_METADATA_FIELDS:
            value = getattr(chunk, field, None)
            if not value:
                continue
                
            formatted_value = self._format_metadata_field(field, value)
            if formatted_value:
                metadata_parts.append(formatted_value)
        
        # Add file context
        if chunk.file_path:
            filename = os.path.basename(chunk.file_path)
            metadata_parts.append(f"File: {filename}")
        
        # Combine metadata with code content
        metadata_context = " | ".join(metadata_parts)
        return f"{metadata_context}\n\nCode Content:\n{chunk.content}"
    
    def _format_metadata_field(self, field: str, value: Any) -> str:
        """Format a metadata field for inclusion in enhanced text."""
        if field == 'chunk_type':
            return f"Type: {value}"
        elif field == 'function_name':
            return f"Name: {value}"
        elif field == 'class_name':
            return f"Class: {value}"
        elif field == 'module_name':
            filename = os.path.basename(value) if '/' in str(value) else value
            return f"Module: {filename}"
        elif field == 'docstring' and value.strip():
            return f"Documentation: {value.strip()}"
        
        return ""
    
    def search(self, query: str, n_results: int = 5) -> List[Dict[str, Any]]:
        """Search for similar code chunks."""
        logger.info(f"Searching for: '{query}' (n_results: {n_results})")
        
        query_embedding = self.embedding_model.encode([query])
        results = self.collection.query(
            query_embeddings=query_embedding.tolist(),
            n_results=n_results
        )
        
        formatted_results = self._format_search_results(results)
        logger.debug(f"Search returned {len(formatted_results)} results")
        return formatted_results
    
    def _format_search_results(self, results: Dict) -> List[Dict[str, Any]]:
        """Format search results into a consistent structure."""
        formatted_results = []
        
        for i in range(len(results['ids'][0])):
            enhanced_doc = results['documents'][0][i]
            original_content = self._extract_original_content(enhanced_doc)
            result = {
                'id': results['ids'][0][i],
                'content': original_content,
                'metadata': results['metadatas'][0][i],
                'distance': results['distances'][0][i] if 'distances' in results else None,
            }
            formatted_results.append(result)
        
        return formatted_results
    
    def list_files(self) -> List[str]:
        """List all files in the database."""
        all_data = self.collection.get()
        file_paths = {
            metadata['file_path'] 
            for metadata in all_data['metadatas'] 
            if 'file_path' in metadata
        }
        return sorted(file_paths)
    
    def reset_database(self) -> None:
        """Reset the entire database (delete all data)."""
        logger.warning("Resetting database - all data will be deleted")
        self.client.delete_collection(name=self.collection_name)
        self.collection = self.client.create_collection(
            name=self.collection_name,
            metadata={"description": "Python code chunks with rich metadata"}
        )
        logger.info("Database reset successfully")
    
    def _prepare_metadata(self, chunk: CodeChunk) -> Dict[str, Any]:
        """Prepare metadata for storage (ensure all values are serializable)."""
        metadata = asdict(chunk)
        metadata.pop('content', None)  # Content stored as document
        
        # Ensure all values are basic types
        for key, value in metadata.items():
            if value is None:
                metadata[key] = ""
            elif isinstance(value, bool):
                metadata[key] = str(value)
            elif not isinstance(value, (str, int, float)):
                metadata[key] = str(value)
        
        return metadata

    def _extract_original_content(self, enhanced_text: str) -> str:
        """Extract original code content from enhanced text."""
        if "\n\nCode Content:\n" in enhanced_text:
            return enhanced_text.split("\n\nCode Content:\n", 1)[1]
        return enhanced_text

    def search_and_reconstruct(self, 
                             query: str, 
                             file_path: Optional[str] = None,
                             n_results: int = 5,
                             max_token_limit: int = 1000) -> Dict[str, str]:
        """
        Search for relevant chunks and reconstruct files with hybrid token-aware expansion.
        
        Args:
            query: Search query
            file_path: Optional specific file path to reconstruct
            n_results: Number of search results to return (default: 10)
            max_token_limit: Maximum tokens per file for reconstruction (default: 1000)
        """
        
        logger.info(f"Searching and reconstructing for: '{query}' (max_token_limit: {max_token_limit})")
        
        search_results = self.search(query, n_results=n_results)
        if not search_results:
            logger.warning("No relevant chunks found")
            return {}
        
        logger.debug(f"Found {len(search_results)} relevant chunks")
        
        # Group search results by file
        results_by_file = {}
        for result in search_results:
            file_path_from_chunk = result['metadata'].get('file_path', '')
            if file_path_from_chunk:
                if file_path_from_chunk not in results_by_file:
                    results_by_file[file_path_from_chunk] = []
                results_by_file[file_path_from_chunk].append(result)
        
        # Filter by specific file if requested
        if file_path:
            results_by_file = {file_path: results_by_file.get(file_path, [])}
        
        file_basenames = [os.path.basename(f) for f in results_by_file.keys()]
        logger.info(f"Files to reconstruct: {file_basenames}")

        # Reconstruct each file
        reconstructed_files = {}
        for target_file, file_search_results in results_by_file.items():
            try:
                most_relevant_chunk = file_search_results[0] if file_search_results else None
                reconstructed_files[target_file] = self._reconstruct_file_hybrid(
                    target_file, most_relevant_chunk, max_token_limit
                )
            except Exception as e:
                logger.error(f"Error reconstructing {target_file}: {e}")
                reconstructed_files[target_file] = f"# Error reconstructing file: {e}"
        
        return reconstructed_files
    
    def _reconstruct_file_hybrid(self, 
                                file_path: str, 
                                most_relevant_chunk: Optional[Dict[str, Any]],
                                max_token_limit: int) -> str:
        """Hybrid file reconstruction starting from most relevant chunk."""
        logger.debug(f"Collecting all chunks for file: {file_path}")
        
        all_chunks = self.collection.get(where={"file_path": {"$eq": file_path}})
        if not all_chunks['ids']:
            logger.warning(f"No chunks found for file: {file_path}")
            return f"# No chunks found for file: {file_path}"
        
        # Convert to structured format
        file_chunks = []
        for i, chunk_id in enumerate(all_chunks['ids']):
            metadata = all_chunks['metadatas'][i]
            content = self._extract_original_content(all_chunks['documents'][i])

            file_chunks.append({
                'content': content,
                'start_line': int(metadata.get('start_line', 0)),
                'end_line': int(metadata.get('end_line', 0)),
                'token_count': int(metadata.get('token_count', 0)),
                'id': chunk_id,
                'chunk_type': metadata.get('chunk_type', 'unknown')
            })
        
        logger.debug(f"Found {len(file_chunks)} chunks for {file_path}")
        
        # Sort chunks by start_line
        sorted_chunks = sorted(file_chunks, key=lambda x: x['start_line'])
        
        return self._expand_from_relevant_chunk(sorted_chunks, most_relevant_chunk, max_token_limit)
    
    def _expand_from_relevant_chunk(self, 
                                  sorted_chunks: List[Dict[str, Any]], 
                                  most_relevant_chunk: Optional[Dict[str, Any]],
                                  max_token_limit: int) -> str:
        """Expand from the most relevant chunk backwards first, then forwards."""
        if not sorted_chunks:
            logger.warning("No chunks available for reconstruction")
            return "# No chunks available for reconstruction"
        
        # Find the starting chunk index
        start_idx = self._find_relevant_chunk_index(sorted_chunks, most_relevant_chunk)
        start_chunk = sorted_chunks[start_idx]
        
        logger.debug(f"Starting reconstruction from chunk at lines {start_chunk['start_line']}-{start_chunk['end_line']} (tokens: {start_chunk['token_count']})")
        
        # Initialize with the most relevant chunk
        selected_chunks = [start_chunk]
        total_tokens = start_chunk['token_count']
        
        # Phase 1: Add chunks before the relevant chunk (backwards)
        logger.debug("Phase 1: Adding chunks BEFORE the relevant chunk (going backwards)...")
        total_tokens = self._add_chunks_backwards(sorted_chunks, start_idx, selected_chunks, total_tokens, max_token_limit)
        
        # Phase 2: Add chunks after the relevant chunk (forwards)
        logger.debug("Phase 2: Adding chunks AFTER the relevant chunk (going forwards)...")
        self._add_chunks_forwards(sorted_chunks, start_idx, selected_chunks, total_tokens, max_token_limit)
        
        logger.debug(f"Final selection: {len(selected_chunks)} chunks")
        return self._combine_selected_chunks(selected_chunks)
    
    def _find_relevant_chunk_index(self, sorted_chunks: List[Dict], most_relevant_chunk: Optional[Dict]) -> int:
        """Find the index of the most relevant chunk in the sorted list."""
        if not most_relevant_chunk:
            return 0
        
        relevant_id = most_relevant_chunk['id']
        for i, chunk in enumerate(sorted_chunks):
            if chunk['id'] == relevant_id:
                return i
        return 0
    
    def _add_chunks_backwards(self, sorted_chunks: List[Dict], start_idx: int, 
                             selected_chunks: List[Dict], total_tokens: int, max_token_limit: int) -> int:
        """Add chunks before the start chunk, going backwards."""
        for i in range(start_idx - 1, -1, -1):
            chunk = sorted_chunks[i]
            
            if total_tokens + chunk['token_count'] > max_token_limit:
                logger.debug(f"  - Chunk at lines {chunk['start_line']}-{chunk['end_line']} would exceed token limit")
                break
            
            selected_chunks.insert(0, chunk)
            total_tokens += chunk['token_count']
            logger.debug(f"  - Added chunk at lines {chunk['start_line']}-{chunk['end_line']} (tokens: +{chunk['token_count']}, total: {total_tokens})")
        
        return total_tokens
    
    def _add_chunks_forwards(self, sorted_chunks: List[Dict], start_idx: int, 
                            selected_chunks: List[Dict], total_tokens: int, max_token_limit: int) -> None:
        """Add chunks after the start chunk, going forwards."""
        for i in range(start_idx + 1, len(sorted_chunks)):
            chunk = sorted_chunks[i]
            
            if total_tokens + chunk['token_count'] > max_token_limit:
                logger.debug(f"  - Chunk at lines {chunk['start_line']}-{chunk['end_line']} would exceed token limit")
                break
            
            selected_chunks.append(chunk)
            total_tokens += chunk['token_count']
            logger.debug(f"  - Added chunk at lines {chunk['start_line']}-{chunk['end_line']} (tokens: +{chunk['token_count']}, total: {total_tokens})")
    
    def _combine_selected_chunks(self, chunks: List[Dict[str, Any]]) -> str:
        """Combine selected chunks into final reconstructed content."""
        if not chunks:
            return ""
        
        result_lines = []
        for chunk in chunks:
            content = chunk['content'].rstrip()
            if content:
                result_lines.extend(content.split('\n'))
        
        reconstructed = '\n'.join(result_lines)
        logger.info(f"Reconstruction complete: {len(result_lines)} lines, {len(reconstructed)} characters")
        
        return reconstructed