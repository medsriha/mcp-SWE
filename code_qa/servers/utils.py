from code_qa.scripts.repo_downloader import RepoDownloader
from code_qa.scripts.repo_processor import RepoProcessor
from code_qa.scripts.vector_database import CodeVectorStore
from code_qa.scripts.python_chunker import CodeChunk

from code_qa.config.settings import get_settings

settings = get_settings()

BASE_DIR = settings.base_dir
DOWNLOADS_DIR = settings.downloads_dir
PROCESSED_DIR = settings.processed_dir
VECTOR_DB_DIR = settings.vector_db_dir


def get_repo_id(repository_url: str) -> str:
    """Generate a consistent repository ID from URL.
    Uses SHA-256 hash to ensure consistency across different runs."""
    import hashlib
    return hashlib.sha256(repository_url.encode()).hexdigest()[:16]


def generate_repo_analysis(repository_url: str) -> str:
    """Generate analysis report for a repository without code chunking.
    
    Args:
        repository_url: URL or path to the repository
        
    Returns:
        Path to the generated analysis report
    """
    repo_id = get_repo_id(repository_url)
    
    # Create directories for this repository
    download_dir = DOWNLOADS_DIR / repo_id
    processed_dir = PROCESSED_DIR / repo_id
    
    # Create directories if they don't exist
    download_dir.mkdir(exist_ok=True)
    processed_dir.mkdir(exist_ok=True)
    
    try:
        # Download the repository if not already downloaded
        downloader = RepoDownloader()
        success = downloader.download(repository_url, str(download_dir))
        
        if not success:
            raise Exception(f"Failed to download repository: {repository_url}")
        
        # Create processor just for analysis
        processor = RepoProcessor(
            repo_path=str(download_dir),
            output_path=str(processed_dir),
            max_chunk_size=settings.chunk_size  # Not used for analysis but required by constructor
        )
        
        # Generate only the analysis report
        return processor.generate_repo_analysis()
        
    except Exception as e:
        raise


def process_repository(repository_url: str) -> CodeVectorStore:
    """Download and process a repository for code search."""
    repo_id = get_repo_id(repository_url)
    
    # Create directories for this repository
    download_dir = DOWNLOADS_DIR / repo_id
    processed_dir = PROCESSED_DIR / repo_id
    vector_db_dir = VECTOR_DB_DIR / repo_id
    
    # Create directories if they don't exist
    download_dir.mkdir(exist_ok=True)
    processed_dir.mkdir(exist_ok=True)
    vector_db_dir.mkdir(exist_ok=True)
    
    try:
        # Download the repository
        downloader = RepoDownloader()
        success = downloader.download(repository_url, str(download_dir))
        
        if not success:
            raise Exception(f"Failed to download repository: {repository_url}")
        
        # Process the repository with enhanced processor
        processor = RepoProcessor(
            repo_path=str(download_dir),
            output_path=str(processed_dir),
            max_chunk_size=settings.chunk_size
        )
        
        # Process and get results
        results = processor.process_repository()
        chunks_data = results['chunks']
        
        # Convert to CodeChunk objects for vector database
        code_chunks = []
        for chunk_data in chunks_data:
            # Create a CodeChunk object from the processed data
            code_chunk = CodeChunk(
                content=chunk_data.get('content', ''),
                start_line=chunk_data.get('start_line', 1),
                end_line=chunk_data.get('end_line', 1),
                file_path=chunk_data.get('file_path', ''),
                chunk_index=0,  # Will be set later
                total_chunks=len(chunks_data),
                chunk_type=chunk_data.get('chunk_type', 'unknown'),
                function_name=chunk_data.get('function_name'),
                class_name=chunk_data.get('class_name'),
                module_name=chunk_data.get('file_path', '').split('/')[-1] if chunk_data.get('file_path') else '',
                token_count=chunk_data.get('token_count', 0),
                docstring=chunk_data.get('docstring'),
            )
            code_chunks.append(code_chunk)
        
        # Create vector database and store chunks
        vector_store = CodeVectorStore(
            db_path=str(vector_db_dir),
            collection_name=f"repo_{repo_id}",
            embedding_model=settings.embedding_model
        )
        
        # Add chunks to vector database
        vector_store.add_chunks(code_chunks)
        
        return vector_store
        
    except Exception as e:
        raise