try:
    from add_types import IngestionInput, Chunking, CodeChunking, CodeFiles
except ModuleNotFoundError:
    from .add_types import IngestionInput, Chunking, CodeChunking
from qdrant_client import QdrantClient, AsyncQdrantClient
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext
from llama_index.readers.docling import DoclingReader
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.schema import TextNode
from typing import Optional, Literal, List
import uuid

reader = DoclingReader()

class IngestAnything:
    """
    A class for ingesting and storing documents in a Qdrant vector database with various chunking strategies.

    This class provides functionality to ingest documents from files or directories, chunk them using
    different strategies, and store them in a Qdrant vector database for later retrieval and search.

    Attributes:

        qdrant_client : QdrantClient, optional
            Synchronous Qdrant client instance
        
        async_qdrant_client : AsyncQdrantClient, optional
            Asynchronous Qdrant client instance
        
        collection_name : str, default = "ingest-anything-" + random UUID
            Name of the collection in Qdrant where documents will be stored
        
        hybrid_search : bool, default=False
            Whether to enable hybrid search capabilities
        
        fastembed_model : str, default="Qdrant/bm25"
            Model to use for sparse embeddings in hybrid search

    At least one of qdrant_client or async_qdrant_client must be provided when initializing the class.
    """
    def __init__(self, qdrant_client: Optional[QdrantClient] = None, async_qdrant_client: Optional[AsyncQdrantClient] = None, collection_name: str = "ingest-anything-"+str(uuid.uuid4()), hybrid_search: bool = False, fastembed_model: str = "Qdrant/bm25"):
        if qdrant_client is None and async_qdrant_client is None:
            raise ValueError("Either sync or async client (or both) must be provided")
        self.vector_store = QdrantVectorStore(collection_name=collection_name, client = qdrant_client, aclient= async_qdrant_client, enable_hybrid=hybrid_search, fastembed_sparse_model=fastembed_model)
    def ingest(
            self,
            files_or_dir: str | List[str],
            embedding_model: str,
            chunker: Literal["token", "sentence", "semantic", "sdpm", "late"],
            tokenizer: Optional[str] = None,
            chunk_size: Optional[int] = None,
            chunk_overlap: Optional[int] = None,
            similarity_threshold: Optional[float] = None,
            min_characters_per_chunk: Optional[int] = None,
            min_sentences: Optional[int] = None,
    ):
        """
        Ingest documents from files or directories with specified chunking strategy.

        Parameters
        ----------
        files_or_dir : str or List[str]
            Path to file(s) or directory to ingest
        embedding_model : str
            Name of the HuggingFace embedding model to use
        chunker : {"token", "sentence", "semantic", "sdpm", "late"}
            Chunking strategy to use
        tokenizer : str, optional
            Tokenizer to use for chunking
        chunk_size : int, optional
            Size of chunks
        chunk_overlap : int, optional
            Number of overlapping tokens/sentences between chunks
        similarity_threshold : float, optional
            Similarity threshold for semantic chunking
        min_characters_per_chunk : int, optional
            Minimum number of characters per chunk
        min_sentences : int, optional
            Minimum number of sentences per chunk

        Returns
        -------
        VectorStoreIndex
            Index containing the ingested and processed documents
        """
        chunking = Chunking(chunker=chunker, chunk_size=chunk_size, chunk_overlap=chunk_overlap, similarity_threshold=similarity_threshold, min_characters_per_chunk=min_characters_per_chunk, min_sentences=min_sentences)
        ingestion_input = IngestionInput(files_or_dir=files_or_dir, chunking=chunking, tokenizer=tokenizer, embedding_model=embedding_model)
        docs = SimpleDirectoryReader(input_files=ingestion_input.files_or_dir, file_extractor={".pdf": reader}).load_data()
        text = "\n\n---\n\n".join([d.text for d in docs])
        chunks = ingestion_input.chunking.chunk(text)
        nodes = [TextNode(text=c.text, id_=str(uuid.uuid4())) for c in chunks]
        storage_context = StorageContext.from_defaults(vector_store=self.vector_store)
        index = VectorStoreIndex(nodes=nodes, embed_model=HuggingFaceEmbedding(model_name=embedding_model), show_progress=True, storage_context=storage_context)
        return index
    
class IngestCode:
    """
    A class for ingesting and indexing code files using Qdrant vector store.

    This class handles the ingestion of code files, chunking them into smaller pieces,
    and storing them in a Qdrant vector store for later retrieval and search.

    Parameters
    ----------
    qdrant_client : QdrantClient, optional
        Synchronous Qdrant client instance
    async_qdrant_client : AsyncQdrantClient, optional
        Asynchronous Qdrant client instance
    collection_name : str, optional
        Name of the collection in Qdrant. Defaults to "ingest-anything-" + random UUID
    hybrid_search : bool, optional
        Whether to enable hybrid search. Defaults to False
    fastembed_model : str, optional
        Model to use for sparse embeddings in hybrid search. Defaults to "Qdrant/bm25"

    Methods
    -------
    ingest(files, embedding_model, language, return_type=None, tokenizer=None, 
           chunk_size=None, include_nodes=None)
        Ingest code files and create a searchable vector index.

    Raises
    ------
    ValueError
        If neither sync nor async client is provided
    """
    def __init__(self, qdrant_client: Optional[QdrantClient] = None, async_qdrant_client: Optional[AsyncQdrantClient] = None, collection_name: str = "ingest-anything-"+str(uuid.uuid4()), hybrid_search: bool = False, fastembed_model: str = "Qdrant/bm25"):
        if qdrant_client is None and async_qdrant_client is None:
            raise ValueError("Either sync or async client (or both) must be provided")
        self.vector_store = QdrantVectorStore(collection_name=collection_name, client = qdrant_client, aclient= async_qdrant_client, enable_hybrid=hybrid_search, fastembed_sparse_model=fastembed_model)
    def ingest(
            self,
            files: List[str],
            embedding_model: str,
            language: str,
            return_type: Optional[Literal["chunks", "texts"]] = None,
            tokenizer: Optional[str] = None,
            chunk_size: Optional[int] = None,
            include_nodes: Optional[bool] = None,
    ):
        """
        Ingest code files and create a searchable vector index.

        Parameters
        ----------
            files (List[str]): List of file paths to ingest
            embedding_model (str): Name of the HuggingFace embedding model to use
            language (str): Programming language of the code files
            return_type (Literal["chunks", "texts"], optional): Type of return value from chunking
            tokenizer (str, optional): Name of tokenizer to use
            chunk_size (int, optional): Size of chunks for text splitting
            include_nodes (bool, optional): Whether to include AST nodes in chunking
        
        Returns
        --------
            VectorStoreIndex: Index containing the ingested and embedded code chunks
        """
        fls = CodeFiles(files=files)
        chunking = CodeChunking(language=language, return_type=return_type, tokenizer=tokenizer, chunk_size=chunk_size, include_nodes=include_nodes)
        docs = SimpleDirectoryReader(input_files=fls.files).load_data()
        text = "\n\n---\n\n".join([d.text for d in docs])
        chunks = chunking.chunker.chunk(text)
        nodes = [TextNode(text=c.text, id_=str(uuid.uuid4())) for c in chunks]
        storage_context = StorageContext.from_defaults(vector_store=self.vector_store)
        index = VectorStoreIndex(nodes=nodes, embed_model=HuggingFaceEmbedding(model_name=embedding_model), show_progress=True, storage_context=storage_context)
        return index
        
        