## ingest_anything package

### Submodules

#### ingest_anything.add\_types module

This module defines the data models for configuring text chunking and ingestion inputs.

*   **Classes**

    *   `Chunking(BaseModel)`: A Pydantic model for configuring text chunking parameters.

        *   Inherits from: `pydantic.BaseModel`
        *   Description: This class defines the configuration for different text chunking strategies and their associated parameters.
        *   Attributes:
            *   `chunker` (`Literal["token", "sentence", "semantic", "sdpm", "late"]`): The chunking strategy to use.
                *   `"token"`: Split by number of tokens.
                *   `"sentence"`: Split by sentences.
                *   `"semantic"`: Split by semantic meaning.
                *   `"sdpm"`: Split using sentence distance probability matrix.
                *   `"late"`: Delayed chunking strategy.
            *   `chunk_size` (`Optional[int]`): The target size for each chunk. Defaults to 512 if not specified.
            *   `chunk_overlap` (`Optional[int]`): The number of overlapping units between consecutive chunks. Defaults to 128 if not specified.
            *   `similarity_threshold` (`Optional[float]`): The minimum similarity threshold for semantic chunking. Defaults to 0.7 if not specified.
            *   `min_characters_per_chunk` (`Optional[int]`): The minimum number of characters required for a valid chunk. Defaults to 24 if not specified.
            *   `min_sentences` (`Optional[int]`): The minimum number of sentences required for a valid chunk. Defaults to 1 if not specified.
        *   Example:

            ```python
            >>> from ingest_anything.add_types import Chunking
            >>> chunking_config = Chunking(chunker="semantic", chunk_size=256, chunk_overlap=64, similarity_threshold=0.8, min_characters_per_chunk=50, min_sentences=2)
            ```

    *   `IngestionInput(BaseModel)`: A class that validates and processes ingestion inputs for document processing.

        *   Inherits from: `pydantic.BaseModel`
        *   Description: This class handles different types of document inputs and chunking strategies, converting files and setting up appropriate chunking mechanisms based on the specified configuration.
        *   Attributes:
            *   `files_or_dir` (`Union[str, List[str]]`): Path to directory containing files or list of file paths to process.
            *   `chunking` (`Chunking`): Configuration for the chunking strategy to be used.
            *   `tokenizer` (`Optional[str]`, default=`None`): Name or path of the tokenizer model to be used (required for 'token' and 'sentence' chunking).
            *   `embedding_model` (`str`): Name or path of the embedding model to be used.
        *   Example:

            ```python
            >>> from ingest_anything.add_types import IngestionInput, Chunking
            >>> ingestion_config = IngestionInput(
            ...     files_or_dir="path/to/documents",
            ...     chunking=Chunking(chunker="token", chunk_size=256, chunk_overlap=64),
            ...     tokenizer="bert-base-uncased",
            ...     embedding_model="sentence-transformers/all-mpnet-base-v2"
            ... )
            ```

#### ingest\_anything.ingestion module

This module defines the `IngestAnything` class, which handles the ingestion and storage of documents in a Qdrant vector database.

*   **Classes**

    *   `IngestAnything`: A class for ingesting and storing documents in a Qdrant vector database with various chunking strategies.

        *   `__init__(qdrant_client: Optional[QdrantClient] = None, async_qdrant_client: Optional[AsyncQdrantClient] = None, collection_name: str = "IngestAnythingCollection", hybrid_search: bool = False, fastembed_model: str = "Qdrant/bm25")`
            *   Parameters:
                *   `qdrant_client` (`Optional[QdrantClient]`, default=`None`): Synchronous Qdrant client instance. At least one of `qdrant_client` or `async_qdrant_client` must be provided.
                *   `async_qdrant_client` (`Optional[AsyncQdrantClient]`, default=`None`): Asynchronous Qdrant client instance.
                *   `collection_name` (`str`, default="IngestAnythingCollection"): Name of the collection in Qdrant where documents will be stored.
                *   `hybrid_search` (`bool`, default=`False`): Whether to enable hybrid search capabilities.
                *   `fastembed_model` (`str`, default="Qdrant/bm25"): Model to use for sparse embeddings in hybrid search.
            *   Example:

                ```python
                >>> from qdrant_client import QdrantClient
                >>> from ingest_anything.ingestion import IngestAnything
                >>> client = QdrantClient(":memory:")
                >>> ingestor = IngestAnything(qdrant_client=client, collection_name="my_collection")
                ```

        *   `ingest(files_or_dir: str | List[str], embedding_model: str, chunker: Literal["token", "sentence", "semantic", "sdpm", "late"], tokenizer: Optional[str] = None, chunk_size: Optional[int] = None, chunk_overlap: Optional[int] = None, similarity_threshold: Optional[float] = None, min_characters_per_chunk: Optional[int] = None, min_sentences: Optional[int] = None) -> VectorStoreIndex`
            *   Parameters:
                *   `files_or_dir` (`str | List[str]`): Path to file(s) or directory to ingest.
                *   `embedding_model` (`str`): Name of the HuggingFace embedding model to use.
                *   `chunker` (`Literal["token", "sentence", "semantic", "sdpm", "late"]`): Chunking strategy to use.
                *   `tokenizer` (`Optional[str]`, default=`None`): Tokenizer to use for chunking. Required for "token" and "sentence" chunking.
                *   `chunk_size` (`Optional[int]`, default=`None`): Size of chunks.
                *   `chunk_overlap` (`Optional[int]`, default=`None`): Number of overlapping tokens/sentences between chunks.
                *   `similarity_threshold` (`Optional[float]`, default=`None`): Similarity threshold for semantic chunking.
                *   `min_characters_per_chunk` (`Optional[int]`, default=`None`): Minimum number of characters per chunk.
                *   `min_sentences` (`Optional[int]`, default=`None`): Minimum number of sentences per chunk.
            *   Returns:
                *   `VectorStoreIndex`: Index containing the ingested and processed documents.
            *   Example:

                ```python
                >>> from qdrant_client import QdrantClient
                >>> from ingest_anything.ingestion import IngestAnything
                >>> client = QdrantClient(":memory:")
                >>> ingestor = IngestAnything(qdrant_client=client, collection_name="my_collection")
                >>> index = ingestor.ingest(
                ...     files_or_dir="path/to/documents",
                ...     embedding_model="sentence-transformers/all-mpnet-base-v2",
                ...     chunker="semantic",
                ...     similarity_threshold=0.8
                ... )
                ```