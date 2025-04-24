from pydantic import BaseModel, model_validator
from typing import List, Literal, Optional
from typing_extensions import Self
from chonkie import SentenceTransformerEmbeddings, SemanticChunker, SDPMChunker, TokenChunker, SentenceChunker, LateChunker
from tokenizers import Tokenizer
from pdfitdown.pdfconversion import Converter

pdf_converter = Converter()

class Chunking(BaseModel):
    """A Pydantic model for configuring text chunking parameters.

    This class defines the configuration for different text chunking strategies and their associated parameters.

    Attributes:
        chunker (Literal["token", "sentence", "semantic", "sdpm", "late"]): 
            The chunking strategy to use. Options are:
            - "token": Split by number of tokens
            - "sentence": Split by sentences
            - "semantic": Split by semantic meaning
            - "sdpm": Split using sentence distance probability matrix
            - "late": Delayed chunking strategy
        
        chunk_size (Optional[int]): 
            The target size for each chunk. Defaults to 512 if not specified.
        
        chunk_overlap (Optional[int]): 
            The number of overlapping units between consecutive chunks. Defaults to 128 if not specified.
        
        similarity_threshold (Optional[float]): 
            The minimum similarity threshold for semantic chunking. Defaults to 0.7 if not specified.
        
        min_characters_per_chunk (Optional[int]): 
            The minimum number of characters required for a valid chunk. Defaults to 24 if not specified.
        
        min_sentences (Optional[int]): 
            The minimum number of sentences required for a valid chunk. Defaults to 1 if not specified.
    """
    chunker: Literal["token", "sentence", "semantic", "sdpm", "late"]
    chunk_size: Optional[int] = None
    chunk_overlap: Optional[int] = None
    similarity_threshold: Optional[float] = None
    min_characters_per_chunk: Optional[int] = None
    min_sentences: Optional[int] = None
    @model_validator(mode="after")
    def validate_chunking(self) -> Self:
        if self.chunk_size is None:
            self.chunk_size = 512
        if self.chunk_overlap is None:
            self.chunk_overlap = 128
        if self.similarity_threshold is None:
            self.similarity_threshold = 0.7
        if self.min_characters_per_chunk is None:
            self.min_characters_per_chunk = 24
        if self.min_sentences is None:
            self.min_sentences = 1
        return self

class IngestionInput(BaseModel):
    """
    A class that validates and processes ingestion inputs for document processing.

    This class handles different types of document inputs and chunking strategies, converting
    files and setting up appropriate chunking mechanisms based on the specified configuration.
    
    Attributes:
        
        files_or_dir : Union[str, List[str]]
            Path to directory containing files or list of file paths to process
        
        chunking : Chunking
            Configuration for the chunking strategy to be used
        
        tokenizer : Optional[str], default=None
            Name or path of the tokenizer model to be used (required for 'token' and 'sentence' chunking)
        
        embedding_model : str
            Name or path of the embedding model to be used
    """
    files_or_dir: str | List[str]
    chunking: Chunking
    tokenizer: Optional[str] = None
    embedding_model: str
    @model_validator(mode="after")
    def validate_ingestion(self) -> Self:
        if isinstance(self.files_or_dir, str):
            self.files_or_dir = pdf_converter.convert_directory(self.files_or_dir)
        elif isinstance(self.files_or_dir, list):
            self.files_or_dir = pdf_converter.multiple_convert(file_paths=self.files_or_dir)
        self.embedding_model = SentenceTransformerEmbeddings(model=self.embedding_model)
        if self.chunking.chunker == "token":
            if self.tokenizer is None:
                raise ValueError(f"Tokenizer cannot be None if {self.chunking.chunker} chunking approach is chosen")
            self.tokenizer = Tokenizer.from_pretrained(self.tokenizer)
            self.chunking = TokenChunker(tokenizer=self.tokenizer, chunk_size=self.chunking.chunk_size, chunk_overlap=self.chunking.chunk_overlap)
        elif self.chunking.chunker == "sentence":
            if self.tokenizer is None:
                raise ValueError(f"Tokenizer cannot be None if {self.chunking.chunker} chunking approach is chosen")
            self.tokenizer = Tokenizer.from_pretrained(self.tokenizer)
            self.chunking = SentenceChunker(tokenizer_or_token_counter=self.tokenizer, chunk_size=self.chunking.chunk_size, chunk_overlap=self.chunking.chunk_overlap, min_sentences_per_chunk=self.chunking.min_sentences)
        elif self.chunking.chunker == "late":
            self.chunking = LateChunker(embedding_model=self.embedding_model, chunk_size=self.chunking.chunk_size, min_characters_per_chunk=self.chunking.min_characters_per_chunk)
        elif self.chunking.chunker == "sdpm":
            self.chunking = SDPMChunker(embedding_model=self.embedding_model, chunk_size=self.chunking.chunk_size, threshold=self.chunking.similarity_threshold, min_sentences=self.chunking.min_sentences)
        elif self.chunking.chunker == "semantic":
            self.chunking = SemanticChunker(embedding_model=self.embedding_model, threshold=self.chunking.similarity_threshold, min_sentences=self.chunking.min_sentences, chunk_size=self.chunking.chunk_size)
        return self


    