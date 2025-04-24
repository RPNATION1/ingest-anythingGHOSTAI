from add_types import IngestionInput, Chunking, Tokenizer, TokenChunker, SentenceTransformerEmbeddings
import os
from pydantic import ValidationError
import pathlib
from collections import Counter

def test_chunking():
    test_cases = [
        {
            "chunker": "token",
            "chunk_size": 129,
            "chunk_overlap": 100,
            "similarity_threshold": 0.1,
            "min_characters_per_chunk": 100,
            "min_sentences": 4,
            "expected": ["token", 129, 100, 0.1, 100, 4]
        },
        {
            "chunker": "token",
            "chunk_size": None,
            "chunk_overlap": None,
            "similarity_threshold": None,
            "min_characters_per_chunk": None,
            "min_sentences": None,
            "expected": ["token", 512, 128, 0.7, 24, 1]
        },
        {
            "chunker": "toke",
            "chunk_size": 129,
            "chunk_overlap": 100,
            "similarity_threshold": 0.1,
            "min_characters_per_chunk": 100,
            "min_sentences": 4,
            "expected": None
        },
    ]
    for c in test_cases:
        try:
            chunks = Chunking(chunker=c["chunker"], chunk_size=c["chunk_size"], chunk_overlap=c["chunk_overlap"], similarity_threshold = c["similarity_threshold"], min_characters_per_chunk=c["min_characters_per_chunk"], min_sentences=c["min_sentences"])
        except ValidationError:
            outcome = None
        else:
            outcome = [chunks.chunker, chunks.chunk_size, chunks.chunk_overlap, chunks.similarity_threshold, chunks.min_characters_per_chunk, chunks.min_sentences]
        assert outcome == c["expected"]

def test_ingestion_input():
    test_cases = [
        {
            "chunker": "token",
            "chunk_size": None,
            "chunk_overlap": None,
            "similarity_threshold": None,
            "min_characters_per_chunk": None,
            "min_sentences": None,
            "files_or_dir": "tests/data",
            "tokenizer": "gpt2",
            "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
            "expected": [Counter(['tests/data/test.pdf', 'tests/data/test0.pdf', 'tests/data/test1.pdf', 'tests/data/test2.pdf', 'tests/data/test3.pdf', 'tests/data/test4.pdf', 'tests/data/test5.pdf']), True, True, True],
        },
        {
            "chunker": "token",
            "chunk_size": None,
            "chunk_overlap": None,
            "similarity_threshold": None,
            "min_characters_per_chunk": None,
            "min_sentences": None,
            "files_or_dir": ['tests/data/test.docx', 'tests/data/test0.png', 'tests/data/test1.csv', 'tests/data/test2.json', 'tests/data/test3.md', 'tests/data/test4.xml', 'tests/data/test5.zip'],
            "tokenizer": "gpt2",
            "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
            "expected": [Counter(['tests/data/test.pdf', 'tests/data/test0.pdf', 'tests/data/test1.pdf', 'tests/data/test2.pdf', 'tests/data/test3.pdf', 'tests/data/test4.pdf', 'tests/data/test5.pdf']), True, True, True]
        },
        {
            "chunker": "token",
            "chunk_size": None,
            "chunk_overlap": None,
            "similarity_threshold": None,
            "min_characters_per_chunk": None,
            "min_sentences": None,
            "files_or_dir": "tests/data",
            "tokenizer": None,
            "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
            "expected": None
        },
        {
            "chunker": "token",
            "chunk_size": None,
            "chunk_overlap": None,
            "similarity_threshold": None,
            "min_characters_per_chunk": None,
            "min_sentences": None,
            "files_or_dir": "tests/err",
            "tokenizer": "gpt2",
            "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
            "expected": None
        },
        {
            "chunker": "token",
            "chunk_size": None,
            "chunk_overlap": None,
            "similarity_threshold": None,
            "min_characters_per_chunk": None,
            "min_sentences": None,
            "files_or_dir": 3,
            "tokenizer": "gpt2",
            "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
            "expected": None
        },
    ]
    for c in test_cases:
        try:
            chunks = Chunking(chunker=c["chunker"], chunk_size=c["chunk_size"], chunk_overlap=c["chunk_overlap"], similarity_threshold = c["similarity_threshold"], min_characters_per_chunk=c["min_characters_per_chunk"], min_sentences=c["min_sentences"])
            ingestion = IngestionInput(chunking=chunks, files_or_dir=c["files_or_dir"], tokenizer=c["tokenizer"], embedding_model=c["embedding_model"])
        except ValidationError:
            outcome = None
        else:
            outcome = [Counter(ingestion.files_or_dir), isinstance(ingestion.chunking, TokenChunker), isinstance(ingestion.tokenizer, Tokenizer), isinstance(ingestion.embedding_model, SentenceTransformerEmbeddings)]
        for f in ['tests/data/test.pdf', 'tests/data/test0.pdf', 'tests/data/test1.pdf', 'tests/data/test2.pdf', 'tests/data/test3.pdf', 'tests/data/test4.pdf', 'tests/data/test5.pdf']:
            if pathlib.Path(f).is_file():
                os.remove(f)
        assert outcome == c["expected"]