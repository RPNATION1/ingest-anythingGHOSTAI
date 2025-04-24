from ingestion import IngestAnything, QdrantClient, AsyncQdrantClient, VectorStoreIndex
import uuid
import random as r
import pathlib
import os
from dotenv import load_dotenv

load_dotenv()

client = QdrantClient(api_key=os.getenv("qdrant_api_key"), url=os.getenv("qdrant_url"))
aclient = AsyncQdrantClient(api_key=os.getenv("qdrant_api_key"), url=os.getenv("qdrant_url"))
truths = {0: True, 1: False}

def test_ingestion():
    test_cases = [
        {
            "chunker": "late",
            "chunk_size": None,
            "chunk_overlap": None,
            "similarity_threshold": None,
            "min_characters_per_chunk": None,
            "min_sentences": None,
            "files_or_dir": "tests/data",
            "tokenizer": "gpt2",
            "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
            "expected": [True, True],
        },
        {
            "chunker": "semantic",
            "chunk_size": None,
            "chunk_overlap": None,
            "similarity_threshold": None,
            "min_characters_per_chunk": None,
            "min_sentences": None,
            "files_or_dir": ['tests/data/test.docx', 'tests/data/test0.png', 'tests/data/test1.csv', 'tests/data/test2.json', 'tests/data/test3.md', 'tests/data/test4.xml', 'tests/data/test5.zip'],
            "tokenizer": "gpt2",
            "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
            "expected": [True, True],
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
            "expected": [True, True],
        },
        {
            "chunker": "sdpm",
            "chunk_size": None,
            "chunk_overlap": None,
            "similarity_threshold": None,
            "min_characters_per_chunk": None,
            "min_sentences": None,
            "files_or_dir": ['tests/data/test.docx', 'tests/data/test0.png', 'tests/data/test1.csv', 'tests/data/test2.json', 'tests/data/test3.md', 'tests/data/test4.xml', 'tests/data/test5.zip'],
            "tokenizer": "gpt2",
            "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
            "expected": [True, True],
        },
        {
            "chunker": "sentence",
            "chunk_size": None,
            "chunk_overlap": None,
            "similarity_threshold": None,
            "min_characters_per_chunk": None,
            "min_sentences": None,
            "files_or_dir": ['tests/data/test.docx', 'tests/data/test0.png', 'tests/data/test1.csv', 'tests/data/test2.json', 'tests/data/test3.md', 'tests/data/test4.xml', 'tests/data/test5.zip'],
            "tokenizer": "gpt2",
            "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
            "expected": [True, True],
        },
    ]
    for c in test_cases:
        try:
            coll_name = str(uuid.uuid4())
            hybr = truths[int(round(r.random(),0))]
            ingestor = IngestAnything(qdrant_client=client, async_qdrant_client=aclient, collection_name=coll_name, hybrid_search=hybr)
            index = ingestor.ingest(chunker=c["chunker"], chunk_size=c["chunk_size"], chunk_overlap=c["chunk_overlap"], similarity_threshold = c["similarity_threshold"], min_characters_per_chunk=c["min_characters_per_chunk"], min_sentences=c["min_sentences"],files_or_dir=c["files_or_dir"], tokenizer=c["tokenizer"], embedding_model=c["embedding_model"])
            outcome = [isinstance(index, VectorStoreIndex), client.collection_exists(collection_name=coll_name)]
        except Exception as e:
            outcome = [None, e.__str__()]
        for f in ['tests/data/test.pdf', 'tests/data/test0.pdf', 'tests/data/test1.pdf', 'tests/data/test2.pdf', 'tests/data/test3.pdf', 'tests/data/test4.pdf', 'tests/data/test5.pdf']:
            if pathlib.Path(f).is_file():
                os.remove(f)
        client.delete_collection(collection_name=coll_name)
        assert outcome == c["expected"]