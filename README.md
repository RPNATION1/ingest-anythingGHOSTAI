# ingest-everything

## From data to vector database effortlessly

**`ingest-anything`** is a python package aimed at providing a smooth solution to ingest non-PDF files into vector databases, given that most ingestion pipelines are focused on PDF/markdown files. Using [chonkie](), [PdfItDown](), [Llamaindex](), [Sentence Transformers]() embeddings and [Qdrant](), `ingest-anything` proves a versatile package that automates files ingestion.

## Installation and usage

`ingest-anything` can be installed using `pip` in the following way:

```bash
pip install ingest-anything
# or, for a faster installation
uv pip install ingest-anything
```

And is available in your python scripts:

- You can **initialize** it:

```python
from ingest_anything.ingestion import IngestAnything, QdrantClient, AsyncQdrantClient

coll_name = "Flowers"
client = QdrantClient(api_key=os.getenv("qdrant_api_key"), url=os.getenv("qdrant_url"))
aclient = AsyncQdrantClient(api_key=os.getenv("qdrant_api_key"), url=os.getenv("qdrant_url"))
ingestor = IngestAnything(qdrant_client=client, async_qdrant_client=aclient, collection_name=coll_name, hybrid_search=True)
```

- And **ingest** your files:

```python
# with a list of files
ingestor.ingest(chunker="late", files_or_dir=['tests/data/test.docx', 'tests/data/test0.png', 'tests/data/test1.csv', 'tests/data/test2.json', 'tests/data/test3.md', 'tests/data/test4.xml', 'tests/data/test5.zip'], embedding_model="sentence-transformers/all-MiniLM-L6-v2")
# with a directory
ingestor.ingest(chunker="token", files_or_dir="tests/data", tokenizer="gpt2", embedding_model="sentence-transformers/all-MiniLM-L6-v2")
```

You can find a complete reference for the package in [REFERENCE.md](https://github.com/AstraBert/ingest-anything/tree/main/REFERENCE.md)

### Contributing

Contributions are always welcome!

Find contribution guidelines at [CONTRIBUTING.md](https://github.com/AstraBert/ingest-anything/tree/main/CONTRIBUTING.md)

### License and Funding

This project is open-source and is provided under an [MIT License](https://github.com/AstraBert/ingest-anything/tree/main/LICENSE).

If you found it useful, please consider [funding it](https://github.com/sponsors/AstraBert).