👻 ghostai-ingest 👻
Effortless Data Ingestion for Vector Databases by GhostAI 🚀

Join Our Discord Community: https://discord.gg/y3ymyrveGb 💬
[GhostAI-Ingest Logo: In a rendered format, the logo image would appear here. Source URL: https://raw.githubusercontent.com/AstraBert/ingest-anything/main/logo.png]

ghostai-ingest is a white-label fork of ingest-anything, rebranded by GhostAI to deliver a powerful, open-source solution for ingesting non-PDF files into vector databases 🌟. Designed for GhostAI's cutting-edge workflows, this package avoids paid APIs by leveraging local LLMs and embedding models from Hugging Face (HF) containers 🛠️. It integrates chonkie (https://docs.chonkie.ai/getting-started/introduction), PdfItDown (https://github.com/AstraBert/PdfItDown), and LlamaIndex (https://www.llamaindex.ai) to provide an automated ingestion pipeline with minimal code ⚡.

Check out GhostAI's services on LinkedIn: https://www.linkedin.com/company/ghostai-services 📈
Learn more about ghostai-ingest on GhostAI's Documentation website (under construction) 📚

Workflow 🔄

[GhostAI-Ingest Workflow Diagram: In a rendered format, the workflow image would appear here. Source URL: https://raw.githubusercontent.com/AstraBert/ingest-anything/main/workflow.png]

📝 For Text Files
- Input files are converted to PDF by PdfItDown 📄.
- PDF text is extracted using a LlamaIndex-compatible reader 📖.
- Text is chunked using Chonkie's functionalities ✂️.
- Chunks are embedded with a local Hugging Face model (e.g., sentence-transformers/all-MiniLM-L6-v2) 🧠.
- Embeddings are loaded into a LlamaIndex-compatible vector database 💾.

💻 For Code Files
- Text is extracted from code files using LlamaIndex SimpleDirectoryReader 📂.
- Text is chunked using Chonkie's CodeChunker ✂️.
- Chunks are embedded with a local Hugging Face model (e.g., sentence-transformers/all-MiniLM-L6-v2) 🧠.
- Embeddings are loaded into a LlamaIndex-compatible vector database 💾.

🌐 For Web Data
- HTML content is scraped from URLs with crawlee (https://crawlee.dev) 🕸️.
- HTML files are converted to PDFs with PdfItDown 📄.
- Text is extracted from PDFs using LlamaIndex PyMuPdfReader 📖.
- Text is chunked using Chonkie's chunkers ✂️.
- Chunks are embedded with a local Hugging Face model (e.g., sentence-transformers/all-MiniLM-L6-v2) 🧠.
- Embeddings are loaded into a LlamaIndex-compatible vector database 💾.

🤖 For Agent Workflow
- Initialize a vector database (e.g., Qdrant, Weaviate) 🗄️.
- Initialize a local 7B language model (LLM) from Hugging Face (e.g., meta-llama/LLaMA-7B) 🧠.
- Create an IngestAgent instance ⚙️.
- Use the create_agent method to generate a specific agent type (e.g., IngestAnythingFunctionAgent, IngestCodeReActAgent) 🤖.
- Ingest data using the agent's ingest method 📥.
- Retrieve the agent using the get_agent method for querying and interaction 🔍.

Usage 🛠️

ghostai-ingest can be installed using pip:

pip install ghostai-ingest
# or, for a faster installation
uv pip install ghostai-ingest

Running with Hugging Face Containers 🐳

To run the 7B model locally, use Hugging Face containers. For example, pull and run a container for meta-llama/LLaMA-7B:

docker pull ghcr.io/huggingface/text-generation-inference:1.4
docker run -d --gpus all -p 8080:80 -v /path/to/models:/models ghcr.io/huggingface/text-generation-inference:1.4 --model-id meta-llama/LLaMA-7B

This sets up a local inference server for the 7B LLM, which you can point to in your scripts. Ensure you have sufficient GPU memory (e.g., 16GB VRAM) to run the 7B model efficiently 💻.

Initialize the Interface for Text-Based Files 📝

from qdrant_client import QdrantClient, AsyncQdrantClient
from llama_index.vector_stores.qdrant import QdrantVectorStore
from ghostai_ingest.ingestion import IngestAnything

client_qdrant = QdrantClient("http://localhost:6333")
aclient_qdrant = AsyncQdrantClient("http://localhost:6333")
vector_store_qdrant = QdrantVectorStore(
    collection_name="GhostAICollection", client=client_qdrant, aclient=aclient_qdrant
)
ingestor = IngestAnything(vector_store=vector_store_qdrant)

Ingest Your Files 📥

# With a list of files
ingestor.ingest(
    chunker="late",
    files_or_dir=[
        "tests/data/test.docx",
        "tests/data/test0.png",
        "tests/data/test1.csv",
        "tests/data/test2.json",
        "tests/data/test3.md",
        "tests/data/test4.xml",
        "tests/data/test5.zip",
    ],
    embedding_model="sentence-transformers/all-MiniLM-L6-v2",
)
# With a directory
ingestor.ingest(
    chunker="token",
    files_or_dir="tests/data",
    tokenizer="gpt2",
    embedding_model="sentence-transformers/all-MiniLM-L6-v2",
)

Initialize the Interface for Code Files 💻

import weaviate
from llama_index.vector_stores.weaviate import WeaviateVectorStore
from ghostai_ingest.ingestion import IngestCode

client_weaviate = weaviate.Client("http://localhost:8080")
vector_store_weaviate = WeaviateVectorStore(
    weaviate_client=client_weaviate, index_name="GhostAICollection"
)

ingestor = IngestCode(vector_store=vector_store_weaviate)

Ingest Your Code Files 📥

ingestor.ingest(
    files=[
        "tests/code/acronym.go",
        "tests/code/animal_magic.go",
        "tests/code/atbash_cipher_test.go",
    ],
    embedding_model="sentence-transformers/all-MiniLM-L6-v2",
    language="go",
)

Ingest Data from the Web 🌐

import weaviate
from llama_index.vector_stores.weaviate import WeaviateVectorStore
from ghostai_ingest.web_ingestion import IngestWeb

client_weaviate = weaviate.Client("http://localhost:8080")
vector_store_weaviate = WeaviateVectorStore(
    weaviate_client=client_weaviate, index_name="GhostAICollection"
)

ingestor = IngestWeb(vector_store=vector_store_weaviate)

Ingest from URLs 📥

import asyncio
async def main():
    await ingestor.ingest(
        urls=[
            "https://astrabert.github.io/hophop-science/AI-is-turning-nuclear-a-review/",
            "https://astrabert.github.io/hophop-science/BrAIn-next-generation-neurons/",
            "https://astrabert.github.io/hophop-science/Attention-and-open-source-is-all-you-need/",
        ],
        chunker="slumber",
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
    )

if __name__ == "__main__":
    asyncio.run(main())

Create a RAG Agent 🤖

from qdrant_client import QdrantClient
from transformers import pipeline
from llama_index.vector_stores.qdrant import QdrantVectorStore
from ghostai_ingest.agent import IngestAgent

# Initialize Vector Database
client = QdrantClient(":memory:")  # Or your Qdrant setup

# Initialize Local 7B LLM from Hugging Face
llm = pipeline("text-generation", model="meta-llama/LLaMA-7B", model_kwargs={"load_in_4bit": True})

# Initialize IngestAgent
agent_factory = IngestAgent()
vector_store = QdrantVectorStore(
    client=client, collection_name="GhostAICollection"
)

# Create Agent
agent = agent_factory.create_agent(
    vector_database=vector_store,
    llm=llm,
    ingestion_type="anything",  # or "code"
    agent_type="function_calling",  # or "react"
)

# Ingest Data
agent.ingest(
    files_or_dir="path/to/documents",
    embedding_model="sentence-transformers/all-mpnet-base-v2",
    chunker="semantic",
    similarity_threshold=0.8,
)

# Get Agent for Querying
function_agent = agent.get_agent()  # or react_agent = agent.get_agent() if you chose react

Agent Workflow Diagram 📊

graph LR
A[Initialize Vector Database] --> B(Initialize LLM);
B --> C{Create IngestAgent};
C --> D{Create Agent with create_agent};
D --> E{Ingest Data with ingest};
E --> F{Get Agent with get_agent};
F --> G[Ready for Querying];

You can find a complete reference for the package in REFERENCE.md (https://github.com/AstraBert/ingest-anything/tree/main/REFERENCE.md) 📜

Contributing 🤝

Contributions are always welcome! 🌟

Find contribution guidelines at CONTRIBUTING.md (https://github.com/AstraBert/ingest-anything/tree/main/CONTRIBUTING.md)

License and Funding 📋

This project is open-source and is provided under an MIT License (https://github.com/AstraBert/ingest-anything/tree/main/LICENSE).

If you found it useful, please consider funding it (https://github.com/sponsors/AstraBert) 💸.

https://github.com/GhostAI/ghostai-ingest/blob/main/README.md
