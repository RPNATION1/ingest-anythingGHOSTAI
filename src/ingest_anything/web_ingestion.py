import time
import os
from typing import Literal, List, Optional

try:
    from .ingestion import (
        BasePydanticVectorStore,
        BaseReader,
        PyMuPDFReader,
        SimpleDirectoryReader,
        uuid,
        StorageContext,
        VectorStoreIndex,
        TextNode,
    )
    from .embeddings import ChonkieAutoEmbedding
    from .add_types import Converter, Chunking, IngestionInput
except ImportError:
    from ingestion import (
        BasePydanticVectorStore,
        BaseReader,
        PyMuPDFReader,
        SimpleDirectoryReader,
        uuid,
        StorageContext,
        VectorStoreIndex,
        TextNode,
    )
    from embeddings import ChonkieAutoEmbedding
    from add_types import Converter, Chunking, IngestionInput
from oarc_crawlers import WebCrawler

crawler = WebCrawler()
default_reader = PyMuPDFReader()
pdf_converter = Converter()


class IngestWeb:
    def __init__(
        self,
        vector_database: BasePydanticVectorStore,
        reader: Optional[BaseReader] = None,
    ) -> None:
        self.vector_store = vector_database
        if reader is None:
            self.reader = default_reader
        self.reader = reader

    async def _fetch_from_web(self, url: str) -> str:
        html_content = await crawler.fetch_url_content(url)
        current_time = str(time.time()).replace(".", "")
        with open(f"{current_time}.html", "w") as f:
            f.write(html_content)
        f.close()
        fl_pt = pdf_converter.convert(
            file_path=f"{current_time}.html", output_path=f"{current_time}.pdf"
        )
        os.remove(f"{current_time}.html")
        return fl_pt

    async def _batch_fetch_from_web(self, urls: List[str]):
        fls = []
        for url in urls:
            fl = await self._fetch_from_web(url)
            if fl is not None:
                fls.append(fl)
        if len(fls) == 0:
            raise ValueError("None of the passed URLs was correctly extracted")
        return fls

    async def ingest(
        self,
        urls: str | List[str],
        embedding_model: str,
        chunker: Literal[
            "token", "sentence", "semantic", "sdpm", "late", "neural", "slumber"
        ],
        tokenizer: Optional[str] = None,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        similarity_threshold: Optional[float] = None,
        min_characters_per_chunk: Optional[int] = None,
        min_sentences: Optional[int] = None,
        slumber_genie: Optional[Literal["openai", "gemini"]] = None,
        slumber_model: Optional[str] = None,
    ):
        if isinstance(urls, str):
            fl = await self._fetch_from_web(urls)
            if fl is None:
                raise ValueError("The passed URL was not correctly extracted")
            fls = [fl]
        if isinstance(urls, list):
            fls = await self._batch_fetch_from_web(urls)
        chunking = Chunking(
            chunker=chunker,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            similarity_threshold=similarity_threshold,
            min_characters_per_chunk=min_characters_per_chunk,
            min_sentences=min_sentences,
            slumber_genie=slumber_genie,
            slumber_model=slumber_model,
        )
        ingestion_input = IngestionInput(
            files_or_dir=fls,
            chunking=chunking,
            tokenizer=tokenizer,
            embedding_model=embedding_model,
        )
        docs = SimpleDirectoryReader(
            input_files=ingestion_input.files_or_dir,
            file_extractor={".pdf": self.reader},
        ).load_data()
        text = "\n\n---\n\n".join([d.text for d in docs])
        chunks = ingestion_input.chunking.chunk(text)
        nodes = [TextNode(text=c.text, id_=str(uuid.uuid4())) for c in chunks]
        storage_context = StorageContext.from_defaults(vector_store=self.vector_store)
        index = VectorStoreIndex(
            nodes=nodes,
            embed_model=ChonkieAutoEmbedding(model_name=embedding_model),
            show_progress=True,
            storage_context=storage_context,
        )
        for fl in fls:
            os.remove(fl)
        return index
