import time
try:
    from .ingestion import BasePydanticVectorStore, BaseReader, Optional, PyMuPDFReader
    from .add_types import Converter
except ImportError:
    from ingestion import BasePydanticVectorStore, BaseReader, Optional, PyMuPDFReader
    from add_types import Converter
from oarc_crawlers import WebCrawler


crawler = WebCrawler()
default_reader = PyMuPDFReader()
pdf_converter = Converter()

class IngestWeb():
    def __init__(
        self,
        vector_database: BasePydanticVectorStore,
        reader: Optional[BaseReader] = None,
    ) -> None:
        self.vector_store = vector_database
        if self.reader is None:
            reader = default_reader
        self.reader=reader
    async def _fetch_from_web(self, url: str, extract_text: bool = True) -> str:
        html_content = await crawler.fetch_url_content(url)
        if extract_text:
            text_content = await crawler.extract_text_from_html(html_content)
        else:
            current_time = str(time.time()).replace(".","")
            with open(f"{current_time}.html", "w") as f:
                f.write(html_content)
            f.close()
            pdf_converter.convert(file_path=f"{current_time}.html", output_path=f"{current_time}.pdf")
            content = self.reader.load_data(file_path=f"{current_time}.pdf")
            text_content = content[0].text
        return text_content
