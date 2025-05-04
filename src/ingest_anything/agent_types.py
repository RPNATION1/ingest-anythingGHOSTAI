from llama_index.core.llms.llm import LLM
from llama_index.core.agent.workflow import FunctionAgent, ReActAgent
from llama_index.core.tools import BaseTool, QueryEngineTool
from llama_index.core.indices.query.query_transform import HyDEQueryTransform
from llama_index.core.indices.query.query_transform.base import (
    StepDecomposeQueryTransform,
)
from llama_index.core.query_engine import TransformQueryEngine, MultiStepQueryEngine
from typing import Callable, Awaitable, List, Optional, Literal

try:
    from ingestion import (
        IngestAnything,
        IngestCode,
        BasePydanticVectorStore,
        BaseReader,
    )
except ModuleNotFoundError:
    from .ingestion import (
        IngestAnything,
        IngestCode,
        BasePydanticVectorStore,
        BaseReader,
    )


class IngestAnythingFunctionAgent(IngestAnything):
    def __init__(
        self,
        vector_database: BasePydanticVectorStore,
        llm: LLM,
        reader: Optional[BaseReader] = None,
        tools: Optional[List[BaseTool | Callable | Awaitable]] = None,
        query_transform: Optional[Literal["hyde", "multi_step"]] = None,
    ) -> None:
        super().__init__(vector_store=vector_database, reader=reader)
        self.llm = llm
        self.query_transform = query_transform
        if tools is None:
            tools = []
        self.tools = tools

    def ingest(
        self,
        files_or_dir: str | List[str],
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
        gemini_model: Optional[str] = None,
    ):
        self.vector_store_index = super().ingest(
            files_or_dir,
            embedding_model,
            chunker,
            tokenizer,
            chunk_size,
            chunk_overlap,
            similarity_threshold,
            min_characters_per_chunk,
            min_sentences,
            gemini_model,
        )

    def _get_query_engine_tool(self) -> None:
        query_engine = self.vector_store_index.as_query_engine(llm=self.llm)
        if self.query_transform == "hyde":
            qt = HyDEQueryTransform(llm=self.llm)
            self.query_engine = TransformQueryEngine(
                query_engine=query_engine, query_transform=qt
            )
        elif self.query_transform == "multi_step":
            qt = StepDecomposeQueryTransform(llm=self.llm, verbose=True)
            self.query_engine = MultiStepQueryEngine(
                query_engine=query_engine, query_transform=qt
            )
        else:
            self.query_engine = query_engine
        self.query_engine_tool = QueryEngineTool.from_defaults(
            query_engine=self.query_engine,
            name="Query Engine Tool",
            description="Retrieves information from a vector database",
        )

    def get_agent(
        self,
        name: str = "FunctionAgent",
        description: str = "A useful AI agent",
        system_prompt: str = "You are a useful assistant who uses the tools available to you whenever it is needed",
    ) -> FunctionAgent:
        self._get_query_engine_tool()
        agent_tools = self.tools + [self.query_engine_tool]
        agent = FunctionAgent(
            name=name,
            description=description,
            system_prompt=system_prompt,
            tools=agent_tools,
        )
        return agent


class IngestCodeFunctionAgent(IngestCode):
    def __init__(
        self,
        vector_database: BasePydanticVectorStore,
        llm: LLM,
        tools: Optional[List[BaseTool | Callable | Awaitable]] = None,
        query_transform: Optional[Literal["hyde", "multi_step"]] = None,
    ) -> None:
        super().__init__(vector_store=vector_database)
        self.llm = llm
        self.query_transform = query_transform
        if tools is None:
            tools = []
        self.tools = tools

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
        self.vector_store_index = super().ingest(
            files,
            embedding_model,
            language,
            return_type,
            tokenizer,
            chunk_size,
            include_nodes,
        )

    def _get_query_engine_tool(self) -> None:
        query_engine = self.vector_store_index.as_query_engine(llm=self.llm)
        if self.query_transform == "hyde":
            qt = HyDEQueryTransform(llm=self.llm)
            self.query_engine = TransformQueryEngine(
                query_engine=query_engine, query_transform=qt
            )
        elif self.query_transform == "multi_step":
            qt = StepDecomposeQueryTransform(llm=self.llm, verbose=True)
            self.query_engine = MultiStepQueryEngine(
                query_engine=query_engine, query_transform=qt
            )
        else:
            self.query_engine = query_engine
        self.query_engine_tool = QueryEngineTool.from_defaults(
            query_engine=self.query_engine,
            name="Query Engine Tool",
            description="Retrieves information from a vector database containing code snippets",
        )

    def get_agent(
        self,
        name: str = "FunctionAgent",
        description: str = "A useful AI agent",
        system_prompt: str = "You are a useful assistant who uses the tools available to you whenever it is needed",
    ) -> FunctionAgent:
        self._get_query_engine_tool()
        agent_tools = self.tools + [self.query_engine_tool]
        agent = FunctionAgent(
            llm=self.llm,
            name=name,
            description=description,
            system_prompt=system_prompt,
            tools=agent_tools,
        )
        return agent


class IngestAnythingReActAgent(IngestAnythingFunctionAgent):
    def get_agent(
        self,
        name="ReActAgent",
        description="A useful AI agent",
        system_prompt="You are a useful assistant who uses the tools available to you whenever it is needed",
    ) -> ReActAgent:
        self._get_query_engine_tool()
        agent_tools = self.tools + [self.query_engine_tool]
        agent = ReActAgent(
            llm=self.llm,
            name=name,
            description=description,
            system_prompt=system_prompt,
            tools=agent_tools,
        )
        return agent


class IngestCodeReActAgent(IngestCodeFunctionAgent):
    def get_agent(
        self,
        name="ReActAgent",
        description="A useful AI agent",
        system_prompt="You are a useful assistant who uses the tools available to you whenever it is needed",
    ) -> ReActAgent:
        self._get_query_engine_tool()
        agent_tools = self.tools + [self.query_engine_tool]
        agent = ReActAgent(
            llm=self.llm,
            name=name,
            description=description,
            system_prompt=system_prompt,
            tools=agent_tools,
        )
        return agent
