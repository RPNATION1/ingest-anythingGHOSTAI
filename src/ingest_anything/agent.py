try:
    from agent_types import *
except ModuleNotFoundError:
    from .agent_types import *


class IngestAgent:
    def __init__(self) -> None:
        pass
    def create_agent(
        self,
        vector_database: BasePydanticVectorStore,
        llm: LLM,
        reader: Optional[BaseReader] = None,
        ingestion_type: Literal["anything", "code"] = "anything",
        agent_type: Literal["function_calling", "react"] = "function_calling",
        tools: Optional[List[BaseTool | Callable | Awaitable]] = None,
        query_transform: Optional[Literal["hyde", "multi_step"]] = None,
    ) -> (
        IngestAnythingFunctionAgent
        | IngestAnythingReActAgent
        | IngestCodeFunctionAgent
        | IngestCodeReActAgent
    ):
        if ingestion_type == "anything":
            if agent_type == "function_calling":
                return IngestAnythingFunctionAgent(
                    vector_database=vector_database,
                    reader=reader,
                    llm=llm,
                    tools=tools,
                    query_transform=query_transform,
                )
            else:
                return IngestAnythingReActAgent(
                    vector_database=vector_database,
                    reader=reader,
                    llm=llm,
                    tools=tools,
                    query_transform=query_transform,
                )
        else:
            if agent_type == "function_calling":
                return IngestCodeFunctionAgent(
                    vector_database=vector_database,
                    llm=llm,
                    tools=tools,
                    query_transform=query_transform,
                )
            else:
                return IngestCodeReActAgent(
                    vector_database=vector_database,
                    llm=llm,
                    tools=tools,
                    query_transform=query_transform,
                )

