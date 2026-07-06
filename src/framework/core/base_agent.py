from abc import ABC

from src.framework.interfaces.interfaces import (
    IAgent,
    ILLMClient,
    IVectorStore,
)
from src.framework.rag.retriever import Retriever
from src.framework.core.data_models import RAGResponse
from src.framework.utils.logger import Logger


class BaseAgent(IAgent, ABC):
    """
    Base class for all TaxPayBuddy agents.
    """

    def __init__(
        self,
        llm: ILLMClient,
        vector_store: IVectorStore,
        collection_name: str,
        system_prompt: str,
    ):

        self.llm = llm
        self.retriever = Retriever(vector_store)
        self.collection_name = collection_name
        self.system_prompt = system_prompt

    def process_query(self, query: str) -> RAGResponse:
        """
        Complete RAG pipeline.
        """

        Logger.info("Processing user query...")

        # Retrieve relevant chunks
        search_result = self.retriever.retrieve(
            query=query,
            collection_name=self.collection_name,
        )

        # ----- Traceability -----
        Logger.info("Retrieved Chunks")

        for chunk in search_result.chunks:
            print("-" * 60)
            print(chunk.text)

        # Build context
        context = "\n\n".join(
            chunk.text
            for chunk in search_result.chunks
        )

        # Build prompt
        prompt = f"""
{self.system_prompt}

Context:
{context}

Question:
{query}

Answer:
"""

        # Generate response
        answer = self.llm.generate(prompt)

        Logger.success("Answer generated.")

        return RAGResponse(
            question=query,
            retrieved_chunks=search_result.chunks,
            answer=answer,
        )