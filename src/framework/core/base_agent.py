from abc import ABC

from src.framework.interfaces.interfaces import (
    IAgent,
    ILLMClient,
    IVectorStore,
)

from src.framework.rag.retriever import Retriever
from src.framework.core.data_models import RAGResponse


class BaseAgent(IAgent, ABC):
    """
    Base class for all TaxPayBuddy agents.
    """

    def __init__(
        self,
        llm: ILLMClient,
        vector_store: IVectorStore,
        collection_name: str,
        system_prompt_file: str,
        top_k: int,
    ):

        self.llm = llm
        self.retriever = Retriever(vector_store)

        self.collection_name = collection_name
        self.system_prompt_file = system_prompt_file
        self.top_k = top_k

    def process_query(self, question: str) -> RAGResponse:
        """
        Process the user's question using the RAG pipeline.
        """

        # Retrieve relevant chunks
        results = self.retriever.retrieve(
            query=question,
            collection_name=self.collection_name,
            top_k=self.top_k,
        )

        # Build context
        # Build context
        context = "\n\n".join(
            f"Reference {i + 1}:\n{chunk.text}"
            for i, chunk in enumerate(results.chunks)
        )
        
        

        # Read system prompt
        with open(
            self.system_prompt_file,
            "r",
            encoding="utf-8",
        ) as file:
            system_prompt = file.read()

        # Generate answer
        answer = self.llm.generate(
            prompt=question,
            system_instruction=f"""
{system_prompt}

Context:
{context}
"""
        )

        # Return response
        return RAGResponse(
            question=question,
            retrieved_chunks=results.chunks,
            answer=answer,
        )