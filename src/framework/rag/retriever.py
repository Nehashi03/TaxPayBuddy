from src.framework.interfaces.interfaces import IVectorStore
from src.framework.core.data_models import SearchResult
from src.framework.utils.logger import Logger


class Retriever:
    """
    Retrieves relevant document chunks from the vector database.
    """

    def __init__(self, vector_store: IVectorStore):
        self.vector_store = vector_store

    def retrieve(
        self,
        query: str,
        collection_name: str,
        k: int = 3,
    ) -> SearchResult:
        """
        Search the vector database for the most relevant chunks.
        """

        Logger.info("Searching knowledge base...")

        result = self.vector_store.search(
            query=query,
            collection_name=collection_name,
            k=k,
        )

        Logger.success(f"{len(result.chunks)} chunks retrieved.")

        return result