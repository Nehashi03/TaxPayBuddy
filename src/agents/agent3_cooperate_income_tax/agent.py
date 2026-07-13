from src.framework.core.base_agent import BaseAgent

from .config import (
    COLLECTION_NAME,
    TOP_K,
    SYSTEM_PROMPT_FILE
)


class CorporateIncomeTaxAgent(BaseAgent):
    """
    Agent responsible for answering
    Corporate Income Tax questions.
    """

   