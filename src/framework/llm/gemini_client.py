import google.generativeai as genai

from src.framework.interfaces.interfaces import ILLMClient
from src.framework.utils.config_manager import ConfigManager
from src.framework.utils.logger import Logger


class GeminiClient(ILLMClient):
    """
    Gemini implementation of the LLM interface.
    """

    def __init__(self):

        config = ConfigManager()

        api_key = config.get_gemini_api_key()

        genai.configure(api_key=api_key)

        self.model = genai.GenerativeModel("gemini-2.5-flash")

    def generate(self, prompt: str) -> str:
        """
        Generate a response from Gemini.
        """

        Logger.info("Generating response from Gemini...")

        response = self.model.generate_content(prompt)

        Logger.success("Response generated successfully.")

        return response.text