import os

from dotenv import load_dotenv

load_dotenv()

import google.generativeai as genai

from src.framework.interfaces.interfaces import ILLMClient


class GeminiClient(ILLMClient):
    """
    Gemini API client.
    """

    def __init__(self):

        api_key = os.getenv("GEMINI_API_KEY")

        if not api_key:
            raise ValueError("Gemini API key not found.")

        genai.configure(api_key=api_key)

        self.model = genai.GenerativeModel("gemini-2.5-flash")

    def generate(self, prompt: str, system_instruction: str = "") -> str:

        full_prompt = f"""
{system_instruction}

Question:
{prompt}
"""

        response = self.model.generate_content(full_prompt)

        return response.text