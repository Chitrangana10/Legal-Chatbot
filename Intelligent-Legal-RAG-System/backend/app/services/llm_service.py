"""Generate legal answers through the configured Gemini model provider."""

from __future__ import annotations

from typing import Optional

from backend.app.config import get_settings


class GeminiProvider:
    """Small Gemini text generation provider backed by google-generativeai."""

    def __init__(self, api_key: Optional[str] = None, model_name: Optional[str] = None) -> None:
        """Create a Gemini provider from explicit values or application settings."""
        settings = get_settings()
        self.api_key = api_key or settings.gemini_api_key
        self.model_name = model_name or settings.default_llm_model

    def generate(self, prompt: str) -> str:
        """Generate a response for the supplied prompt, returning errors as text."""
        if not self.api_key:
            return "Gemini error: GEMINI_API_KEY is not configured."

        try:
            import google.generativeai as genai

            genai.configure(api_key=self.api_key)
            model = genai.GenerativeModel(self.model_name)
            response = model.generate_content(
                prompt,
                generation_config={"temperature": 0.1},
            )
            text = getattr(response, "text", None)
            if not text:
                return "Gemini error: The model returned an empty response."
            return text.strip()
        except Exception as exc:
            return f"Gemini error: {exc}"


def generate(prompt: str, model: Optional[str] = None) -> str:
    """Generate text with the configured Gemini provider."""
    return GeminiProvider(model_name=model).generate(prompt)


def generate_answer(prompt: str, provider: Optional[str] = None, model: Optional[str] = None) -> str:
    """Generate an answer with the configured LLM provider."""
    if provider and provider.lower() != "gemini":
        return f"LLM error: unsupported provider '{provider}'."
    return generate(prompt, model=model)

