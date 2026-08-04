"""LLM Factory — single source of truth for creating LLM instances.

Usage:
    from utils.llm_factory import get_llm
    llm = get_llm()
"""

from __future__ import annotations

import os
from typing import Any

from dotenv import load_dotenv

load_dotenv()


def get_llm(
    model: str | None = None,
    base_url: str | None = None,
) -> Any:
    """Create the configured chat model.

    Args:
        model: Optional provider-specific model name.
        base_url: Optional Ollama server URL.

    Returns:
        A configured LangChain chat model.
    """
    provider = os.getenv("LLM_PROVIDER", "ollama").strip().lower()
    if provider == "gemini":
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY is required when LLM_PROVIDER=gemini.")
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=model or os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
            google_api_key=api_key,
            temperature=0,
        )
    if provider != "ollama":
        raise RuntimeError("LLM_PROVIDER must be either 'ollama' or 'gemini'.")

    from langchain_ollama import ChatOllama
    return ChatOllama(
        model=model or os.getenv("OLLAMA_MODEL", "llama3.2"),
        base_url=base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        temperature=0,
    )
