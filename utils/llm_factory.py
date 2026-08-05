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

    Reads the provider and API key from ``st.session_state`` when available
    (set by the sidebar UI), falling back to environment variables.

    Args:
        model: Optional provider-specific model name.
        base_url: Optional Ollama server URL.

    Returns:
        A configured LangChain chat model.
    """
    provider = _get_provider()
    if provider == "gemini":
        api_key = _get_gemini_key()
        if not api_key:
            raise RuntimeError("Gemini API key is required. Add it in the sidebar.")
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=model or os.getenv("GEMINI_MODEL", "gemini-3.5-flash"),
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


def _get_provider() -> str:
    """Return the active provider ('ollama' or 'gemini')."""
    try:
        import streamlit as st

        return (st.session_state.get("llm_provider") or "ollama").strip().lower()
    except (ImportError, RuntimeError):
        pass
    return os.getenv("LLM_PROVIDER", "ollama").strip().lower()


def _get_gemini_key() -> str | None:
    """Return the Gemini API key from session state or environment."""
    try:
        import streamlit as st

        key = st.session_state.get("gemini_api_key", "").strip()
        if key:
            return key
    except (ImportError, RuntimeError):
        pass
    return os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
