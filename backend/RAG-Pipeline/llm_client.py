"""
llm_client.py
"""

import logging
import time
from typing import Optional

import requests

import config

logger = logging.getLogger(__name__)


class LLMError(RuntimeError):
    """Raised when the configured LLM backend can't be reached or errors out."""


class LLMClient:
    """A small dispatcher that sends (system_prompt, user_prompt) to whichever
    LLM provider is configured, and returns the generated answer text."""

    def __init__(self, provider: str = None):
        self.provider = (provider or config.LLM_PROVIDER).strip().lower()

        self._dispatch = {
            "ollama": self._generate_ollama,
            "groq": self._generate_openai_compatible_groq,
            "openai_compatible": self._generate_openai_compatible_generic,
        }
        if self.provider not in self._dispatch:
            raise LLMError(
                f"Unknown LLM_PROVIDER '{self.provider}'. "
                f"Expected one of: {', '.join(self._dispatch)}."
            )

    # -- Step 4a — Public entry point ---------------------------------------
    def generate(self, system_prompt: str, user_prompt: str) -> str:
        """
        Sends one request to the configured LLM backend, retrying a
        couple of times on transient network errors before giving up.
        """
        last_error: Optional[Exception] = None
        for attempt in range(1, config.LLM_MAX_RETRIES + 2):
            try:
                return self._dispatch[self.provider](system_prompt, user_prompt)
            except (requests.RequestException, LLMError) as e:
                last_error = e
                logger.warning(
                    f"LLM call via '{self.provider}' failed on attempt "
                    f"{attempt}/{config.LLM_MAX_RETRIES + 1}: {e}"
                )
                if attempt <= config.LLM_MAX_RETRIES:
                    time.sleep(1.5 * attempt)  # simple backoff

        raise LLMError(
            f"LLM provider '{self.provider}' failed after "
            f"{config.LLM_MAX_RETRIES + 1} attempt(s): {last_error}"
        )

    # -- Step 4b — Ollama (local) --------------------------------------------
    def _generate_ollama(self, system_prompt: str, user_prompt: str) -> str:
        url = f"{config.OLLAMA_BASE_URL.rstrip('/')}/api/chat"
        payload = {
            "model": config.OLLAMA_MODEL,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "stream": False,
            "options": {"temperature": config.LLM_TEMPERATURE},
        }
        try:
            response = requests.post(url, json=payload, timeout=config.LLM_TIMEOUT_SECONDS)
        except requests.ConnectionError as e:
            raise LLMError(
                f"Could not reach Ollama at {config.OLLAMA_BASE_URL}. "
                f"Is it installed and running? See https://ollama.com — "
                f"then run e.g. `ollama pull {config.OLLAMA_MODEL}`."
            ) from e

        if response.status_code != 200:
            raise LLMError(f"Ollama returned HTTP {response.status_code}: {response.text[:300]}")

        data = response.json()
        content = data.get("message", {}).get("content")
        if not content:
            raise LLMError(f"Ollama response had no message content: {data}")
        return content.strip()

    # -- Step 4c — Groq (OpenAI-compatible cloud API) ------------------------
    def _generate_openai_compatible_groq(self, system_prompt: str, user_prompt: str) -> str:
        if not config.GROQ_API_KEY:
            raise LLMError(
                "LLM_PROVIDER is 'groq' but GROQ_API_KEY is not set. "
                "Get a free key at https://console.groq.com/keys and add it to .env."
            )
        return self._call_openai_compatible(
            base_url=config.GROQ_BASE_URL,
            api_key=config.GROQ_API_KEY,
            model=config.GROQ_MODEL,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )

    # -- Step 4d — Any other OpenAI-chat-format endpoint ----------------------
    def _generate_openai_compatible_generic(self, system_prompt: str, user_prompt: str) -> str:
        if not config.OPENAI_COMPATIBLE_BASE_URL or not config.OPENAI_COMPATIBLE_MODEL:
            raise LLMError(
                "LLM_PROVIDER is 'openai_compatible' but OPENAI_COMPATIBLE_BASE_URL "
                "and/or OPENAI_COMPATIBLE_MODEL is not set in .env."
            )
        return self._call_openai_compatible(
            base_url=config.OPENAI_COMPATIBLE_BASE_URL,
            api_key=config.OPENAI_COMPATIBLE_API_KEY,
            model=config.OPENAI_COMPATIBLE_MODEL,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )

    # -- Shared OpenAI-chat-format request helper -----------------------------
    def _call_openai_compatible(
        self, base_url: str, api_key: str, model: str, system_prompt: str, user_prompt: str
    ) -> str:
        url = f"{base_url.rstrip('/')}/chat/completions"
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": config.LLM_TEMPERATURE,
            "max_tokens": config.LLM_MAX_TOKENS,
        }

        response = requests.post(url, json=payload, headers=headers, timeout=config.LLM_TIMEOUT_SECONDS)
        if response.status_code != 200:
            raise LLMError(f"{url} returned HTTP {response.status_code}: {response.text[:300]}")

        data = response.json()
        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError) as e:
            raise LLMError(f"Unexpected response shape from {url}: {data}") from e

        if not content:
            raise LLMError(f"Empty completion returned from {url}: {data}")
        return content.strip()
