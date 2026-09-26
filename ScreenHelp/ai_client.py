"""
ai_client.py - AI provider abstraction layer for ScreenHelp
Supports: OpenAI (GPT-4o), Google Gemini (gemini-1.5-flash), Anthropic Claude,
          Llama (via Groq API or local Ollama)

Each provider's stream_response() is a generator that yields text chunks,
allowing the UI to display tokens as they arrive (real-time typing effect).

Usage:
    client = AIClient(provider="openai", api_key="sk-...", model="gpt-4o")
    for chunk in client.stream_response(image_bytes, question):
        print(chunk, end="", flush=True)

    # Llama via Groq:
    client = AIClient(provider="llama", api_key="gsk_...", model="llama-4-scout-17b-16e-instruct")
    # Llama via local Ollama (no API key needed, pass "ollama"):
    client = AIClient(provider="llama", api_key="ollama", model="llama3.2-vision")
"""

import base64
import io
import logging
from typing import Generator, Optional

logger = logging.getLogger(__name__)

# System prompt injected into every AI request
SYSTEM_PROMPT = (
    "You are ScreenHelp, an expert real-time AI screen assistant. "
    "Your primary goal is to automatically detect and solve any question, coding bug, "
    "error message, quiz, multiple-choice problem, or document task shown on the user's screen.\n\n"
    "Output Format Rules:\n"
    "1. **Direct Answer First**: Provide the definitive answer, correct choice, code fix, or solution at the very top.\n"
    "2. **Concise Explanation**: Follow with a brief, clear explanation of why it is correct.\n"
    "3. **Zero Fluff**: Do not include conversational filler like 'Sure!' or 'Here is your answer'."
)


class AIClient:
    """
    Unified interface for OpenAI, Google Gemini, Anthropic Claude,
    and Llama (Groq API / local Ollama).
    Instantiate with a provider name, API key, and model string.
    Call stream_response() to get a token generator.
    """

    PROVIDERS = ("openai", "gemini", "anthropic", "llama")

    def __init__(self, provider: str, api_key: str, model: str):
        """
        Args:
            provider: One of "openai", "gemini", "anthropic".
            api_key:  The API key for the chosen provider.
            model:    The model identifier string.
        """
        if provider not in self.PROVIDERS:
            raise ValueError(f"Unknown provider '{provider}'. Choose from: {self.PROVIDERS}")

        self.provider = provider
        self.api_key = api_key
        self.model = model

    # ── Public API ─────────────────────────────────────────────────────────────

    def stream_response(
        self,
        image_bytes: bytes,
        question: str = "",
    ) -> Generator[str, None, None]:
        """
        Stream the AI response for the given screenshot and question.

        Args:
            image_bytes: Raw PNG/JPEG bytes of the screenshot.
            question:    User's text question (may be empty).

        Yields:
            String chunks (tokens) as they arrive from the API.

        Raises:
            RuntimeError: if the API call fails (catch in calling code).
        """
        user_text = question.strip() or (
            "Analyze this screenshot image. Automatically detect and extract any question, "
            "problem, coding error, quiz, or exercise shown on the screen. "
            "State the direct correct answer/solution clearly at the very top, "
            "followed by a concise step-by-step explanation."
        )

        if self.provider == "openai":
            yield from self._stream_openai(image_bytes, user_text)
        elif self.provider == "gemini":
            yield from self._stream_gemini(image_bytes, user_text)
        elif self.provider == "anthropic":
            yield from self._stream_anthropic(image_bytes, user_text)
        elif self.provider == "llama":
            yield from self._stream_llama(image_bytes, user_text)

    # ── OpenAI ─────────────────────────────────────────────────────────────────

    def _stream_openai(self, image_bytes: bytes, question: str) -> Generator[str, None, None]:
        """Stream from OpenAI GPT-4o using the vision API."""
        try:
            from openai import OpenAI
        except ImportError:
            raise RuntimeError(
                "openai package is not installed. Run: pip install openai"
            )

        b64_image = base64.b64encode(image_bytes).decode("utf-8")

        client = OpenAI(api_key=self.api_key)

        messages = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{b64_image}",
                            "detail": "high",
                        },
                    },
                    {
                        "type": "text",
                        "text": question,
                    },
                ],
            },
        ]

        try:
            with client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=2048,
                stream=True,
            ) as stream:
                for chunk in stream:
                    delta = chunk.choices[0].delta
                    if delta and delta.content:
                        yield delta.content
        except Exception as exc:
            raise RuntimeError(f"OpenAI API error: {exc}") from exc

    # ── Google Gemini ──────────────────────────────────────────────────────────

    def _stream_gemini(self, image_bytes: bytes, question: str) -> Generator[str, None, None]:
        """Stream from Google Gemini using the new google.genai SDK."""
        try:
            from google import genai
            from google.genai import types as genai_types
        except ImportError:
            raise RuntimeError(
                "google-genai not installed. Run: pip install google-genai"
            )

        client = genai.Client(api_key=self.api_key)

        # Encode image as inline base64 part
        b64_image = base64.b64encode(image_bytes).decode("utf-8")

        contents = [
            genai_types.Content(
                role="user",
                parts=[
                    genai_types.Part.from_bytes(data=image_bytes, mime_type="image/png"),
                    genai_types.Part.from_text(text=question),
                ],
            )
        ]

        config = genai_types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            max_output_tokens=2048,
        )

        models_to_try = [self.model]
        for fallback in ["gemini-flash-latest", "gemini-3.1-flash-lite", "gemini-2.5-flash-lite", "gemini-3.8-flash"]:
            if fallback not in models_to_try:
                models_to_try.append(fallback)

        last_exc = None
        for model_name in models_to_try:
            try:
                for chunk in client.models.generate_content_stream(
                    model=model_name,
                    contents=contents,
                    config=config,
                ):
                    if chunk.text:
                        yield chunk.text
                return  # Stream completed successfully
            except Exception as exc:
                last_exc = exc
                err_str = str(exc)
                if "503" in err_str or "404" in err_str or "not found" in err_str.lower() or "high demand" in err_str.lower():
                    logger.warning(f"[gemini] Model {model_name} unavailable ({err_str[:60]}), trying fallback...")
                    continue
                raise RuntimeError(f"Gemini API error: {exc}") from exc

        if last_exc:
            raise RuntimeError(f"Gemini API error: {last_exc}") from last_exc

    # ── Anthropic Claude ───────────────────────────────────────────────────────

    def _stream_anthropic(self, image_bytes: bytes, question: str) -> Generator[str, None, None]:
        """Stream from Anthropic Claude using the anthropic SDK."""
        try:
            import anthropic
        except ImportError:
            raise RuntimeError(
                "anthropic package is not installed. Run: pip install anthropic"
            )

        b64_image = base64.b64encode(image_bytes).decode("utf-8")

        client = anthropic.Anthropic(api_key=self.api_key)

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": b64_image,
                        },
                    },
                    {
                        "type": "text",
                        "text": question,
                    },
                ],
            }
        ]

        try:
            with client.messages.stream(
                model=self.model,
                max_tokens=2048,
                system=SYSTEM_PROMPT,
                messages=messages,
            ) as stream:
                for text_chunk in stream.text_stream:
                    yield text_chunk
        except Exception as exc:
            raise RuntimeError(f"Anthropic API error: {exc}") from exc

    # ── Llama (Groq API or local Ollama) ───────────────────────────────────────

    def _stream_llama(self, image_bytes: bytes, question: str) -> Generator[str, None, None]:
        """
        Stream from Llama vision models.

        Routing:
          - If api_key == "ollama"  → uses local Ollama (http://localhost:11434)
          - Otherwise               → uses Groq's OpenAI-compatible endpoint
                                      (get a free key at console.groq.com)

        Recommended vision-capable models:
          Groq : llama-4-scout-17b-16e-instruct (latest Llama 4 Scout w/ vision)
                 llama-4-maverick-17b-128e-instruct
          Ollama: llama3.2-vision  (11B, local)
        """
        try:
            from openai import OpenAI  # Groq uses an OpenAI-compatible API
        except ImportError:
            raise RuntimeError(
                "openai package is not installed. Run: pip install openai"
            )

        b64_image = base64.b64encode(image_bytes).decode("utf-8")

        use_ollama = self.api_key.strip().lower() == "ollama"

        if use_ollama:
            # Local Ollama server — no API key required
            client = OpenAI(
                base_url="http://localhost:11434/v1",
                api_key="ollama",  # required by SDK but unused by Ollama
            )
            logger.info("[llama] Using local Ollama endpoint.")
        else:
            # Groq cloud API — OpenAI-compatible
            client = OpenAI(
                api_key=self.api_key,
                base_url="https://api.groq.com/openai/v1",
            )
            logger.info("[llama] Using Groq cloud API.")

        messages = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{b64_image}",
                        },
                    },
                    {
                        "type": "text",
                        "text": question,
                    },
                ],
            },
        ]

        try:
            stream = client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=2048,
                stream=True,
            )
            for chunk in stream:
                delta = chunk.choices[0].delta
                if delta and delta.content:
                    yield delta.content
        except Exception as exc:
            source = "Ollama" if use_ollama else "Groq"
            raise RuntimeError(f"{source} Llama API error: {exc}") from exc


def validate_api_key(provider: str, api_key: str) -> tuple[bool, str]:
    """
    Quick validation that an API key looks plausible (format check only,
    no actual network call). Returns (is_valid, message).
    """
    if not api_key or not api_key.strip():
        return False, f"No API key provided for {provider}."

    key = api_key.strip()

    if provider == "openai" and not key.startswith("sk-"):
        return False, "OpenAI keys typically start with 'sk-'. Please double-check."

    if provider == "llama":
        # Groq keys start with "gsk_"; local Ollama uses keyword "ollama"; other providers use hex tokens
        if key.lower() == "ollama":
            return True, "Using local Ollama — make sure Ollama is running on port 11434."
        if key.startswith("gsk_"):
            return True, "Groq API key format looks OK."
        return True, "Llama API key accepted."

    if len(key) < 10:
        return False, "API key seems too short. Please check it."

    return True, "Key format looks OK."

