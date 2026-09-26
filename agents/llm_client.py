"""
Central LLM access point.

Uses Google Gemini (gemini-1.5-flash) via langchain-google-genai — free tier,
natively multimodal (text + image in a single call), no separate CV/OCR
pipeline needed for the architecture diagrams.

If no GOOGLE_API_KEY is present, `llm_available()` returns False and every
agent module falls back to its deterministic simulated output instead of
raising an error. This keeps the demo fully runnable offline / before a key
is added.
"""

from __future__ import annotations
import os
import base64
from io import BytesIO
from typing import Optional, Type, TypeVar

from dotenv import load_dotenv
from pydantic import BaseModel
from PIL import Image

load_dotenv()

MODEL_NAME = os.getenv("LLM_MODEL", "gemini-1.5-flash")
_API_KEY = os.getenv("GOOGLE_API_KEY")

T = TypeVar("T", bound=BaseModel)

_llm = None


def llm_available() -> bool:
    """True if a Gemini API key is configured."""
    return bool(_API_KEY)


def _get_llm():
    global _llm
    if _llm is None:
        from langchain_google_genai import ChatGoogleGenerativeAI

        _llm = ChatGoogleGenerativeAI(
            model=MODEL_NAME,
            google_api_key=_API_KEY,
            temperature=0.2,
        )
    return _llm


def _image_to_data_url(image: Image.Image) -> str:
    buf = BytesIO()
    image.convert("RGB").save(buf, format="PNG")
    encoded = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


def call_structured(
    system_prompt: str,
    user_prompt: str,
    schema: Type[T],
    image: Optional[Image.Image] = None,
) -> T:
    """
    Call Gemini with a system+user prompt (optionally with an image attached),
    forcing the response into `schema` via LangChain's structured output.
    Raises if no API key is configured — callers should check
    `llm_available()` first and use their deterministic fallback otherwise.
    """
    if not llm_available():
        raise RuntimeError(
            "No GOOGLE_API_KEY configured — call the deterministic fallback instead."
        )

    llm = _get_llm().with_structured_output(schema)

    content = [{"type": "text", "text": user_prompt}]
    if image is not None:
        content.append(
            {"type": "image_url", "image_url": {"url": _image_to_data_url(image)}}
        )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": content},
    ]

    return llm.invoke(messages)
