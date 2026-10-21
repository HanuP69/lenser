import base64
import re
from typing import Any, Optional

from langchain_core.messages import HumanMessage

CONFIDENCE_FOOTER_PATTERN = re.compile(r"\n?\s*CONFIDENCE\s*:\s*([0-9]*\.?[0-9]+)\s*$", re.IGNORECASE)


def build_image_message(prompt_text: str, image_bytes: bytes, mime_type: str) -> HumanMessage:
    encoded = base64.b64encode(image_bytes).decode("utf-8")
    data_uri = f"data:{mime_type};base64,{encoded}"
    content_blocks = [
        {"type": "text", "text": prompt_text},
        {"type": "image_url", "image_url": {"url": data_uri}}
    ]
    return HumanMessage(content=content_blocks)


def response_to_text(response: Any) -> str:
    content = getattr(response, "content", response)
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        text_parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                text_parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                text_parts.append(str(block.get("text", "")))
        return "\n".join(text_parts).strip()
    return str(content).strip()


def extract_provider_name(response: Any, default_name: str = "unknown") -> str:
    additional = getattr(response, "additional_kwargs", None)
    if isinstance(additional, dict):
        provider = additional.get("lecturelens_provider")
        if provider:
            return str(provider)
    return default_name


def parse_confidence_footer(text: str, default_confidence: float = 0.5) -> tuple[str, float]:
    match = CONFIDENCE_FOOTER_PATTERN.search(text)
    if match is None:
        return text.strip(), default_confidence
    try:
        confidence = float(match.group(1))
    except ValueError:
        return text.strip(), default_confidence
    if confidence > 1.0:
        confidence = confidence / 100.0 if confidence <= 100.0 else 1.0
    # The pattern cannot match a leading minus sign, so a negative confidence
    # is never parsed. The clamp below was unreachable.
    cleaned_text = text[:match.start()].strip()
    return cleaned_text, confidence
