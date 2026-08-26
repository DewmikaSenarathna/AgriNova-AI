"""
image_model_tool.py
====================
PHASE 9 — "Image Agent -> Image Model" tool.

Turns a farmer-submitted crop photo into a plain-language description
of what's visible (leaf discoloration, spots, wilting, insect damage,
growth stage...) using a vision-capable LLM, via
`LLMClient.generate_vision()` (see ../../RAG-Pipeline/llm_client.py).

This tool deliberately does NOT try to output a diagnosis itself —
that's what Disease/Pest Agent are for, using the shared knowledge
base. Keeping this tool's job to "describe what the image shows" (not
"decide what disease this is") means:
  - it stays useful even when the knowledge base has no matching
    disease/pest entry,
  - and its output is explicitly `grounded=False` model inference
    (never presented to the farmer as a citable source), matching the
    project's rule of being honest about what's evidence-backed.

Basic validation (is this actually image data, is it a reasonable
size) is done with the standard library only — no extra dependency —
since the point of this file is orchestration, not image processing.
"""

import base64
import binascii
import logging

import agent_config
from tools.base_tool import BaseTool
from tools.tool_types import ToolResult
from rag_bridge import LLMClient, LLMError

logger = logging.getLogger(__name__)

# Minimal magic-byte sniffing so a clearly-non-image upload fails fast
# with a clear error instead of being sent to the vision model.
_MAGIC_BYTES = {
    b"\xff\xd8\xff": "image/jpeg",
    b"\x89PNG\r\n\x1a\n": "image/png",
    b"GIF87a": "image/gif",
    b"GIF89a": "image/gif",
    b"RIFF": "image/webp",  # (WEBP container; good enough for a sanity check)
}

_SYSTEM_PROMPT = (
    "You are AgriNova AI's Image Agent. You are shown a photo a farmer submitted, "
    "usually of a crop, leaf, stem, fruit, or pest. Describe ONLY what is visibly "
    "present in the image: plant part shown, color/texture abnormalities, spots, "
    "lesions, wilting, discoloration, holes, webbing, visible insects, growth stage. "
    "Do NOT name a specific disease or pest species and do NOT recommend treatment — "
    "that judgement belongs to the Disease/Pest specialists who will read your "
    "description alongside the knowledge base. If the image is unclear, blurry, or "
    "not plant-related, say so plainly instead of guessing."
)


class ImageModelTool(BaseTool):
    name = "image_model"
    description = (
        "Describes what's visible in a farmer-submitted crop photo (symptoms, "
        "affected plant part, visible pests) using a vision-capable LLM."
    )

    def __init__(self, llm: LLMClient = None):
        self.llm = llm or LLMClient()

    @staticmethod
    def _sniff_image_mime(raw: bytes) -> str:
        for magic, mime in _MAGIC_BYTES.items():
            if raw.startswith(magic):
                return mime
        return ""

    def run(self, image_base64: str, question: str = "") -> ToolResult:
        if not image_base64:
            return ToolResult(ok=False, error="No image was provided.")

        # Strip a data: URL prefix if the frontend sent one whole, e.g.
        # "data:image/jpeg;base64,/9j/4AAQ..." — keep only the payload.
        if image_base64.strip().startswith("data:") and ";base64," in image_base64:
            image_base64 = image_base64.split(";base64,", 1)[1]

        try:
            raw = base64.b64decode(image_base64, validate=True)
        except (binascii.Error, ValueError) as e:
            return ToolResult(ok=False, error=f"Image data is not valid base64: {e}")

        if len(raw) < 64:
            return ToolResult(ok=False, error="Image data is too small to be a real photo.")
        if len(raw) > agent_config.IMAGE_MAX_BYTES:
            return ToolResult(
                ok=False,
                error=(
                    f"Image is larger than the {agent_config.IMAGE_MAX_BYTES // (1024 * 1024)}MB "
                    f"limit — please resize/compress it and try again."
                ),
            )

        mime = self._sniff_image_mime(raw)
        if not mime:
            return ToolResult(
                ok=False,
                error="This doesn't look like a supported image (expected JPEG, PNG, GIF, or WEBP).",
            )

        user_prompt = (
            f"FARMER'S QUESTION (for context only, don't answer it directly): "
            f"{question or '(no question text given, just describe the photo)'}\n\n"
            "Describe exactly what is visible in this photo."
        )

        try:
            description = self.llm.generate_vision(
                _SYSTEM_PROMPT, user_prompt, image_base64=image_base64, image_mime=mime
            )
        except LLMError as e:
            return ToolResult(
                ok=False,
                error=(
                    f"Could not reach a vision-capable model: {e}. See "
                    f"RAG-Pipeline/.env.example's vision-model settings "
                    f"(OLLAMA_VISION_MODEL / GROQ_VISION_MODEL / "
                    f"OPENAI_COMPATIBLE_VISION_MODEL)."
                ),
            )

        return ToolResult(
            ok=True,
            data={"description": description, "mime_type": mime, "size_bytes": len(raw)},
            text=description,
            source={"source": "Image Model (vision LLM)", "grounded": False},
        )
