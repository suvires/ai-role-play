#
# Lets the LLM see the user's camera when the user asks about something visual
#

import base64
import io
import re
import time
import unicodedata
from typing import Optional

from loguru import logger
from PIL import Image

from pipecat.frames.frames import Frame, UserImageRawFrame
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContextFrame
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.anthropic.llm import AnthropicLLMContext

# Phrases (accent-insensitive) that suggest the user wants the bot to look.
_VISUAL_CUES = re.compile(
    # Not "esta/estas" (está/estás) or "ver" ("a ver"): too common outside visual questions.
    r"\b(mira\w*|ves|veo|viendo|vista|camara|esto|aqui"
    r"|mano\w*|dedo\w*|cuant[oa]s|tengo puest\w*|llevo|llevando|vestid\w*|ropa|camiseta|gafas|color\w*"
    r"|detras|delante|fondo|cara|pelo|aspecto|parezco|lee|leer|pone|ensen\w*|muestro)\b"
)


_IMAGE_TYPES = ("image_url", "image")  # OpenAI, Anthropic


def _text_of(content) -> Optional[str]:
    """Plain text of a message whose content is a string or only text blocks, else None."""
    if isinstance(content, str):
        return content
    if isinstance(content, list) and all(part.get("type") == "text" for part in content):
        return " ".join(part["text"] for part in content)
    return None


def _strip_images(messages: list):
    """Replace image+text user messages with just their text."""
    for message in messages:
        content = message.get("content")
        if isinstance(content, list) and any(part.get("type") in _IMAGE_TYPES for part in content):
            message["content"] = " ".join(
                part["text"] for part in content if part.get("type") == "text"
            )


def _normalize(text: str) -> str:
    text = unicodedata.normalize("NFD", text.lower())
    return "".join(c for c in text if unicodedata.category(c) != "Mn")


class CameraVisionProcessor(FrameProcessor):
    """Keeps the latest camera frame and attaches it to visual questions (for LLMs that
    can't reliably use a camera tool, like the small local models).

    Place it between the user context aggregator and the LLM. Camera frames
    (UserImageRawFrame, ~30 per second) are stored and not forwarded. When a
    context is about to go to the LLM and the last user message looks like a
    visual question (or `always=True`), the latest frame is added to that same
    message, so user/assistant turns keep alternating.

    Images cost ~1.4s of extra LLM latency on Gemma 4 E4B, hence the cue check.
    Once an image has been sent, follow-ups within `follow_up_secs` ("¿y ahora?")
    get a fresh frame too. Older images are removed from the history so the LLM
    never answers from a stale frame.
    """

    def __init__(
        self,
        *,
        auto_attach: bool = True,
        always: bool = False,
        follow_up_secs: float = 45.0,
        max_age_secs: float = 2.0,
        max_side: int = 640,
    ):
        """auto_attach=False: only swallow the camera stream (e.g. when the LLM has a camera tool)."""
        super().__init__()
        self._auto_attach = auto_attach
        self._always = always
        self._follow_up_secs = follow_up_secs
        self._max_age_secs = max_age_secs
        self._max_side = max_side
        self._latest: Optional[UserImageRawFrame] = None
        self._latest_time = 0.0
        self._last_attach_time = float("-inf")

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, UserImageRawFrame) and frame.request is None:
            self._latest = frame
            self._latest_time = time.monotonic()
            return

        if isinstance(frame, OpenAILLMContextFrame) and self._auto_attach:
            self._maybe_attach_image(frame.context)

        await self.push_frame(frame, direction)

    def has_recent_frame(self) -> bool:
        return self._latest is not None and time.monotonic() - self._latest_time <= self._max_age_secs

    def _maybe_attach_image(self, context):
        messages = context.messages
        if not messages or messages[-1].get("role") != "user":
            return
        text = _text_of(messages[-1].get("content"))
        if text is None:
            return  # already has an image (or isn't plain text)

        _strip_images(messages[:-1])

        now = time.monotonic()
        follow_up = now - self._last_attach_time < self._follow_up_secs
        if not self._always and not follow_up and not _VISUAL_CUES.search(_normalize(text)):
            return
        if not self.has_recent_frame():
            logger.debug("Visual question but no recent camera frame")
            return

        jpeg_b64 = self._encode(self._latest)
        if isinstance(context, AnthropicLLMContext):
            image = {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": jpeg_b64}}
        else:
            image = {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + jpeg_b64}}
        messages[-1]["content"] = [{"type": "text", "text": text}, image]
        self._last_attach_time = now
        logger.debug(f"Attached camera frame {self._latest.size} to: [{text.strip()}]")

    def _encode(self, frame: UserImageRawFrame) -> str:
        image = Image.frombytes(frame.format, frame.size, frame.image)
        image.thumbnail((self._max_side, self._max_side))
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=80)
        return base64.b64encode(buffer.getvalue()).decode()
