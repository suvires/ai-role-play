#
# Turns the LLM's leading emotion tag ("[alegre] ...") into an avatar expression
#

import re

from loguru import logger

from pipecat.frames.frames import (
    Frame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
    StartInterruptionFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.processors.frameworks.rtvi import RTVIServerMessageFrame

# Tag the prompt asks for -> VRM expression preset
EMOTIONS = {
    "neutral": "neutral",
    "alegre": "happy",
    "relajado": "relaxed",
    "triste": "sad",
    "enfadado": "angry",
    "sorprendido": "surprised",
}

_TAG = re.compile(r"^\s*\[\s*([a-záéíóú]+)\s*\]\s*", re.IGNORECASE)
_STRAY_TAG = re.compile(r"\[\s*(?:" + "|".join(EMOTIONS) + r")\s*\]\s*", re.IGNORECASE)
_MAX_TAG_LEN = 20


class EmotionTagProcessor(FrameProcessor):
    """Place between the LLM and the TTS.

    Strips the emotion tag the LLM writes at the start of each response, so the TTS
    doesn't read it out, and sends it to the client as an RTVI server message:
    {"type": "emotion", "emotion": "<VRM expression>"}.
    """

    def __init__(self):
        super().__init__()
        self._buffer = ""
        self._at_start = False

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, LLMFullResponseStartFrame):
            self._buffer, self._at_start = "", True
        elif isinstance(frame, StartInterruptionFrame):
            self._buffer, self._at_start = "", False
        elif isinstance(frame, LLMTextFrame):
            if self._at_start:
                await self._handle_start_text(frame.text)
            else:
                await self._push_text(frame.text)
            return
        elif isinstance(frame, LLMFullResponseEndFrame) and self._buffer:
            await self._push_text(self._buffer)  # response ended before a full tag
            self._buffer, self._at_start = "", False

        await self.push_frame(frame, direction)

    async def _handle_start_text(self, text: str):
        self._buffer += text
        stripped = self._buffer.lstrip()
        if not stripped:
            return
        if not stripped.startswith("["):
            self._at_start = False
        elif "]" in stripped:
            self._at_start = False
            match = _TAG.match(self._buffer)
            if match:
                await self._send_emotion(match.group(1).lower())
                self._buffer = self._buffer[match.end() :]
        elif len(stripped) <= _MAX_TAG_LEN:
            return  # still inside the tag, wait for more tokens
        else:
            self._at_start = False

        text, self._buffer = self._buffer, ""
        await self._push_text(text)

    async def _push_text(self, text: str):
        text = _STRAY_TAG.sub("", text)  # models sometimes repeat a tag mid-response
        if text:
            await self.push_frame(LLMTextFrame(text))

    async def _send_emotion(self, tag: str):
        emotion = EMOTIONS.get(tag)
        if emotion is None:
            logger.debug(f"Unknown emotion tag: [{tag}]")
            return
        logger.debug(f"Emotion: {tag} -> {emotion}")
        await self.push_frame(RTVIServerMessageFrame(data={"type": "emotion", "emotion": emotion}))
