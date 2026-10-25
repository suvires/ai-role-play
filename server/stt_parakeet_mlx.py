#
# Parakeet TDT (NVIDIA) speech-to-text on MLX
# ~6x faster than Whisper large-v3-turbo on Apple Silicon, multilingual (25 European languages)
#

import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import AsyncGenerator, Optional

import numpy as np
from loguru import logger

from pipecat.frames.frames import ErrorFrame, Frame, StartFrame, TranscriptionFrame
from pipecat.services.stt_service import SegmentedSTTService
from pipecat.transcriptions.language import Language
from pipecat.utils.time import time_now_iso8601


class ParakeetSTTServiceMLX(SegmentedSTTService):
    """Transcribes each user turn with Parakeet TDT running on MLX.

    Parakeet detects the language automatically; `language` is only used to tag
    the transcription frames.
    """

    def __init__(
        self,
        *,
        model: str = "mlx-community/parakeet-tdt-0.6b-v3",
        language: Language = Language.ES,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.set_model_name(model)
        self._language = language
        self._model = None
        self._load_task: Optional[asyncio.Future] = None
        # MLX arrays are bound to the thread that created them, so all MLX work
        # (loading and inference) runs on one dedicated thread.
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="parakeet")

        self._settings = {"language": language, "engine": "parakeet-mlx"}

    def can_generate_metrics(self) -> bool:
        return True

    def _load(self):
        from parakeet_mlx import from_pretrained

        self._model = from_pretrained(self.model_name)
        # The first inference compiles kernels (~0.6s); do it now on a second of silence.
        self._transcribe(np.zeros(self._model.preprocessor_config.sample_rate, dtype=np.float32))
        logger.info(f"Loaded Parakeet model {self.model_name}")

    def _transcribe(self, audio_float: np.ndarray) -> str:
        import mlx.core as mx
        from parakeet_mlx.audio import get_logmel

        mel = get_logmel(mx.array(audio_float), self._model.preprocessor_config)
        return self._model.generate(mel)[0].text.strip()

    def _ensure_loaded(self) -> asyncio.Future:
        if self._load_task is None:
            loop = asyncio.get_running_loop()
            self._load_task = loop.run_in_executor(self._executor, self._load)
        return self._load_task

    async def start(self, frame: StartFrame):
        """Load the model in the background so the first turn doesn't pay for it."""
        await super().start(frame)
        self._ensure_loaded()

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame, None]:
        try:
            await self._ensure_loaded()

            await self.start_processing_metrics()
            await self.start_ttfb_metrics()

            # Divide by 32768 because we have signed 16-bit data.
            audio_float = np.frombuffer(audio, dtype=np.int16).astype(np.float32) / 32768.0
            loop = asyncio.get_running_loop()
            text = await loop.run_in_executor(self._executor, self._transcribe, audio_float)

            await self.stop_ttfb_metrics()
            await self.stop_processing_metrics()

            if text:
                logger.debug(f"Transcription: [{text}]")
                yield TranscriptionFrame(text, self._user_id, time_now_iso8601(), self._language)

        except Exception as e:
            logger.exception(f"Parakeet transcription error: {e}")
            yield ErrorFrame(f"Parakeet transcription error: {str(e)}")
