#
# Parakeet TDT (NVIDIA) speech-to-text on ONNX Runtime, CPU only
# Same model as stt_parakeet_mlx.py (int8), for machines without Apple Silicon (Windows, Linux).
# ~0.4s per turn on a modern CPU.
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


class ParakeetSTTServiceONNX(SegmentedSTTService):
    """Transcribes each user turn with Parakeet TDT running on ONNX Runtime (CPU).

    Parakeet detects the language automatically; `language` is only used to tag
    the transcription frames.
    """

    def __init__(
        self,
        *,
        model: str = "nemo-parakeet-tdt-0.6b-v3",
        quantization: Optional[str] = "int8",
        language: Language = Language.ES,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.set_model_name(model)
        self._quantization = quantization
        self._language = language
        self._model = None
        self._load_task: Optional[asyncio.Future] = None
        # One inference at a time; ONNX Runtime already uses every CPU core for each one.
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="parakeet")

        self._settings = {"language": language, "engine": "parakeet-onnx"}

    def can_generate_metrics(self) -> bool:
        return True

    def _load(self):
        import onnx_asr

        self._model = onnx_asr.load_model(
            self.model_name, quantization=self._quantization, providers=["CPUExecutionProvider"]
        )
        # Warm up on a second of silence so the first turn isn't slower.
        self._transcribe(np.zeros(16000, dtype=np.float32))
        logger.info(f"Loaded Parakeet model {self.model_name} ({self._quantization}, ONNX CPU)")

    def _transcribe(self, audio_float: np.ndarray) -> str:
        return self._model.recognize(audio_float, sample_rate=self.sample_rate).strip()

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
