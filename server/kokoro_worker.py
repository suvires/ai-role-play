#!/usr/bin/env python3
"""
Standalone Kokoro TTS worker process.

This worker runs in complete isolation to avoid Metal threading conflicts.
It communicates via JSON over stdin/stdout.

Usage:
    python kokoro_worker.py

Commands:
    {"cmd": "init", "model": "mlx-community/Kokoro-82M-bf16", "voice": "af_heart"}
    {"cmd": "generate", "text": "Hello world"}
"""

import os
import sys
import json

# stdout is the JSON channel to the bot: keep it for our responses only and send everything
# else libraries print (e.g. pip installing a spaCy model on first English use) to stderr
_responses = os.fdopen(os.dup(sys.stdout.fileno()), "w", buffering=1)
os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
sys.stdout = sys.stderr
import base64
import traceback
import numpy as np

# Add logging to worker
import logging
logging.basicConfig(level=logging.INFO, format='WORKER: %(message)s')

try:
    import mlx.core as mx
    from mlx_audio.tts.utils import load_model
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False


class Worker:
    def __init__(self):
        self.model = None
        self.voice = None
        
    def initialize(self, model_name, voice):
        if not MLX_AVAILABLE:
            return {"error": "MLX not available"}
        try:
            self.model = load_model(model_name)
            self.voice = voice
            # Kokoro voices are prefixed with their language code (e.g. "ef_dora" -> "e" for Spanish)
            self.lang_code = voice[0] if voice else "a"
            # Test generation to ensure everything works
            list(self.model.generate(text="test", voice=voice, speed=1.0, lang_code=self.lang_code))
            return {"success": True}
        except Exception as e:
            return {"error": str(e)}
    
    def generate(self, text, voice=None):
        try:
            if not self.model:
                return {"error": "Not initialized"}

            # A voice per request lets one warmed-up worker serve any character
            voice = voice or self.voice
            lang_code = voice[0] if voice else self.lang_code
            segments = []
            for result in self.model.generate(text=text, voice=voice, speed=1.0, lang_code=lang_code):
                # Convert MLX array to numpy immediately
                audio_data = np.array(result.audio, copy=True)
                print(f"Generated segment shape: {audio_data.shape}, min: {audio_data.min():.4f}, max: {audio_data.max():.4f}", file=sys.stderr)
                segments.append(audio_data)
            
            if not segments:
                return {"error": "No audio"}
                
            # Concatenate all segments
            if len(segments) == 1:
                audio = segments[0]
            else:
                audio = np.concatenate(segments, axis=0)
            
            print(f"Final audio shape: {audio.shape}, min: {audio.min():.4f}, max: {audio.max():.4f}", file=sys.stderr)
            
            # Check if audio is silent
            if np.max(np.abs(audio)) < 1e-6:
                return {"error": "Generated audio is silent"}
            
            # Convert to 16-bit PCM
            audio_int16 = (audio * 32767).astype(np.int16)
            audio_b64 = base64.b64encode(audio_int16.tobytes()).decode()
            
            return {"success": True, "audio": audio_b64}
        except Exception as e:
            import traceback
            return {"error": f"{str(e)}\n{traceback.format_exc()}"}


def main():
    """Main worker loop - reads commands from stdin, writes responses to stdout."""
    worker = Worker()
    
    for line in sys.stdin:
        try:
            req = json.loads(line.strip())
            if req["cmd"] == "init":
                resp = worker.initialize(req["model"], req["voice"])
            elif req["cmd"] == "generate":
                resp = worker.generate(req["text"], req.get("voice"))
            else:
                resp = {"error": "Unknown command"}
            print(json.dumps(resp), file=_responses, flush=True)
        except Exception as e:
            print(json.dumps({"error": str(e)}), file=_responses, flush=True)


if __name__ == "__main__":
    main()