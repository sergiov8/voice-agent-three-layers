"""PCM helpers for the Live API: 16 kHz, 16-bit, mono, little-endian, sent in ~100 ms chunks.

Format requirements come from the Live transcription guide:
https://ai.google.dev/gemini-api/docs/live-api/live-transcribe
"""

import asyncio
import math
import wave
from array import array
from collections.abc import AsyncIterator

RATE = 16_000
CHUNK_MS = 100
CHUNK_BYTES = RATE * 2 * CHUNK_MS // 1000  # 3,200 bytes of 16-bit mono audio.
MIME_TYPE = f"audio/pcm;rate={RATE}"


def wav_duration_seconds(path: str) -> float:
  with wave.open(path, "rb") as w:
    return w.getnframes() / w.getframerate()


async def read_pcm_chunks(path: str, *, realtime: bool = True) -> AsyncIterator[bytes]:
  """Yields 100 ms PCM chunks from a 16 kHz mono 16-bit WAV file.

  With realtime=True the chunks are paced like a live microphone.
  Convert other audio first, for example:
    ffmpeg -i input.mp3 -ar 16000 -ac 1 -sample_fmt s16 output.wav
  """
  with wave.open(path, "rb") as w:
    if (w.getframerate(), w.getnchannels(), w.getsampwidth()) != (RATE, 1, 2):
      raise ValueError(
          f"{path} must be 16 kHz mono 16-bit PCM, got {w.getframerate()} Hz, "
          f"{w.getnchannels()} channel(s), {w.getsampwidth() * 8}-bit")
    while chunk := w.readframes(CHUNK_BYTES // 2):
      yield chunk
      if realtime:
        await asyncio.sleep(CHUNK_MS / 1000)


def rms(chunk: bytes) -> float:
  """Root mean square amplitude of a 16-bit little-endian PCM chunk."""
  samples = array("h", chunk)
  if not samples:
    return 0.0
  return math.sqrt(sum(s * s for s in samples) / len(samples))


class EndOfSpeechDetector:
  """Energy-based client-side VAD that decides when the speaker has stopped.

  The server keeps detecting the start of speech. This detector only ends the turn early
  so the client can send audio_stream_end instead of waiting for the server's silence timeout.
  Tune threshold for your audio. Telephony audio usually needs a lower value.
  """

  def __init__(self, *, threshold: float = 500.0, silence_ms: int = 600):
    self.threshold = threshold
    self.silence_ms = silence_ms
    self._speaking = False
    self._quiet_ms = 0

  def feed(self, chunk: bytes, chunk_ms: int = CHUNK_MS) -> bool:
    """Returns True exactly once per utterance, when trailing silence reaches silence_ms."""
    if rms(chunk) >= self.threshold:
      self._speaking, self._quiet_ms = True, 0
      return False
    if not self._speaking:
      return False
    self._quiet_ms += chunk_ms
    if self._quiet_ms >= self.silence_ms:
      self._speaking, self._quiet_ms = False, 0
      return True
    return False
