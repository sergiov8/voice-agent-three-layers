"""Layer 3: hybrid VAD. The client ends the turn the moment it hears silence.

Server-side VAD alone waits out a silence timeout before finalizing, which sounds like
dead air on a call. Here a local energy detector sends audio_stream_end as soon as the
speaker stops, and the server's own VAD remains the fallback if it never fires.
The script prints how long the finalized transcript took after end of speech.

Example:
  python -m layer3_telephony.hybrid_vad_sender utterance.wav --threshold 400
"""

import argparse
import asyncio
import sys
import time

from google import genai
from google.genai import types

from common.pcm import MIME_TYPE, EndOfSpeechDetector, read_pcm_chunks

TRANSCRIBE_MODEL = "gemini-3.5-transcribe-live"


async def run(path: str, threshold: float, silence_ms: int) -> None:
  client = genai.Client()
  detector = EndOfSpeechDetector(threshold=threshold, silence_ms=silence_ms)
  end_of_speech_at: list[float] = []

  async def send(session) -> None:
    async for chunk in read_pcm_chunks(path):
      await session.send_realtime_input(audio=types.Blob(data=chunk, mime_type=MIME_TYPE))
      if detector.feed(chunk):
        end_of_speech_at.append(time.monotonic())
        print("[client] silence detected, sending audio_stream_end")
        # This sample handles one utterance per run. Check that your session accepts
        # more audio after audio_stream_end before reusing it for multi-turn calls.
        await session.send_realtime_input(audio_stream_end=True)
        return
    await session.send_realtime_input(audio_stream_end=True)

  async def receive(session) -> None:
    async for response in session.receive():
      content = response.server_content
      if content and content.input_transcription and content.input_transcription.text:
        lag = (f"{(time.monotonic() - end_of_speech_at[0]) * 1000:.0f} ms after end of speech"
               if end_of_speech_at else "server VAD finalized it")
        print(f"[final]  {content.input_transcription.text.strip()}  ({lag})")
        return

  config = types.LiveConnectConfig(
      response_modalities=["TEXT"],
      input_audio_transcription=types.AudioTranscriptionConfig(),
  )
  async with client.aio.live.connect(model=TRANSCRIBE_MODEL, config=config) as session:
    await asyncio.wait_for(asyncio.gather(send(session), receive(session)), timeout=120)


def main() -> int:
  parser = argparse.ArgumentParser(
      description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument("audio", help="16 kHz mono 16-bit WAV file with one utterance")
  parser.add_argument("--threshold", type=float, default=500.0,
                      help="RMS level treated as speech")
  parser.add_argument("--silence-ms", type=int, default=600,
                      help="Trailing silence that ends the turn")
  args = parser.parse_args()
  asyncio.run(run(args.audio, args.threshold, args.silence_ms))
  return 0


if __name__ == "__main__":
  sys.exit(main())
